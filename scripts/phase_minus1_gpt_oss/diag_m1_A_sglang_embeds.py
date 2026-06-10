"""Phase −1.A — SGLang + gpt-oss + input_embeds: prove embeds are CONSUMED.

An HTTP-200 is insufficient — the generic SGLang path can accept input_embeds
and silently ignore them (that's why patches/ ships a gemma3_mm.py routing
patch). This spike posts TWO /generate payloads with identical shapes but
different embed values at one position and asserts greedy outputs DIFFER.

Outcomes:
  PASS            outputs differ → embeds consumed, RL transport viable.
  EMBEDS_IGNORED  outputs identical → write a gpt_oss routing patch (gemma3
                  pattern) or fall back to HF-generate rollouts (~5× slower).
  REJECTED        server 4xx/5xx on input_embeds → needs the NLA transport
                  patches (patches/apply_sglang_patches.sh) or the gpt_oss path
                  doesn't accept embeds at all.

Prereq (separate tmux pane):
  python -m sglang.launch_server --model openai/gpt-oss-20b \
      --disable-radix-cache --port 30000
Run: python scripts/phase_minus1_gpt_oss/diag_m1_A_sglang_embeds.py
"""

import argparse
import json
import traceback
from pathlib import Path

import httpx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def post_generate(url: str, embeds: list, max_new: int) -> dict:
    payload = {
        "input_embeds": embeds,
        "sampling_params": {
            "temperature": 0.0,  # greedy — determinism is what makes the diff meaningful
            "max_new_tokens": max_new,
        },
    }
    r = httpx.post(f"{url}/generate", json=payload, timeout=300.0)
    r.raise_for_status()
    return r.json()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out", default="/data/logs/diag_m1_A.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Embedding table only — load on CPU, slice the rows we need.
    print("loading embedding table (CPU)…")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    embed = model.get_input_embeddings()

    msgs = [{"role": "user", "content": "Briefly describe the following concept: <concept>X</concept>"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tok.encode(prompt, add_special_tokens=False)
    ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        base = embed(ids_t).float()  # [1, T, d]

    # Perturb ONE mid-sequence position (the 'X' inside <concept>) by replacing
    # its row with a large-norm random vector — same scale class as injection.
    x_id = tok.encode("X", add_special_tokens=False)
    pos_candidates = [i for i, t in enumerate(input_ids) if t in x_id and i > 5]
    assert pos_candidates, f"couldn't find perturbation position; ids={input_ids[:50]}…"
    pos = pos_candidates[-1]
    row_norm = base[0, pos].norm().item()
    g = torch.Generator().manual_seed(42)
    perturbed = base.clone()
    perturbed[0, pos] = torch.randn(base.shape[-1], generator=g) * (row_norm * 10 / base.shape[-1] ** 0.5)
    print(f"T={len(input_ids)} d={base.shape[-1]} perturbed pos={pos} "
          f"(row_norm {row_norm:.2f} → {perturbed[0, pos].norm().item():.2f})")

    result: dict = {"pos": pos, "n_tokens": len(input_ids)}
    try:
        out_a = post_generate(args.url, base[0].tolist(), args.max_new_tokens)
        out_b = post_generate(args.url, perturbed[0].tolist(), args.max_new_tokens)
        text_a, text_b = out_a.get("text", ""), out_b.get("text", "")
        print(f"\nbaseline : {text_a!r}\nperturbed: {text_b!r}")
        result |= {"text_baseline": text_a, "text_perturbed": text_b}
        if text_a != text_b:
            result["verdict"] = "PASS: differing embeds → differing greedy outputs (embeds consumed)"
            code = 0
        else:
            result["verdict"] = ("EMBEDS_IGNORED: identical greedy outputs — gpt_oss path drops "
                                 "input_embeds; needs a routing patch or HF-generate fallback")
            code = 1
    except httpx.HTTPStatusError as e:
        traceback.print_exc()
        result["verdict"] = f"REJECTED: {e.response.status_code} {e.response.text[:500]}"
        code = 1
    except Exception:
        traceback.print_exc()
        result["verdict"] = f"ERROR: {traceback.format_exc()[-500:]}"
        code = 1

    print(f"\n{result['verdict']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")
    raise SystemExit(code)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise
