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
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer


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
    ap.add_argument("--inj-scale", type=float, default=5500.0,
                    help="L2 norm for the perturbation row (phase0 p50–p90 of layer-17 norms)")
    ap.add_argument("--out", default="/data/logs/diag_m1_A.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Embedding table ONLY, straight from safetensors on CPU. Do NOT
    # from_pretrained the model here: the MXFP4 quantizer runs CUDA packing
    # even with device_map="cpu", and the SGLang server already owns the GPU
    # (observed: client-side OOM with the server holding 69GiB).
    print("loading embed_tokens from safetensors (CPU)…")
    key = "model.embed_tokens.weight"
    idx = json.load(open(hf_hub_download(args.model, "model.safetensors.index.json")))
    shard_path = hf_hub_download(args.model, idx["weight_map"][key])
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        embed_w = f.get_tensor(key)  # [vocab, d] bf16

    # Use the REAL actor prompt + injection marker (㎡, single-token by
    # construction — verified by −1.D). A literal content char like "X" BPE-
    # merges with <concept> under Harmony and can't be found by token ID.
    from nla.datagen.stage3_build import _DEFAULT_ACTOR_TEMPLATE
    content = _DEFAULT_ACTOR_TEMPLATE.format(injection_char="㎡")
    input_ids = tok.apply_chat_template([{"role": "user", "content": content}],
                                        tokenize=True, add_generation_prompt=True)
    with torch.no_grad():
        base = embed_w[torch.tensor(input_ids, dtype=torch.long)].float().unsqueeze(0)  # [1, T, d]

    inj_id = 83806  # committed injection_token_cache.yaml entry for gpt-oss
    pos_candidates = [i for i, t in enumerate(input_ids) if t == inj_id]
    assert len(pos_candidates) == 1, (
        f"marker {inj_id} appears {len(pos_candidates)}× (expected 1); ids={input_ids[:50]}…")
    pos = pos_candidates[0]
    # Replace the marker row with a random vector at realistic injection scale
    # (phase0: layer-17 extracted-position norms p50–p90 ≈ 5447–5901).
    g = torch.Generator().manual_seed(42)
    v = torch.randn(base.shape[-1], generator=g)
    perturbed = base.clone()
    perturbed[0, pos] = v * (args.inj_scale / v.norm())
    print(f"T={len(input_ids)} d={base.shape[-1]} perturbed pos={pos} "
          f"(row_norm {base[0, pos].norm().item():.2f} → {perturbed[0, pos].norm().item():.2f})")

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
