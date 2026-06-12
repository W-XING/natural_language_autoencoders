"""Phase-2 smoke gate: real-vs-random loss gap by response position + CJK leak.

Teacher-forces the SFT'd actor on held-out (explanation, activation_vector)
pairs twice — once with each row's TRUE vector injected at the ㎡ marker, once
with a MISMATCHED vector (derangement: roll by 1) — and buckets the per-token
NLL of the response span by position within the response. The gates this
implements (gpt-oss execution plan, Phase 2):

  1. real-vs-random gap > 0.1 nats overall — injection carries signal;
  2. the gap must NOT collapse in the 100-150 bucket vs the 0-50 bucket —
     gpt-oss alternates W=128 sliding-window layers, so response tokens past
     position ~128 can no longer attend directly to the marker. Collapse ⇒
     cap responses at 128 or move extraction layer K;
  3. CJK-leak rate < 1% of generated chars — if injection silently fails the
     actor sees the literal ㎡ and free-associates Chinese (CLAUDE.md smell).

Eval rows come from a stage-2 *_explained.parquet (columns activation_vector,
api_explanation) — use the AR split: the actor never trained on those rows.
The teacher-forced target is wrap_explanation(api_explanation), exactly the
stage-3 training target construction.

Usage:
    python -m nla.scripts.eval_av_gap \
        --checkpoint /workspace/checkpoints/smoke_av_sft/iter_0200_hf \
        --eval-parquet /workspace/data/nla_gpt_oss_20b_1k/splits/ar_sft_explained.parquet \
        --sidecar /workspace/data/nla_gpt_oss_20b_1k/av_sft_shuf.parquet \
        --injection-scale 5500 --output /workspace/logs/smoke_gap.json
"""

import argparse
import json
import re
import traceback
from pathlib import Path

import pyarrow.parquet as pq
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.injection import inject_at_marked_positions
from nla.models import mxfp4_dequantize_kwargs
from nla.schema import (
    compute_harmony_affixes,
    normalize_activation,
    resolve_target_scale,
    sidecar_path_for,
    wrap_explanation,
)

_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")


def _load_sidecar(path: str) -> dict:
    sc = sidecar_path_for(path)
    assert sc.exists(), f"sidecar not found: {sc}"
    return yaml.safe_load(sc.read_text())


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="HF checkpoint dir of the SFT'd actor")
    p.add_argument("--eval-parquet", required=True,
                   help="*_explained.parquet with activation_vector + api_explanation "
                        "(use the AR split — held out from actor SFT)")
    p.add_argument("--sidecar", required=True,
                   help="dataset sidecar source (parquet path; .nla_meta.yaml appended) "
                        "for injection token/template/d_model")
    p.add_argument("--injection-scale", required=True,
                   help="same value used in training (float, 'raw', 'sqrt_d_model')")
    p.add_argument("--n-rows", type=int, default=256)
    p.add_argument("--n-generate", type=int, default=32)
    p.add_argument("--max-response-tokens", type=int, default=150)
    p.add_argument("--bucket-edges", default="50,100,150")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", required=True, help="verdict JSON path")
    args = p.parse_args()

    meta = _load_sidecar(args.sidecar)
    tokens = meta["tokens"]
    inj_char = tokens["injection_char"]
    inj_id = tokens["injection_token_id"]
    left_id = tokens["injection_left_neighbor_id"]
    right_id = tokens["injection_right_neighbor_id"]
    actor_template = meta["prompt_templates"]["actor"]
    d_model = meta["extraction"]["d_model"]
    assert meta.get("actor_reasoning_mode") == "forced_final", (
        f"this eval implements the forced_final layout; sidecar says "
        f"{meta.get('actor_reasoning_mode')!r}"
    )
    scale = resolve_target_scale(args.injection_scale, d_model)

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    prefix_ids, close_ids = compute_harmony_affixes(tok)

    user_content = actor_template.format(injection_char=inj_char)
    head = tok.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=True, add_generation_prompt=True,
    )
    gen_prompt_ids = head + prefix_ids
    # sanity: marker present exactly once with canonical neighbors
    hits = [i for i in range(1, len(gen_prompt_ids) - 1)
            if gen_prompt_ids[i] == inj_id
            and gen_prompt_ids[i - 1] == left_id and gen_prompt_ids[i + 1] == right_id]
    assert len(hits) == 1, f"expected exactly 1 marker site in prompt, found {len(hits)}"

    table = pq.read_table(args.eval_parquet,
                          columns=["activation_vector", "api_explanation"])
    n = min(args.n_rows, table.num_rows)
    vecs = torch.tensor(table.column("activation_vector").to_pylist()[:n],
                        dtype=torch.float32)
    assert vecs.shape[1] == d_model, f"vector dim {vecs.shape[1]} != d_model {d_model}"
    targets = [wrap_explanation(e) for e in table.column("api_explanation").to_pylist()[:n]]
    vecs_norm = normalize_activation(vecs, scale)
    vecs_rand = torch.roll(vecs_norm, shifts=1, dims=0)  # derangement: wrong row's vector

    print(f"loading {args.checkpoint} …")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, device_map="auto",
        **mxfp4_dequantize_kwargs(args.checkpoint),
    )
    model.eval()
    embed = model.get_input_embeddings()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    def forward_nll(row_idx: list[int], vec_source: torch.Tensor):
        """→ list of (response_position, nll) over all rows in row_idx."""
        out = []
        for batch_idx in _batched(row_idx, args.batch_size):
            seqs, resp_starts = [], []
            for i in batch_idx:
                resp_ids = tok(targets[i], add_special_tokens=False)["input_ids"]
                resp_ids = resp_ids[: args.max_response_tokens]
                seqs.append(gen_prompt_ids + resp_ids + close_ids)
                resp_starts.append(len(gen_prompt_ids))
            maxlen = max(len(s) for s in seqs)
            input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
            attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for r, s in enumerate(seqs):
                input_ids[r, : len(s)] = torch.tensor(s)
                attn[r, : len(s)] = 1
            device = embed.weight.device
            input_ids, attn = input_ids.to(device), attn.to(device)
            embeds = embed(input_ids)
            v = vec_source[batch_idx].to(device=device, dtype=embeds.dtype)
            embeds = inject_at_marked_positions(
                input_ids, embeds, v, inj_id, left_id, right_id)
            with torch.no_grad():
                logits = model(inputs_embeds=embeds, attention_mask=attn,
                               use_cache=False).logits.float()
            logprobs = torch.log_softmax(logits[:, :-1], dim=-1)
            tgt = input_ids[:, 1:]
            nll = -logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [B, S-1]
            for r, (i, s) in enumerate(zip(batch_idx, seqs, strict=False)):
                rs = resp_starts[r]
                for pos in range(rs, len(s) - len(close_ids)):
                    # nll[r, pos-1] is the NLL of predicting token at pos
                    out.append((pos - rs, nll[r, pos - 1].item()))
        return out

    row_idx = list(range(n))
    print(f"teacher-forcing {n} rows × 2 conditions …")
    true_nll = forward_nll(row_idx, vecs_norm)
    rand_nll = forward_nll(row_idx, vecs_rand)

    edges = [int(x) for x in args.bucket_edges.split(",")]
    def bucketize(pairs):
        sums = [0.0] * len(edges); cnts = [0] * len(edges)
        for pos, v in pairs:
            for bi, e in enumerate(edges):
                if pos < e:
                    sums[bi] += v; cnts[bi] += 1
                    break
        return [s / c if c else float("nan") for s, c in zip(sums, cnts, strict=True)], cnts

    true_b, counts = bucketize(true_nll)
    rand_b, _ = bucketize(rand_nll)
    gap_b = [r - t for t, r in zip(true_b, rand_b, strict=True)]
    gap_overall = (sum(p[1] for p in rand_nll) / len(rand_nll)
                   - sum(p[1] for p in true_nll) / len(true_nll))

    print(f"generating {args.n_generate} samples for CJK check …")
    cjk_chars = total_chars = 0
    gen_texts = []
    device = embed.weight.device
    for batch_idx in _batched(row_idx[: args.n_generate], args.batch_size):
        input_ids = torch.tensor([gen_prompt_ids] * len(batch_idx), device=device)
        embeds = embed(input_ids)
        v = vecs_norm[batch_idx].to(device=device, dtype=embeds.dtype)
        embeds = inject_at_marked_positions(
            input_ids, embeds, v, inj_id, left_id, right_id)
        with torch.no_grad():
            gen = model.generate(
                inputs_embeds=embeds,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=args.max_response_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        for row in gen:
            text = tok.decode(row, skip_special_tokens=True)
            gen_texts.append(text)
            cjk_chars += len(_CJK_RE.findall(text))
            total_chars += max(len(text), 1)
    cjk_rate = cjk_chars / max(total_chars, 1)

    # collapse: late-bucket gap under half of early-bucket gap
    collapse = (len(gap_b) >= 3 and gap_b[0] == gap_b[0]  # not NaN
                and gap_b[2] == gap_b[2] and gap_b[2] < 0.5 * gap_b[0])
    verdict = {
        "checkpoint": args.checkpoint,
        "n_rows": n,
        "injection_scale": scale,
        "bucket_edges": edges,
        "bucket_token_counts": counts,
        "nll_true_by_bucket": true_b,
        "nll_random_by_bucket": rand_b,
        "gap_by_bucket": gap_b,
        "gap_overall": gap_overall,
        "cjk_rate": cjk_rate,
        "gen_samples": gen_texts[:5],
        "gates": {
            "gap_gt_0.1": gap_overall > 0.1,
            "no_positional_collapse": not collapse,
            "cjk_lt_1pct": cjk_rate < 0.01,
        },
    }
    verdict["pass"] = all(verdict["gates"].values())
    Path(args.output).write_text(json.dumps(verdict, indent=2))
    print(json.dumps({k: v for k, v in verdict.items() if k != "gen_samples"}, indent=2))
    print("VERDICT:", "PASS" if verdict["pass"] else "FAIL")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
