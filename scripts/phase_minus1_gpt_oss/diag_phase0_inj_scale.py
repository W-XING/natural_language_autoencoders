"""Phase 0 — INJ_SCALE calibration for gpt-oss-20b (1×H100).

Forwards N UltraFineWeb docs through the base model, hooks the extraction
layer (output of block K — same convention as datagen/extractors.py), and
measures the residual-norm distribution AT THE POSITIONS STAGE-0 WOULD
EXTRACT (>= _MIN_POSITION, non-special). Reports p10/p50/p90/p99 and
proposes INJ_SCALE from p50–p90 of that distribution — NOT the Qwen
median×1.7 constant.

Also runs the advisory logit-lens KL probe over K ∈ {15, 17, 19}.

Run: python scripts/phase_minus1_gpt_oss/diag_phase0_inj_scale.py
"""

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.datagen.stage0_extract import _MIN_POSITION


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--layers", type=int, nargs="+", default=[15, 17, 19])
    ap.add_argument("--n-docs", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--out", default="/data/logs/diag_phase0_inj_scale.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()
    layers = model.model.layers
    norm_f, lm_head = model.model.norm, model.lm_head

    print(f"streaming {args.n_docs} UltraFineWeb docs…")
    ds = load_dataset("openbmb/Ultra-FineWeb", split="en", streaming=True)
    texts = [r["content"] for _, r in zip(range(args.n_docs), iter(ds))]

    norms: dict[int, list[float]] = {k: [] for k in args.layers}
    kl_sum: dict[int, float] = {k: 0.0 for k in args.layers}
    kl_n = 0
    special_ids = set(tok.all_special_ids)

    captured: dict[int, torch.Tensor] = {}

    def mk_hook(k):
        def hook(_m, _inp, out):
            captured[k] = out[0] if isinstance(out, tuple) else out
        return hook

    handles = [layers[k].register_forward_hook(mk_hook(k)) for k in args.layers]
    try:
        with torch.no_grad():
            for i, text in enumerate(texts):
                enc = tok(text, return_tensors="pt", truncation=True,
                          max_length=args.max_length)
                ids = enc.input_ids.cuda()
                out = model(input_ids=ids)
                # positions stage0 would sample: >= _MIN_POSITION, non-special
                valid = [p for p in range(ids.shape[1])
                         if p >= _MIN_POSITION and int(ids[0, p]) not in special_ids]
                if not valid:
                    continue
                pos = torch.tensor(valid, device=ids.device)
                final_logp = torch.log_softmax(out.logits[0, pos].float(), -1)
                for k in args.layers:
                    h = captured[k][0, pos]  # [P, d]
                    norms[k].extend(h.float().norm(dim=-1).tolist())
                    # logit-lens: final-LN + lm_head on the intermediate stream
                    lens_logp = torch.log_softmax(lm_head(norm_f(h)).float(), -1)
                    kl_sum[k] += torch.sum(
                        torch.exp(final_logp) * (final_logp - lens_logp)
                    ).item()
                kl_n += len(valid)
                if (i + 1) % 8 == 0:
                    print(f"  {i + 1}/{len(texts)} docs, {kl_n} positions")
    finally:
        for h in handles:
            h.remove()

    result: dict = {"n_docs": len(texts), "n_positions": kl_n, "layers": {}}
    print(f"\n{'K':>3} {'p10':>9} {'p50':>9} {'p90':>9} {'p99':>9} {'KL(final‖lens)':>15}")
    for k in args.layers:
        a = np.array(norms[k])
        p10, p50, p90, p99 = np.percentile(a, [10, 50, 90, 99])
        kl = kl_sum[k] / max(kl_n, 1)
        result["layers"][k] = {"p10": p10, "p50": p50, "p90": p90, "p99": p99,
                               "mean_kl_final_vs_lens": kl}
        print(f"{k:>3} {p10:9.1f} {p50:9.1f} {p90:9.1f} {p99:9.1f} {kl:15.4f}")

    k_smoke = 17
    p50, p90 = result["layers"][k_smoke]["p50"], result["layers"][k_smoke]["p90"]
    result["proposal"] = {
        "layer": k_smoke,
        "inj_scale_range": [p50, p90],
        "note": "set INJ_SCALE inside [p50, p90] of layer-17 extracted-position "
                "norms; record the choice in phase0_notes.md. KL column is "
                "advisory only — the real SWA test is the Phase-2 smoke "
                "loss-gap bucketed by response position.",
    }
    print(f"\nproposed INJ_SCALE for K=17: in [{p50:.0f}, {p90:.0f}]")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, default=float))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
