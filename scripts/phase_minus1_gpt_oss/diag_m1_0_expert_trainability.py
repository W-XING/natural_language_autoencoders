"""Phase −1.0 — actor expert trainability under MXFP4 (gpt-oss-20b).

Loads openai/gpt-oss-20b exactly as the actor does (plain
AutoModelForCausalLM.from_pretrained, nla/train_actor.py NLATextOnlyCausalLM
shim → no quantization handling), then reports:

  1. quantization_config as transformers resolved it (Mxfp4Config: packed vs
     dequantized).
  2. Every expert/router tensor: parameter-vs-buffer, dtype, requires_grad.
  3. A 1-step forward+backward: which expert tensors actually receive .grad.
  4. _no_split_modules (feeds −1.B′'s FSDP wrap-granularity check).

Gate: if expert MLPs are buffers (requires_grad=False) or get no grad, SFT/RL
would silently train only attention/router/embeddings — research decision
needed (train-attn-only vs dequant-to-trainable vs abort).

Run (1×H100): python scripts/phase_minus1_gpt_oss/diag_m1_0_expert_trainability.py
"""

import argparse
import json
import traceback
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EXPERT_PAT = ("expert", "router", "gate")  # substring match on tensor names


def classify(name: str) -> str:
    low = name.lower()
    for p in EXPERT_PAT:
        if p in low:
            return p
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--dtype", default="bfloat16",
                    help="actor path passes bf16; 'auto' shows HF's own choice")
    ap.add_argument("--attn", default=None,
                    help="attn_implementation override (actor: miles passes its own)")
    ap.add_argument("--out", default="/data/logs/diag_m1_0.json")
    args = ap.parse_args()

    kwargs: dict = {"trust_remote_code": True}
    if args.dtype != "auto":
        kwargs["torch_dtype"] = getattr(torch, args.dtype)
    else:
        kwargs["torch_dtype"] = "auto"
    if args.attn:
        kwargs["attn_implementation"] = args.attn

    print(f"loading {args.model} kwargs={kwargs} …")
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.cuda()

    qc = getattr(model.config, "quantization_config", None)
    print(f"\nquantization_config: {qc}")
    print(f"_no_split_modules: {model._no_split_modules}")

    # --- inventory: parameters vs buffers --------------------------------
    report: dict = {
        "quantization_config": repr(qc),
        "no_split_modules": list(model._no_split_modules or []),
        "params": defaultdict(lambda: {"n_tensors": 0, "n_elem": 0,
                                       "requires_grad": set(), "dtypes": set()}),
        "buffers": defaultdict(lambda: {"n_tensors": 0, "n_elem": 0, "dtypes": set()}),
    }
    sample_names: dict[str, list[str]] = defaultdict(list)
    for name, p in model.named_parameters():
        c = classify(name)
        e = report["params"][c]
        e["n_tensors"] += 1
        e["n_elem"] += p.numel()
        e["requires_grad"].add(bool(p.requires_grad))
        e["dtypes"].add(str(p.dtype))
        if len(sample_names[f"param/{c}"]) < 3:
            sample_names[f"param/{c}"].append(f"{name} {tuple(p.shape)} {p.dtype} rg={p.requires_grad}")
    for name, b in model.named_buffers():
        c = classify(name)
        e = report["buffers"][c]
        e["n_tensors"] += 1
        e["n_elem"] += b.numel()
        e["dtypes"].add(str(b.dtype))
        if len(sample_names[f"buffer/{c}"]) < 3:
            sample_names[f"buffer/{c}"].append(f"{name} {tuple(b.shape)} {b.dtype}")

    print("\n=== tensor inventory (by name class) ===")
    for kind in ("params", "buffers"):
        for c, e in sorted(report[kind].items()):
            print(f"{kind:8s} {c:8s} tensors={e['n_tensors']:5d} "
                  f"elems={e['n_elem']/1e9:7.3f}B dtypes={sorted(e['dtypes'])}"
                  + (f" requires_grad={sorted(e['requires_grad'])}" if kind == "params" else ""))
    print("\n=== sample tensor names ===")
    for k, names in sorted(sample_names.items()):
        for n in names:
            print(f"  {k}: {n}")

    # --- 1-step backward: do expert tensors receive grad? ----------------
    print("\n=== forward+backward grad probe ===")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = tok("The capital of France is Paris because", return_tensors="pt").input_ids.cuda()
    grad_result: dict[str, dict] = {}
    try:
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        got, none_ = defaultdict(int), defaultdict(int)
        for name, p in model.named_parameters():
            c = classify(name)
            if p.grad is not None and p.grad.abs().sum() > 0:
                got[c] += 1
            else:
                none_[c] += 1
        for c in sorted(set(got) | set(none_)):
            grad_result[c] = {"with_grad": got[c], "without_grad": none_[c]}
            print(f"  {c:8s} with_grad={got[c]:5d} without={none_[c]:5d}")
    except Exception:
        traceback.print_exc()
        grad_result = {"error": traceback.format_exc()}

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"\npeak CUDA mem: {peak:.1f} GiB")

    # --- verdict ----------------------------------------------------------
    ep = report["params"].get("expert", {"n_tensors": 0, "requires_grad": set()})
    experts_trainable = (
        ep["n_tensors"] > 0
        and ep["requires_grad"] == {True}
        and grad_result.get("expert", {}).get("with_grad", 0) > 0
    )
    verdict = "PASS: expert MLPs are trainable parameters and receive grads" if experts_trainable else (
        "GATE FAILED: expert MLPs are NOT trainable as the actor loads them — "
        "research decision needed (train-attn-only / dequant-to-bf16 / abort)")
    print(f"\n{verdict}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "verdict": verdict,
        "experts_trainable": experts_trainable,
        "quantization_config": repr(qc),
        "no_split_modules": report["no_split_modules"],
        "params": {c: {k: (sorted(v) if isinstance(v, set) else v) for k, v in e.items()}
                   for c, e in report["params"].items()},
        "buffers": {c: {k: (sorted(v) if isinstance(v, set) else v) for k, v in e.items()}
                    for c, e in report["buffers"].items()},
        "grad_probe": grad_result,
        "sample_names": dict(sample_names),
        "peak_cuda_gib": peak,
        "load_kwargs": {k: str(v) for k, v in kwargs.items()},
    }, indent=2))
    print(f"wrote {args.out}")
    raise SystemExit(0 if experts_trainable else 1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise
