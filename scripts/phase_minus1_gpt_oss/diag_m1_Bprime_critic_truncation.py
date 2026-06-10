"""Phase −1.B′ — MoE critic truncation audit (gpt-oss-20b).

Confirms the critic path handles the first MoE backbone:

  1. Enumerate EVERY list-valued attr in the gpt-oss config whose length ==
     num_hidden_layers; flag any that nla/models.py:_truncate_config_layers
     does NOT slice (it knows layer_types / sliding_window_pattern /
     no_rope_layers — gpt-oss may carry others).
  2. NLACriticModel.from_pretrained(nla_num_layers=K) on the real checkpoint:
     per kept layer, count router tensors and expert tensors (expect router +
     32 experts' worth of MLP weights per layer).
  3. _no_split_modules must contain the MoE decoder-block class — leaf-level
     FSDP wrap of 32 experts/layer is an NCCL storm.

Run (1×H100 or big-RAM CPU): python scripts/phase_minus1_gpt_oss/diag_m1_Bprime_critic_truncation.py
"""

import argparse
import json
import traceback
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoConfig

from nla.models import NLACriticModel, _truncate_config_layers

_KNOWN_SLICED = ("layer_types", "sliding_window_pattern", "no_rope_layers")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--layer-index", type=int, default=17, help="datagen extraction layer K")
    ap.add_argument("--device-map", default="cpu", help="cpu is fine; cuda:0 on the H100")
    ap.add_argument("--out", default="/data/logs/diag_m1_Bprime.json")
    args = ap.parse_args()
    result: dict = {}

    # --- 1. per-layer config attrs vs _truncate_config_layers coverage ----
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    n_layers = cfg.num_hidden_layers
    per_layer_attrs = []
    for k, v in vars(cfg).items():
        if isinstance(v, (list, tuple)) and len(v) == n_layers:
            per_layer_attrs.append(k)
    uncovered = [a for a in per_layer_attrs if a not in _KNOWN_SLICED]
    print(f"num_hidden_layers={n_layers}")
    print(f"per-layer config attrs: {per_layer_attrs}")
    print(f"NOT covered by _truncate_config_layers: {uncovered or 'none'}")
    result["per_layer_attrs"] = per_layer_attrs
    result["uncovered_attrs"] = uncovered

    # Dry-run the truncation on the config alone; transformers >=4.50 validates
    # len(layer_types)==num_hidden_layers at init, so re-instantiating from the
    # truncated dict is the same check from_pretrained would hit.
    trunc_ok = True
    try:
        _truncate_config_layers(cfg, args.layer_index + 1)
        type(cfg)(**{k: v for k, v in cfg.to_dict().items() if k != "transformers_version"})
        print(f"config truncation to {args.layer_index + 1} layers: re-validates OK")
    except Exception:
        traceback.print_exc()
        result["truncate_error"] = traceback.format_exc()
        trunc_ok = False

    # --- 2. real truncated load: router + experts per kept layer ----------
    load_ok, experts_ok, no_split_ok = False, False, False
    try:
        critic = NLACriticModel.from_pretrained(
            args.model, nla_num_layers=args.layer_index,
            torch_dtype=torch.bfloat16, device_map=args.device_map,
        )
        load_ok = True
        per_layer: dict[int, dict] = defaultdict(lambda: {"router": 0, "expert": 0, "other": 0})
        for name, _ in critic.named_parameters():
            if ".layers." not in name:
                continue
            li = int(name.split(".layers.")[1].split(".")[0])
            low = name.lower()
            kind = "router" if ("router" in low or "gate" in low) else (
                "expert" if "expert" in low else "other")
            per_layer[li][kind] += 1
        kept = sorted(per_layer)
        print(f"kept layers: {kept[0]}..{kept[-1]} (expect 0..{args.layer_index})")
        first = per_layer[kept[0]]
        print(f"layer-0 tensor counts: {dict(first)}")
        uniform = all(per_layer[li] == first for li in kept)
        print(f"uniform across kept layers: {uniform}")
        experts_ok = (kept == list(range(args.layer_index + 1))
                      and first["router"] > 0 and first["expert"] > 0 and uniform)
        result["kept_layers"] = [kept[0], kept[-1]]
        result["layer0_counts"] = dict(first)
        result["uniform"] = uniform

        nsm = critic._no_split_modules or []
        block_cls = type(critic.backbone.model.layers[0]).__name__
        no_split_ok = block_cls in nsm
        print(f"_no_split_modules={nsm}, decoder block class={block_cls} "
              f"({'covered' if no_split_ok else 'NOT covered — FSDP leaf-wrap storm'})")
        result["no_split_modules"] = list(nsm)
        result["block_class"] = block_cls
    except Exception:
        traceback.print_exc()
        result["load_error"] = traceback.format_exc()

    ok = trunc_ok and load_ok and experts_ok and no_split_ok
    result["verdict"] = ("PASS: truncation covers gpt-oss per-layer attrs; router+experts "
                         "survive per kept layer; MoE block in _no_split_modules") if ok else (
                         "GATE FAILED: see uncovered_attrs / load_error / no_split_modules above")
    print(f"\n{result['verdict']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise
