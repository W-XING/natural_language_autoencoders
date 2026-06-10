"""Phase −1.D — injection-char / neighbor round-trip through Harmony (CPU-only).

Cheapest catastrophic-bug catch. Verifies, against the live gpt-oss tokenizer:

  1. The committed cache entry (㎡ / 83806) still tokenizes to exactly that ID
     (find_injection_token re-verifies and would assert on drift).
  2. compute_canonical_neighbors survives apply_chat_template on the Harmony
     template — exactly ONE marker occurrence, not at a sequence edge.
  3. inject_at_marked_positions (the runtime hook check, nla/injection.py)
     finds the position with those neighbor IDs — i.e. datagen sidecar and
     training hook agree.
  4. compute_critic_suffix_ids works on the default critic template.

This is exactly the silent token-drift class CLAUDE.md warns about (wrong
position → model sees literal ㎡ → free-associates Chinese).

Run (CPU ok): python scripts/phase_minus1_gpt_oss/diag_m1_D_neighbor_roundtrip.py
"""

import argparse
import json
import traceback
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nla.datagen.injection_tokens import build_token_meta, find_injection_token
from nla.datagen.stage3_build import _DEFAULT_ACTOR_TEMPLATE, _DEFAULT_CRITIC_TEMPLATE
from nla.injection import inject_at_marked_positions

EXPECTED = {"char": "㎡", "token_id": 83806}  # committed injection_token_cache.yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--out", default="/data/logs/diag_m1_D.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    result: dict = {}

    # 1+2+4: cache re-verify + neighbors through Harmony + critic suffix.
    meta = build_token_meta(tok, _DEFAULT_ACTOR_TEMPLATE, _DEFAULT_CRITIC_TEMPLATE)
    char, tid = find_injection_token(tok)
    print(f"injection token: {char!r} id={tid} "
          f"(expected {EXPECTED['char']!r}/{EXPECTED['token_id']})")
    print(f"neighbors: left={meta.injection_left_neighbor_id} "
          f"({tok.decode([meta.injection_left_neighbor_id])!r}) "
          f"right={meta.injection_right_neighbor_id} "
          f"({tok.decode([meta.injection_right_neighbor_id])!r})")
    print(f"critic_suffix_ids: {meta.critic_suffix_ids} → "
          f"{tok.decode(meta.critic_suffix_ids)!r}")
    cache_ok = (char, tid) == (EXPECTED["char"], EXPECTED["token_id"])
    result["token"] = {"char": char, "id": tid, "matches_committed_cache": cache_ok}
    result["neighbors"] = {
        "left": meta.injection_left_neighbor_id,
        "right": meta.injection_right_neighbor_id,
        "left_decoded": tok.decode([meta.injection_left_neighbor_id]),
        "right_decoded": tok.decode([meta.injection_right_neighbor_id]),
    }
    result["critic_suffix"] = {"ids": meta.critic_suffix_ids,
                               "decoded": tok.decode(meta.critic_suffix_ids)}

    # 3: the runtime hook's scan must find the same position the sidecar implies.
    content = _DEFAULT_ACTOR_TEMPLATE.format(injection_char=char)
    ids = tok.apply_chat_template([{"role": "user", "content": content}],
                                  tokenize=True, add_generation_prompt=True)
    ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    d = 2880
    embeds = torch.zeros(1, len(ids), d)
    vec = torch.ones(1, d)
    hook_ok = False
    try:
        out = inject_at_marked_positions(
            ids_t, embeds, vec, tid,
            meta.injection_left_neighbor_id, meta.injection_right_neighbor_id,
        )
        n_injected = int((out.abs().sum(-1) > 0).sum())
        hook_ok = n_injected == 1
        pos = int((out.abs().sum(-1) > 0).nonzero()[0, 1])
        print(f"hook injected at exactly {n_injected} position (pos={pos}, T={len(ids)})")
        result["hook"] = {"n_injected": n_injected, "pos": pos, "T": len(ids)}
    except Exception:
        traceback.print_exc()
        result["hook_error"] = traceback.format_exc()

    ok = cache_ok and hook_ok
    result["verdict"] = ("PASS: marker/neighbors round-trip Harmony; hook and sidecar agree"
                         if ok else "GATE FAILED: token drift or hook/neighbor mismatch — "
                         "do NOT proceed to datagen")
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
