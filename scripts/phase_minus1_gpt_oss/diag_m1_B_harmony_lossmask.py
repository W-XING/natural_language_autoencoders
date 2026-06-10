"""Phase −1.B — Harmony loss-mask + channels (gpt-oss tokenizer, CPU-only).

Three checks against the Harmony chat template:

  1. MultiTurnLossMaskGenerator(tok, "generic")._turn_close_ids() — patch
     0001's contiguous-subsequence assert (mask_utils.py) is expected to fire
     on Harmony's <|channel|>final<|message|>…<|return|> structure. Report
     close_ids if it survives, the assert text if it doesn't.
  2. gen_multi_turn_loss_mask on a (user, assistant) pair — does the masked
     region cover exactly the assistant content?
  3. extract_explanation() on a response that carries an `analysis` channel
     before the `final` channel — the reasoning channel must not hijack the
     <explanation> payload (nla/schema.py EXPLANATION_RE).

Gate: failures here mean we add a tokenizer_type="harmony" branch (anchor on
the final channel, strip control tokens from close_ids) before any SFT.

Run (CPU ok): python scripts/phase_minus1_gpt_oss/diag_m1_B_harmony_lossmask.py
"""

import argparse
import json
import traceback
from pathlib import Path

from transformers import AutoTokenizer

from nla.schema import extract_explanation, wrap_explanation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--out", default="/data/logs/diag_m1_B.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    result: dict = {}

    # --- 0. show the raw Harmony render so failures below are interpretable
    probe = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    full_str = tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=False)
    head_str = tok.apply_chat_template(probe[:1], tokenize=False, add_generation_prompt=True)
    print(f"=== full 2-turn render ===\n{full_str!r}\n=== head+gen render ===\n{head_str!r}\n")
    result["full_render"] = full_str
    result["head_render"] = head_str

    # --- 1+2. miles loss-mask generator on the generic path
    try:
        try:
            from miles.utils.mask_utils import MultiTurnLossMaskGenerator
        except ImportError:
            # miles/__init__ pulls the full training stack (ray etc.) which the
            # Phase −1 pod doesn't install — load mask_utils.py by file path.
            import importlib.util
            import os
            path = os.environ.get("MILES_MASK_UTILS",
                                  "/workspace/miles/miles/utils/mask_utils.py")
            spec = importlib.util.spec_from_file_location("_mask_utils", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            MultiTurnLossMaskGenerator = mod.MultiTurnLossMaskGenerator
        gen = MultiTurnLossMaskGenerator(tok, tokenizer_type="generic")
        close_ids = gen._turn_close_ids()
        decoded = [tok.decode([i]) for i in close_ids]
        print(f"close_ids: {close_ids} → {decoded}")
        result["close_ids"] = {"ids": close_ids, "decoded": decoded}

        token_ids, mask = gen.gen_multi_turn_loss_mask(
            [{"role": "user", "content": "What is 2+2?"},
             {"role": "assistant", "content": "4"}], None)
        masked = [t for t, m in zip(token_ids, mask) if m == 1]
        masked_text = tok.decode(masked)
        print(f"loss-masked region decodes to: {masked_text!r}")
        result["lossmask"] = {"n_tokens": len(token_ids), "n_masked": len(masked),
                              "masked_text": masked_text}
        lossmask_ok = "4" in masked_text
    except Exception:
        traceback.print_exc()
        result["lossmask_error"] = traceback.format_exc()
        lossmask_ok = False

    # --- 3. analysis-channel hijack of <explanation>
    # Simulate a Harmony completion: reasoning in `analysis`, payload in `final`.
    analysis_resp = (
        "<|channel|>analysis<|message|>The user wants an explanation. I should "
        "mention <explanation> tags maybe… actually here is my reasoning, it is "
        "long and may exceed caps.<|end|>"
        "<|channel|>final<|message|>" + wrap_explanation("the actual payload") + "<|return|>"
    )
    got = extract_explanation(analysis_resp)
    hijack_safe = got == "the actual payload"
    print(f"\nextract_explanation on analysis+final response → {got!r} "
          f"({'OK' if hijack_safe else 'HIJACKED/BROKEN'})")
    # Adversarial: analysis channel itself emits an <explanation> block first.
    adversarial = (
        "<|channel|>analysis<|message|>" + wrap_explanation("reasoning leak") + "<|end|>"
        "<|channel|>final<|message|>" + wrap_explanation("real one") + "<|return|>"
    )
    got_adv = extract_explanation(adversarial)
    adv_safe = got_adv == "real one"
    print(f"adversarial (tags inside analysis) → {got_adv!r} "
          f"({'OK' if adv_safe else 'GRABS FIRST MATCH — analysis channel wins'})")
    result["explanation_extraction"] = {
        "plain": got, "plain_ok": hijack_safe,
        "adversarial": got_adv, "adversarial_ok": adv_safe,
    }

    ok = lossmask_ok and hijack_safe
    result["verdict"] = (
        "PASS: generic loss-mask survives Harmony and <explanation> extraction is sane"
        if ok else
        "GATE FAILED: needs tokenizer_type='harmony' branch and/or final-channel-anchored extraction"
    )
    if not adv_safe:
        result["verdict"] += " [WARN: regex takes first match — analysis-channel <explanation> wins]"
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
