"""Phase -1.A — DeepSeek-R1-Distill-Llama-70B loss-mask validation (CPU only).

Empirically confirms the headline finding of the execution plan
(nla/notes/deepseek_r1_distill_llama70b_runpod_execution_plan.md): miles'
`generic`/`distill_qwen` loss-mask absorbs DeepSeek-R1's force-appended
`<｜Assistant｜><think>\n` onto the MASKED prompt side and supervises only the
explanation — so NO new reasoning_mode / loss-mask patch is needed.

Tokenizer-only: needs `transformers` + `miles` on PYTHONPATH; downloads just the
DeepSeek tokenizer (~few MB), no 70B weights, no GPU.

Run:  python -m nla.scripts.phase_minus1a_loss_mask_check
      [--model deepseek-ai/DeepSeek-R1-Distill-Llama-70B] [--miles /tmp/miles]
Exit: 0 = PASS, 1 = FAIL (assertion or error; full stack trace printed).
"""
import argparse
import sys
import traceback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
    ap.add_argument("--miles", default="/tmp/miles",
                    help="path to the installed miles checkout (for mask_utils import)")
    args = ap.parse_args()

    if args.miles and args.miles not in sys.path:
        sys.path.insert(0, args.miles)

    try:
        from transformers import AutoTokenizer
        from miles.utils.mask_utils import MultiTurnLossMaskGenerator
    except Exception:
        traceback.print_exc()
        print("FAIL: could not import transformers / miles.utils.mask_utils", file=sys.stderr)
        return 1

    try:
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        # 1. Confirm DeepSeek's template force-appends <think> at generation time
        #    and exposes <｜Assistant｜> as an added-vocab token (the auto-route key).
        added = set(tok.get_added_vocab())
        assert "<｜Assistant｜>" in added, (
            f"<｜Assistant｜> not in added vocab — auto-route to distill_qwen would "
            f"not fire. added sample={sorted(added)[:8]}"
        )
        gen_prompt = tok.apply_chat_template(
            [{"role": "user", "content": "x"}],
            tokenize=False, add_generation_prompt=True,
        )
        assert gen_prompt.rstrip().endswith("<think>"), (
            f"expected the generation prompt to force a <think> tag; got tail "
            f"{gen_prompt[-60:]!r}"
        )
        print(f"[1] generation-prompt tail: ...{gen_prompt[-40:]!r}")
        print(f"[1] <｜Assistant｜> in added vocab: True  (--loss-mask-type qwen would auto-route)")

        # 2. Build a representative NLA actor pair: user turn (with the injection
        #    marker placeholder) + assistant explanation. Then mask with "generic".
        marker = "㎡"  # ㎡ — the injection-marker family (illustrative here)
        messages = [
            {"role": "user",
             "content": f"Explain the activation at the marked position: {marker}"},
            {"role": "assistant",
             "content": "<explanation>A discussion of renewable energy policy.</explanation>"},
        ]
        maskgen = MultiTurnLossMaskGenerator(tok, tokenizer_type="generic")
        token_ids, loss_mask = maskgen.get_loss_mask(messages)
        assert len(token_ids) == len(loss_mask)

        # 3. There must be a single 0->1 transition (masked prompt, supervised response).
        assert 1 in loss_mask and 0 in loss_mask, "mask is degenerate (all 0 or all 1)"
        first_one = loss_mask.index(1)
        assert all(m == 0 for m in loss_mask[:first_one]), "prompt side has stray supervised tokens"

        prompt_text = tok.decode(token_ids[:first_one])
        response_text = tok.decode(token_ids[first_one:])

        # 4. THE GATE: the forced <｜Assistant｜><think>\n is on the MASKED prompt
        #    side, and the supervised response starts at the explanation — not a
        #    live reasoning trace.
        assert "<think>" in prompt_text, (
            f"<think> is NOT on the masked prompt side — masking is wrong. "
            f"prompt tail={prompt_text[-80:]!r}"
        )
        assert "<think>" not in response_text, (
            f"a live <think> leaked into the SUPERVISED response — actor would be "
            f"trained to emit reasoning. response head={response_text[:80]!r}"
        )
        assert response_text.lstrip().startswith("<explanation>"), (
            f"supervised response does not start at the explanation; "
            f"head={response_text[:80]!r}"
        )

        print(f"[3] mask: {first_one} prompt(masked) + {len(loss_mask) - first_one} response(supervised)")
        print(f"[4] masked prompt tail : ...{prompt_text[-50:]!r}")
        print(f"[4] supervised resp head: {response_text[:60]!r}...")
        print("\nPASS: forced <think> is masked into the prompt; only the "
              "explanation is supervised. No new reasoning_mode / loss-mask "
              "patch needed — use --loss-mask-type generic, reasoning_mode=default.")
        return 0
    except Exception:
        traceback.print_exc()
        print("\nFAIL: Phase -1.A loss-mask validation did not pass.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
