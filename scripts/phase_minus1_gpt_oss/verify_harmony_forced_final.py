"""CPU verification of the −1.0/−1.B fixes (user-confirmed decisions, 2026-06-11).

Covers, with the REAL gpt-oss tokenizer (no GPU, no weights):
  1. nla.schema.compute_harmony_affixes resolves the forced_final prefill
     (<|channel|>final<|message|>) and close (<|return|>) from the live
     tokenizer, matching the Phase −1.B token IDs.
  2. miles patch 0003 harmony loss-mask: prefix masked on the prompt side,
     explanation content + <|return|> supervised; the generic path still
     fails on Harmony (the −1.B repro stays a repro).
  3. SFT prompt tokens == rollout prefill tokens (the forced_final contract
     between the harmony mask and nla_generate).
  4. Hardened extract_explanation survives the −1.B adversarial case (analysis
     channel mentioning <explanation> / the final-channel marker itself).
  5. actor_reasoning_mode sidecar round-trip + config-load assert.
  6. mxfp4_dequantize_kwargs returns Mxfp4Config(dequantize=True) for gpt-oss
     and {} for a dense checkpoint (Phase −1.0 decision; config-only check —
     the actual dequantized load is GPU-verified at the next pod session).

Run (needs miles importable — installed package or a patched checkout on
PYTHONPATH; checks 1/4/5/6 run even without miles):
    python scripts/phase_minus1_gpt_oss/verify_harmony_forced_final.py \
        --model openai/gpt-oss-20b [--miles-path /path/to/miles]
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def check_affixes(tok) -> dict:
    from nla.schema import compute_harmony_affixes

    prefix, close = compute_harmony_affixes(tok)
    decoded = [tok.decode([t]) for t in prefix + close]
    assert decoded == ["<|channel|>", "final", "<|message|>", "<|return|>"], decoded
    return {"prefix": prefix, "close": close}


def check_harmony_mask(tok, affixes: dict) -> dict:
    from miles.utils.mask_utils import MultiTurnLossMaskGenerator

    # −1.B repro must stay broken on the generic path.
    gen = MultiTurnLossMaskGenerator(tok, "generic")
    try:
        gen.get_loss_mask([{"role": "user", "content": "hi"},
                           {"role": "assistant", "content": "yo"}])
        raise AssertionError("generic loss-mask unexpectedly PASSED on Harmony — "
                             "did the chat template change?")
    except AssertionError as e:
        if "unexpectedly PASSED" in str(e):
            raise

    h = MultiTurnLossMaskGenerator(tok, "harmony")
    response = "<explanation>\nthe concept of area in square meters\n</explanation>"
    msgs = [{"role": "user", "content": "Here is the vector: <concept>㎡</concept> explain."},
            {"role": "assistant", "content": response}]
    ids, mask = h.get_loss_mask(msgs)
    assert len(ids) == len(mask)
    n_prompt = mask.index(1)
    assert ids[n_prompt - len(affixes["prefix"]):n_prompt] == affixes["prefix"], (
        "prompt does not end with the forced_final prefix")
    assert set(mask[:n_prompt]) == {0} and set(mask[n_prompt:]) == {1}
    supervised = ids[n_prompt:]
    assert supervised[-len(affixes["close"]):] == affixes["close"]
    assert tok.decode(supervised[:-len(affixes["close"])]) == response

    # Contract: SFT prompt side == rollout prefill (nla_generate builds
    # head + prefix the same way).
    head = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
    rollout_ids = tok.encode(head, add_special_tokens=False) + affixes["prefix"]
    assert ids[:n_prompt] == rollout_ids, "SFT prompt tokens != rollout prefill tokens"
    return {"prompt_tokens": n_prompt, "supervised_tokens": len(supervised)}


def check_extraction() -> None:
    from nla.schema import extract_explanation

    adversarial = (
        "<|channel|>analysis<|message|>I mention <explanation>decoy</explanation> and "
        "even <|channel|>final<|message|> as a string<|end|><|start|>assistant"
        "<|channel|>final<|message|><explanation>\nreal\n</explanation><|return|>"
    )
    assert extract_explanation(adversarial) == "real"
    assert extract_explanation("<explanation>plain</explanation>") == "plain"
    assert extract_explanation("<|channel|>analysis<|message|>no tags<|end|>") is None


def check_sidecar_roundtrip(tok) -> None:
    import tempfile

    import yaml

    from nla.config import load_nla_config
    from nla.datagen.sidecar import NLADatasetMeta, NLAExtractionMeta, deserialize_sidecar, serialize_sidecar
    from nla.schema import NLATokenMeta, compute_canonical_neighbors

    template = "Here is the vector: <concept>{injection_char}</concept> explain."
    left, right = compute_canonical_neighbors(tok, template, "㎡", 83806)
    ext = NLAExtractionMeta(base_model="openai/gpt-oss-20b", d_model=2880, layer_index=17,
                            norm="none", corpus="x", corpus_slice={}, positions_per_doc=10)
    tk = NLATokenMeta(injection_char="㎡", injection_token_id=83806,
                      injection_left_neighbor_id=left, injection_right_neighbor_id=right)
    meta = NLADatasetMeta(dataset_id="t", stage="av_sft", row_count=1, extraction=ext,
                          tokens=tk, prompt_templates={"actor": template},
                          actor_reasoning_mode="forced_final")
    text = serialize_sidecar(meta)
    assert deserialize_sidecar(text).actor_reasoning_mode == "forced_final"
    # absent key (pre-gpt-oss sidecar) defaults to "default"
    assert deserialize_sidecar(
        text.replace("actor_reasoning_mode: forced_final\n", "")
    ).actor_reasoning_mode == "default"

    with tempfile.TemporaryDirectory() as d:
        pq = Path(d) / "av_sft.parquet"
        pq.touch()
        (Path(d) / "av_sft.parquet.nla_meta.yaml").write_text(
            yaml.safe_dump(yaml.safe_load(text)))
        cfg = load_nla_config(str(pq), tok)
        assert cfg.actor_reasoning_mode == "forced_final"


def check_dequant_detection(model: str) -> None:
    from nla.models import mxfp4_dequantize_kwargs

    kw = mxfp4_dequantize_kwargs(model)
    qc = kw.get("quantization_config")
    assert qc is not None and qc.dequantize is True, (
        f"{model}: expected Mxfp4Config(dequantize=True), got {kw!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--miles-path", default=None,
                   help="path to a patched miles checkout (prepended to sys.path)")
    p.add_argument("--out", default=None, help="write verdict JSON here")
    args = p.parse_args()
    if args.miles_path:
        sys.path.insert(0, args.miles_path)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    results: dict = {"model": args.model}
    failures: list[str] = []

    def run(name, fn, *a):
        try:
            results[name] = fn(*a) or "PASS"
            print(f"PASS {name}: {results[name]}")
        except Exception:
            traceback.print_exc()
            results[name] = "FAIL"
            failures.append(name)

    run("affixes", check_affixes, tok)
    try:
        import miles.utils.mask_utils  # noqa: F401
        have_miles = True
    except ImportError:
        traceback.print_exc()
        have_miles = False
        results["harmony_mask"] = "SKIPPED (miles not importable — pass --miles-path)"
        print(results["harmony_mask"])
    if have_miles and isinstance(results["affixes"], dict):
        run("harmony_mask", check_harmony_mask, tok, results["affixes"])
    run("extraction_hardening", check_extraction)
    run("sidecar_roundtrip", check_sidecar_roundtrip, tok)
    run("mxfp4_dequant_detection", check_dequant_detection, args.model)

    results["verdict"] = "FAIL: " + ", ".join(failures) if failures else "PASS"
    print(f"\nverdict: {results['verdict']}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
