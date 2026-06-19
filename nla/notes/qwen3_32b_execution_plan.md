# Qwen3-32B NLA — execution plan

Train NLAs (natural-language autoencoders) on **Qwen3-32B** (dense, 32.8B).
Structural template: `gpt_oss_20b_runpod_execution_plan.md`. Adding a model is, by
design, a new `ModelPreset` + datagen config; everything downstream reads the
sidecar contract.

**Scope (user, 2026-06-17):** auto-run datagen → actor AV-SFT → critic AR-SFT,
then **STOP + report before RL** (honors the 2026-06-15 stop-before-RL directive
and the 16-GPU RL blocker). RL/GRPO is documented at the end as the follow-up
phase, not auto-executed. Data scale: 1k smoke + Phase-0 calibration → gate →
100k. Thinking: disabled for the actor (`enable_thinking=False`). Stage-2
explainer: `claude-sonnet-4-6` (Anthropic Batch API).

> Plan was revised after an adversarial subagent critique. Key correction: miles
> already ships a `qwen3` loss-mask type (`gen_multi_turn_loss_mask_qwen3`,
> `/tmp/miles/miles/utils/mask_utils.py:144`), so thinking handling is a CLI flag
> (`--loss-mask-type qwen3`) + one rollout-side kwarg — **not** a bespoke
> reasoning mode.

---

## Verified Qwen3-32B facts (from the real `Qwen/Qwen3-32B/config.json`)

Confirmed by downloading config.json directly (not from memory):

| Field | Value | Consequence |
|---|---|---|
| `model_type` | `qwen3` (`Qwen3ForCausalLM`) | `arch_adapters` pass-through; embed scale 1.0 |
| `num_hidden_layers` | **64** | `default_layer = (2*64)//3 = 42` |
| `hidden_size` | **5120** | sidecar `d_model`, injection width |
| heads / kv | 64 / 8 (GQA) | standard |
| `head_dim` | **128** (decoupled, ≠ 5120/64=80) | irrelevant to NLA (inject at 5120-wide residual) |
| `intermediate_size` | 25600 | — |
| `vocab_size` | 151936 | — |
| `max_position_embeddings` | 40960 | NLA seqs short; ample |
| `sliding_window` | **None** | **no SWA → no eager-only constraint (unlike gpt-oss)** |
| `rope_theta` | 1000000 | — |
| `torch_dtype` | bfloat16 | **plain differentiable bf16 — no MXFP4/dequantize** |

---

## Implementation status (committed to the working tree, 2026-06-17)

All deterministic wiring is **done and CPU-validated** on the dev box (the actual
datagen/training runs require the GPU pod — see "Execution environment" below).

1. **`nla/datagen/model_presets.py`** — added `qwen3_32b` `ModelPreset`
   (hf_name `Qwen/Qwen3-32B`, num_layers 64, d_model 5120, `extractor_kwargs`
   `{batch_size: 2, max_length: 4096}`, turn_marker `<|im_start|>`).
   Validated: `resolve({"model":"qwen3_32b"})` → `layer_index=42`, `base_model`
   `Qwen/Qwen3-32B`.
2. **`release/model_cards/license_stanzas.py`** — added a `QWEN3` stanza (worded
   for Qwen3-32B, Apache-2.0) and `"qwen3_32b": QWEN3` in `BY_PRESET`. (Reusing
   `QWEN` would have emitted "fine-tuned from Qwen2.5-7B-Instruct".)
3. **`nla/rollout/nla_generate.py`** — thinking gate. Module flag
   `_DISABLE_THINKING`, set in `_lazy_init` from `hf_config.model_type=="qwen3"`
   (reuses the `hf_config` already loaded for the embed-scale check). At the
   rollout `apply_chat_template` it passes `enable_thinking=False` when set
   (empty kwargs otherwise → no-op for every other model).
4. **`configs/datagen/qwen3_32b_ultrafineweb_{1k,100k}.yaml`** — `model:
   qwen3_32b`, `layer_index: 42`, `actor_reasoning_mode: default` (NOT
   forced_final), UFW slice `/workspace/data/ufw_en_slice`, Sonnet-4.6 explainer
   (1k: streaming `AnthropicProvider`; 100k: `AnthropicBatchProvider` + cache_from
   the 1k splits).

### Training side — use miles' shipped masker (no new code)
Pass `LOSS_MASK_TYPE=qwen3` to `actor_sft.sh` / `critic_sft.sh` / `rl.sh` (they
default to `qwen`; `mask_utils.py:245` dispatches `qwen3`). Keep
`actor_reasoning_mode: default`.

### CPU validation of the thinking + loss-mask gate (the flagged must-verify item)
Run on the **real** Qwen3 tokenizer (`Qwen2TokenizerFast`):
- `add_generation_prompt=True` default → prompt ends `<|im_start|>assistant\n`
  (no prefill; model would generate `<think>` live).
- `enable_thinking=False` → prompt ends `<|im_start|>assistant\n<think>\n\n</think>\n\n`
  (empty think block prefilled → live thinking structurally suppressed). ✓
- Full assistant turn render (SFT side) **auto-injects** `<think>\n\n</think>\n\n`
  before `<explanation>` even at default settings.
- Replicated `gen_multi_turn_loss_mask_qwen3` exactly: `gen_token_length=3` masks
  `<|im_start|>assistant\n`; **trained (loss=1) region** =
  `<think>\n\n</think>\n\n<explanation>…</explanation><|im_end|>\n`.

**Documented asymmetry (accepted, not a bug):** at SFT the empty think block is in
the response (loss=1); at rollout (`enable_thinking=False`) it's in the prompt.
Autoregressively consistent — SFT trains `…</think>\n\n<explanation>`, so the model
predicts `<explanation>` after the prefilled block. `extract_explanation` parses
`<explanation>` regardless. The think block is a constant empty string → trivial
to learn, no harm. **Confirm on the pod** that rollout/eval generations contain no
non-empty `<think>` content and that train loss ≈ teacher-forced NLL (not collapsed).

---

## Pending pod-side wiring (can't be validated without a GPU checkpoint)

- **`nla/scripts/eval_av_gap.py` is `forced_final`-only** (`:92` asserts it, `:99`
  builds Harmony affixes). For Qwen3 (default mode) the AV-gap gate needs it
  generalized: drop the assert, build `gen_prompt_ids` with
  `enable_thinking=False`, set `prefix_ids=[]`, and derive `close_ids` generically
  (the `_turn_close_ids` probe: diff a 2-turn render against prompt+raw-response —
  for Qwen that's `<|im_end|>\n`). This is a smoke-time task (matches the
  layout the rollout produces). Alternative: reuse whatever AV-gap path the qwen7b
  run used — but `eval_av_gap.py` is the only one in the repo.

---

## Execution environment (HARD BLOCKER for the runs)

The current session box is **CPU-only** and is NOT the training pod:
- `.venv` has `torch 2.9.1+cpu`, `cuda avail False`, **0 GPUs**.
- `miles` is **not installed** in the venv (source exists at `/tmp/miles`).
- No `/workspace/data/ufw_en_slice` (UFW corpus slice absent).
- `WANDB_API_KEY`, `ANTHROPIC_API_KEY` **unset**; no `wandb` CLI.

→ Datagen (needs GPU + UFW slice + Anthropic key), SFT, and critic SFT **cannot
run here**. The wiring above is committed and validated so the runs are turnkey
once on a pod with: 8×H100(-80GB), miles installed + NLA patches applied, the UFW
slice staged at `/workspace/data/ufw_en_slice`, and `ANTHROPIC_API_KEY` +
`WANDB_API_KEY` exported.

---

## Logging — Weights & Biases

Append to every training launch: `--use-wandb --wandb-project nla-qwen3-32b
--wandb-group <phase>` (e.g. `qwen3_32b_av_sft_1k`). Auth via `WANDB_API_KEY`
(or `--wandb-key`); `--wandb-mode offline` if the pod has no egress. Watch
`train/loss`, `train/grad_norm` (critic-NaN early warning).

---

## Run sequence (on the pod)

0. **Datagen 1k** — `python -m nla.datagen.run_pipeline --config
   configs/datagen/qwen3_32b_ultrafineweb_1k.yaml`.
1. **Phase-0 calibration** — forward ~32 UFW docs at the extracted positions
   through Qwen3-32B @ layer 42; measure residual L2-norm p50–p90; pick a round
   `INJ_SCALE` in-band. (gpt-oss: 5447–5901 @ K=17; Qwen3 d=5120/L=64 differs.)
2. **Actor AV-SFT (smoke)** — `AV_SFT_PARQUET=… INSTRUCT_MODEL=Qwen/Qwen3-32B
   SAVE_DIR=… INJ_SCALE=… LOSS_MASK_TYPE=qwen3 bash configs/actor_sft.sh
   --use-wandb --wandb-project nla-qwen3-32b --wandb-group qwen3_32b_av_sft_1k`.
   Gate (via generalized eval_av_gap): gap > 0.1 nats, 0% CJK, no positional
   collapse, loss/mask sane.
3. **Prepare critic init** — `python -m nla.scripts.prepare_critic_checkpoint
   --base-model Qwen/Qwen3-32B --num-layers 42 --dataset-sidecar $AR_SFT_PARQUET
   --output $CRITIC_INIT_CKPT` (keeps blocks 0..42 → 43 layers in config.json).
4. **Critic AR-SFT (smoke)** — `AR_SFT_PARQUET=… CRITIC_INIT_CKPT=… SAVE_DIR=…
   bash configs/critic_sft.sh --use-wandb --wandb-project nla-qwen3-32b
   --wandb-group qwen3_32b_critic_sft_1k`. Gate: FVE ≈ qwen7b ~0.375; no NaN.
5. **Datagen 100k** — run the 100k config (cache_from → the smoke).
6. **Actor SFT + critic SFT at 100k** with calibrated `INJ_SCALE` /
   `LOSS_MASK_TYPE=qwen3`, wandb groups `qwen3_32b_*_100k`.
   → **STOP: report; do not auto-start RL.**

### Infra
- Actor SFT: 32.8B full FSDP ≈ 66 GB/GPU on 8×H100-80GB — tight. Knobs (math-
  neutral): activation checkpointing, `--micro-batch-size 1-2`, CPU offload.
- `extractor_kwargs.batch_size=2` is a starting guess (66 GB weights) — tune in
  Phase 0 (may need 1, or `device_map`).

---

## RL (GRPO) — follow-up, NOT auto-run

`RL_PARQUET=… INSTRUCT_MODEL=Qwen/Qwen3-32B ACTOR_SFT_CKPT=…/iter_XXXX
CRITIC_SL_CKPT=…/iter_XXXX/hf RUN_DIR=… LOSS_MASK_TYPE=qwen3 bash configs/rl.sh`.
On-policy synchronous `train.py`, `--sglang-disable-radix-cache` (load-bearing),
150-tok cap, embed scale 1.0. **Dominant risk:** `rl.sh` defaults
`ACTOR_GPUS=8 / CRITIC_GPUS=4 / ROLLOUT_GPUS=4` = **16 GPUs (2 nodes)** —
32.8B actor + ~22B critic + 32.8B SGLang rollout on separate GPUs won't fit one
8-GPU pod naively. Needs 2 nodes or a deliberate colocation/offload scheme.
