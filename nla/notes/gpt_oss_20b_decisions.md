# gpt-oss-20b NLA run — decision register

Every material decision taken on the gpt-oss-20b run, with rationale,
alternatives, and trade-offs. Companion to `gpt_oss_20b_run_issues.md`
(problem→fix) and `gpt_oss_20b_runpod_execution_plan.md` (the plan). Dates UTC.
"User" = W-XING; otherwise decided by the operator (Claude) under the standing
autonomy grant (see P1).

---

## R. Research / ML decisions

### R1. MXFP4 experts → dequantize-to-bf16 at the training load sites
**Decision (user, 2026-06-11).** Load the actor/critic with
`Mxfp4Config(dequantize=True)` so the ~19 B MXFP4-packed expert MLPs become
trainable bf16 parameters; all-parameter fine-tune.
**Why.** Phase −1.0 proved the packed MXFP4 experts are `triton_kernels` custom
tensors present in neither `named_parameters` nor `named_buffers`, and the
packed MoE forward is non-differentiable end-to-end (experts AND router get
zero grad). Without dequant only ~1.8 B/20.9 B params train → not comparable to
the dense full-fine-tune baselines.
**Alternatives.** Frozen experts (train attn/embed only — router also frozen;
not comparable); LoRA on experts (less memory, less comparable); don't train on
gpt-oss. **Cost.** ~42 GB bf16 weights + optimizer/grad state; drove the
Phase-4/5 memory re-derivation.

### R2. Harmony actor mode → `forced_final`
**Decision (user, 2026-06-11).** Prefill `<|channel|>final<|message|>` =
`[200005, 17196, 200008]` onto the generation prompt so the actor emits the
explanation directly with no `analysis` channel.
**Why.** The `generic` loss-mask assert fires on Harmony's channel control
tokens, and the `analysis` channel hijacks `<explanation>` extraction. Smallest
fix (~30–60 LOC), CPU-verifiable, no reasoning content to supervise.
**Alternatives.** `free_analysis` (2–4× RL tokens, analysis unsupervised,
sliding-window risk past pos 128); `bounded_analysis`; native two-channel /
learned suppression. Implemented as a configurable `actor_reasoning_mode` so
Phase-2 could compare; default `forced_final`. Pairs with miles patch `0003`
(harmony loss-mask, close=`<|return|>`=200002) and final-channel-anchored
`extract_explanation`.

### R3. Extraction layer K = 17
**Decision.** Layer 17 (the committed datagen yaml), ~2/3 depth of the 24-layer
model.
**Why.** Phase-0 logit-lens KL probe over K∈{15,17,19} was advisory and
non-monotonic; the preset 2/3-rule gives 16, the yaml pins 17. The real test is
the Phase-2 smoke loss-gap bucketed by response position (run, see R6). Kept 17.

### R4. INJ_SCALE = 5500
**Decision.** Injection scale 5500 (a round value inside the Phase-0 measured
band).
**Why.** Phase-0 measured the residual-norm distribution **at the extracted
positions** over 32 UltraFineWeb docs: p50–p90 = **5447–5901 @ K=17** (K=15 ≈
3068, K=19 ≈ 9960 — steeply depth-dependent). Did NOT reuse the Qwen
`median×1.7` constant (the docs contradict that ratio). 5500 ≈ p50.
**Validated.** Phase-2 smoke + Phase-4 held-out real-vs-random gap 0.378 nats,
0 % CJK — injection carries signal at this scale.

### R5. Stage-2 explainer → Sonnet 4.6 via the Message Batches API
**Decision (user, 2026-06-12).** `claude-sonnet-4-6` through
`AnthropicBatchProvider` (Batches API, flat −50 %) → ≈ \$1,100 for the ~500k
100k-run calls.
**Why.** `count_tokens` on real prompts measured 836 input + 126 output
tokens/call → ≈ \$2,200 standard (the plan's \$150–400 was wrong by ~10×). Stage
2 is offline + order-independent → ideal for batching.
**Alternatives rejected.** Standard API (\$2,200, no code). Haiku-4.5-batch
(~\$367) — cheapest but a different explainer model = quality/comparability
risk. GPT-5.5 — \$5/\$30 per Mtok (≥2× Sonnet) + reasoning billed as output +
breaks explainer comparability with the shipped Qwen/Gemma/Llama datasets.
Prompt caching — not viable (372-token instruction prefix < Sonnet 4.6's
2,048-token minimum cacheable prefix).

### R6. RL response cap = 128 tokens (sliding-window attenuation)
**Decision (operator, 2026-06-13).** Cap RL-generated explanations at 128
tokens.
**Why.** gpt-oss alternates W=128 sliding-window attention layers; response
tokens past ~128 cannot attend directly to the injected marker. Phase-4's
train-row eval showed the real-vs-random gap attenuating in the 100–150
response-position bucket (0.139 vs 0.354 in 0–50), tripping the conservative
collapse gate; held-out passed (0.179 vs 0.312). Mean explanation ≈126 tokens,
so the cap is near-free. Plan-sanctioned (Phase 2 delta).

### R7. gpt-oss trains with `--attn-implementation eager` ONLY
**Decision (operator, 2026-06-12).** Force eager attention for all gpt-oss
training; asserted in `NLAFSDPActor.__init__`.
**Why.** The FA2 sink-attention hub kernels (`kernels==0.9.0`) are NON-CAUSAL
under miles' packed layout (`attention_mask=None` + per-sample `position_ids`):
tokens attend to their own future, SFT collapses to copy-forward (fake loss
~0.05). Eager routes through transformers' position-id block-diagonal mask
inference and measures causal-correct (packed per-sample NLL == standalone).
**Cost.** Eager builds the full `[T,T]` mask → slower + more memory than FA2
varlen; re-measure step-time at RL bring-up. SGLang rollout inference
unaffected. See run-issues §B1.

### R8. Critic NaN fix → norm-anchor term (Option A)
**Decision (user, 2026-06-15).** Add `λ·((|pred|-|gold|)/|gold|)²` to
`nla_critic_loss` (λ from env `NLA_CRITIC_NORM_ANCHOR`, default 0.5).
**Why.** The first 100k critic run reached peak FVE 0.32 then NaN'd
irrecoverably at step 592: direction-only MSE is magnitude-invariant, so Adam's
scale-invariance grows `|pred|` ~lr·sign(g)/step until the L17 residual stream
overflows bf16 in the forward. The skip-guard (R-eng patch 0004) + lr-halving
only delayed it. The anchor's gradient is purely RADIAL (⊥ the direction-MSE's
tangential gradient) → pins `|pred|` ≈ `|gold|` (~5300), holding activations out
of the overflow regime. Root-cause fix, per the loss docstring's "add norm
term" prescription. Early validation (rerun step 40): pred_norm 5325 tracking
gold 5343.
**Alternatives rejected.** Lower-LR-further + early-stop (delays not fixes;
preserves objective); fp32 critic backbone (2× memory, may not fit 4×H100).
**Trade-off (accepted by user).** Objective is no longer pure direction-only
MSE → resulting FVE is NOT strictly apples-to-apples with the Qwen/Gemma/Llama
baselines (λ=0 recovers the historical objective).

---

## E. Engineering / infrastructure decisions

### E1. UltraFineWeb → local single-shard slice
Point datagen at `/workspace/data/ufw_en_slice/` (ufw-en part-0001-of-2048,
566,021 docs), NOT the hub repo. `stage0_extract` loads non-streaming and the
canonical `openbmb/Ultra-FineWeb` `en` split is multi-TB. The corpus path is
baked into `doc_id` → the 1k smoke, 100k run, and any `cache_from` MUST use the
identical path for keyed-RNG / cache-join consistency.

### E2. `split_special_tokens=True` in the stage-0 extractor
A ufw doc literally contains `<|endoftext|>`; default encoding parses it to
special id 199999 (= gpt-oss `pad_token_id`), tripping the stage-0 pad-slice
assert and corrupting left-context. Encode special-token strings as literal
characters. Byte-identical for clean docs.

### E3. `AnthropicBatchProvider` implementation
Same `CompletionProvider.complete()` contract + per-row drop semantics as the
streaming provider; ≤16,384 requests/batch (256 MB cap); poll to
`processing_status == "ended"`; resubmit `errored`(server)/`expired` for
`max_retries` rounds (`invalid_request`/`canceled` raise). Wired into the 100k
yaml with `cache_from` the smoke's `*_explained.parquet`.

### E4. miles patch `0004` — skip optimizer step on non-finite grad_norm
Guard `if math.isfinite(grad_norm): step()` at the FSDP step site; `grad_norm`
is the all-reduced global value (rank-consistent), next step's `zero_grad`
clears the poison. `clip_grad_norm_` can't rescue an already-NaN grad
(multiplies by clip/NaN). Necessary but NOT sufficient alone (see R8). Retained
as defensive depth.

### E5. `normalize_activation` relative gradient floor
Floor the norm at `target_scale*1e-3` (was an absolute `1e-12`). Byte-identical
for all non-degenerate vectors; prevents a `scale/floor`≈5e13 gradient on a
near-zero-norm vector. Defensive — was NOT the critic-NaN root cause (that's
R8), kept anyway.

### E6. Checkpoint retention → keep-2-full + preserve-`hf`, plus NaN watchdog
**Changed 2026-06-15 after the first critic run's good checkpoints were lost.**
The keep-1 janitor (fine for the stable actor run) pruned the finite-FVE
critic saves, leaving only the NaN final → no usable checkpoint. New janitor
keeps the newest 2 iters full (resume) and slims older ones to just `iter_N/hf/`
(the ~30 GB usable critic export). A NaN watchdog kills the run after 20
consecutive non-finite loss steps so H100 time isn't wasted on poisoned weights.

### E7. Separate `venv_train`; flash-attn community wheel
Built `/workspace/venv_train` per `docs/setup.md` without touching the in-use
datagen `venv`. flash-attn: no Dao-AILab wheel for torch 2.9 → used the
community prebuilt `flash_attn-2.8.3+cu128torch2.9-cp311` (mjun0812). Pins:
torch 2.9.1+cu128, transformers 4.57.1, sglang 0.5.9, `kernels==0.9.0`.

### E8. 4×H100 (not 8×) — capacity-forced; RL split deferred
8×H100 was out of stock in DC US-NE-1 (where the network volume is pinned and
not hot-detachable), so all GPU work ran on a 4×H100 pod. SFT fits 4 GPUs. The
plan's Phase-5 RL split `4/3/1` assumes 8 GPUs → RL needs a re-derived 4-GPU
split (≈2/1/1) or an 8×H100 pod + volume migration. **Open — to settle at RL
bring-up.**

---

## P. Process / operating decisions (user)

### P1. Autonomy: "keep going as you see fit" (2026-06-12)
Run all phases without per-step approval; use the plan's measurable gates as
automatic go/no-go; stop only on a gate failure (research decision), budget
blowout, or a destructive/irreversible action. See memory
`autonomous-run-directive`.

### P2. STOP before RL (2026-06-15)
P1's autonomy is OVERRIDDEN for the RL phase: finish critic SFT, report, and
**wait for explicit go-ahead** before any RL bring-up (even the 20-step
measurement). RL is the ~70–85 % cost line. Memory `stop-before-rl-directive`.

### P3. No emails (2026-06-11)
Report in-session only; the ntfy email bridge is not to be used (re-confirm
before any future use). Memory `notification-channel`.

### P4. RunPod read/write key provided (2026-06-12)
Earlier the key was read-only (manual pod lifecycle via console). A r/w key was
then supplied → pod create/stop is automated via REST v1; the manual-lifecycle
model is obsolete. Network volume `2e7ynkygr5` (500 GB, US-NE-1).

---

## Status of decisions (2026-06-15)
Settled & validated: R1–R5, R7, E1–E5, E7. Validated-in-progress: R8 (critic
rerun, early signal good). Pending: R6 applies at RL; E8 (RL split) + P2 gate
open. The two load-bearing correctness fixes are R7 (eager) and R8 (norm-anchor).
</content>
