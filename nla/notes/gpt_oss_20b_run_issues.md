# gpt-oss-20b NLA run — problems & fixes (engineering postmortem)

Chronological-by-category record of every failure hit during the gpt-oss-20b
training run (2026-06-11 → 06-15) and its fix. Branch `gpt-oss-20b`. Each entry
names the exact mechanism, flag, API, or data structure. Two are load-bearing
correctness bugs (§B1 non-causal packed FA2, §B2 critic NaN); the rest are
data-pipeline, infra, or expected-behavior items.

---

## A. Data pipeline

### A1. Ultra-FineWeb loaded non-streaming → full multi-TB pull
**Symptom.** `stage0_extract` hung in `load_dataset("openbmb/Ultra-FineWeb",
split="en")` issuing `Downloading data: .../2048 files` at ~8 s/file (ETA
4–5 h) and would have overrun the 500 GB network volume.
**Cause.** `datasets.load_dataset(...)` without `streaming=True` materialises
the entire split; the canonical `openbmb/Ultra-FineWeb` `en` config is
multi-TB (2048 parquet shards). `stage0_extract` deliberately asserts a
concrete `Dataset` (not streaming), so streaming isn't an option there.
**Fix.** Pre-downloaded a single shard via `huggingface_hub.hf_hub_download`
(`data/ultrafineweb_en/ultrafineweb-en-part-0001-of-2048.parquet`, 566,021
rows, 1.3 GB) to `/workspace/data/ufw_en_slice/`, and pointed datagen at the
local directory: `--override corpus.name=/workspace/data/ufw_en_slice
corpus.split=train`. **Invariant created:** `doc_id` is
`f"{corpus}:{split}:{idx}"`, so the corpus path is baked into every `doc_id`;
the 1k smoke, the 100k run, and any `stage2.cache_from` reuse MUST use the
identical path or the per-`(seed, doc_id)` keyed RNG and the
`detokenized_text_truncated` cache-join keys diverge. The 100k yaml was pinned
to the same slice for this reason.

### A2. stage0 pad-id assert fires on a literal `<|endoftext|>` in corpus text
**Symptom.** `AssertionError: pad_token_id 199999 found in res.token_ids for
/workspace/data/ufw_en_slice:train:6021` killing the stage-0 shard.
**Cause.** ufw document 6021 contains the literal 13-character string
`<|endoftext|>` in its body. The extractor tokenised with
`tokenizer(..., add_special_tokens=True)` and the default
`split_special_tokens=False`, so the HF tokenizer parsed that substring into
the **special token id 199999**, which is gpt-oss's `pad_token_id`. The
stage-0 sanity assert (`pad_id_to_check not in res.token_ids` — guards the
`hidden_states[:seq_len]` slice against padding contamination) correctly
fired. Beyond the assert, a mid-document EOT would corrupt the left-context
semantics of every sampled position after it.
**Fix.** `split_special_tokens=True` in `extractors.py`'s tokenizer call →
special-token *strings* in corpus text are encoded as their literal
characters (tiktoken `disallowed_special` semantics), never as control ids.
Byte-identical for every document that doesn't contain such a string (the
entire 1k smoke included). Committed `dcdcb39`.

### A3. Stage-2 API cost was ~10× the plan estimate
**Symptom.** The execution plan budgeted Stage 2 at \$150–400; measurement put
it near \$2,200.
**Cause / measurement.** `client.messages.count_tokens` on the real 1k-smoke
prompts: **836 input + 126 output tokens/call mean** (the instruction
template + truncated doc text in, a 2–3 feature `<analysis>` block out). At
~500k calls for the 100k run and Sonnet 4.6 pricing (\$3/\$15 per Mtok):
input ≈ \$1,250, output ≈ \$940 → ≈ \$2,200 standard.
**Fix / decision (user-confirmed).** `nla.datagen.providers.AnthropicBatchProvider`
— Message Batches API (`client.messages.batches.create/retrieve/results`),
a flat **50 % discount** → ≈ \$1,100. Implemented with the same
`CompletionProvider.complete()` contract and per-row drop semantics as the
streaming `AnthropicProvider`; splits a chunk into ≤16,384-request batches
(the 100k-chunk × ~3.5 KB/prompt brushes the 256 MB batch cap), polls each to
`processing_status == "ended"`, and resubmits `errored`(server)/`expired`
results for `max_retries` rounds (`invalid_request` and `canceled` raise).
Prompt caching was rejected: the fixed instruction prefix is 372 tokens, below
Sonnet 4.6's 2,048-token minimum cacheable prefix. GPT-5.5 rejected: \$5/\$30
per Mtok (≥2× Sonnet) plus reasoning tokens billed as output, and it would
break explainer-model comparability with the shipped Qwen/Gemma/Llama
datasets. Committed `e94c691`; live-validated 20/20 on real prompts.

### A4. Anthropic account out of credits mid-Stage-2
**Symptom.** `anthropic.BadRequestError 400 invalid_request_error: "Your credit
balance is too low to access the Anthropic API"` after Stage 0/1 completed.
The API key itself was valid (1-token Haiku probe returned 200; Tier-4 limits).
**Fix.** User funded the account; Stage 2 resumed via `--stages 2,3,shuffle`
(stage-2 is chunk-resumable — each completed chunk writes
`{output}.chunks/chunk_NNNNN.parquet` and is skipped on restart, so no API
calls were paid twice).

### A5. `.env` race on a freshly-provisioned pod
**Symptom.** AR-split Stage 2 crashed with `TypeError: Could not resolve
authentication method. Expected one of api_key, auth_token, or credentials`.
**Cause.** The datagen tmux session was launched before the `.env` rsync/scp to
that pod had landed, so `ANTHROPIC_API_KEY` was unset in the process env.
**Fix.** `scp` the `.env`, then relaunch with `set -a; source /root/.env; set +a`.

---

## B. Training-numerics correctness bugs (load-bearing)

### B1. gpt-oss FA2 sink-attention is NON-CAUSAL under miles' packed layout
**Symptom.** The Phase-2 AV-SFT smoke drove train loss to ~0.05 in <50 steps —
"too good." Generated text under the converted checkpoint was degenerate
(`<explanation> The? The? The? …`), and the real-vs-random gap was ~0.04 nats
(noise).
**Root cause.** miles' thd ("packed") training path concatenates the
microbatch's sequences into one `[1, T_total]` stream, passes
`attention_mask=None`, and supplies per-sample `position_ids` that reset to 0
at each sequence boundary. The gpt-oss attention path used by transformers
4.57 with the hub `kernels==0.9.0` sink-attention FA2 kernels does **not**
reconstruct a block-diagonal causal mask from those position-id resets — it
applies a single causal mask over the whole pack, and the attention-sink term
further lets a token attend to its own future. Net effect: tokens see
right-context, so next-token prediction collapses to copy-forward and the
"loss" is meaningless.
**Diagnosis (how it was isolated).** Eliminated in order: (1) eval/train token
layout — `MultiTurnLossMaskGenerator(tok,"harmony").get_loss_mask` output was
byte-identical to the eval harness's `gen_prompt + prefix + resp + close`
(297 tokens, 130 supervised); (2) Distributed Checkpoint (DCP, PyTorch's
`torch.distributed.checkpoint` sharded format)→HF conversion — converted tensors
differed from base by max 1.7e-3 (i.e. genuinely trained, not a load bug),
and DCP↔converted diff was 0.0; (3) stale saves — `iter_0000180` vs
`iter_0000189` embeddings differed by 1.4e-4 (weights evolving); (4)
vector↔explanation pairing — an 8×8 cross-NLL matrix (each explanation scored
against each injected vector) was flat (diag 5.07 vs off-diag 5.23, argmin off
the diagonal), proving the injected vector wasn't being used. The decisive
test: **packed per-sample NLL on the trained checkpoint** = 0.02–0.05
*including pack position 0*, versus 5.0+ for the same samples run standalone —
a pack-position-0 difference is impossible under correct causal masking.
`attn_implementation="eager"` routes through transformers'
`masking_utils` position-id block-diagonal inference and measures
causal-correct (packed == standalone).
**Fix.** Train all gpt-oss models with `--attn-implementation eager`, asserted
in `NLAFSDPActor.__init__` (`model_type == "gpt_oss"` → require eager, else
raise). SGLang rollout inference is unaffected (its own kernels; Phase-1.A
verified `input_embeds` are consumed). Cost: eager builds the full
`[T,T]` mask → slower + more memory than FA2 varlen; re-measure step-time at
RL bring-up. Committed (CLAUDE.md § Debugging updated; memory
`gpt-oss-eager-attention-bug`). The first 189-step smoke was discarded.

### B2. Critic SFT diverges to NaN (gpt-oss eager bf16 backward → garbage gradients)

**Summary (attempt-by-attempt).** Reaching a usable critic took eight attempts.
The training problem had two genuine root causes — (a) the gpt-oss critic's
eager bf16 *backward* emits enormous/non-finite gradients on a fraction of
batches, and (b) a slower magnitude drift in the prediction norm — and the last
two failures were operational mistakes of mine, not training problems
(checkpoint-retention overflowing the disk, and a checkpoint-export bug).
Terms used below:
- **prediction norm (`pred_norm`)** — the L2 norm of the critic's predicted
  activation vector; the gold target norm is ~5300, so a healthy run keeps
  `pred_norm` near there.
- **gradient-norm skip** — a guard (miles patch 0004) that *discards* an
  optimizer step when the global gradient norm is non-finite (and, from attempt
  7 on, also when it is implausibly large, `> 1000`), rather than letting a
  garbage gradient update the weights.
- **norm-anchor** — an added loss term `λ·((|pred|−|gold|)/|gold|)²` that pulls
  the prediction norm toward the gold norm, to stop root cause (b).
- **checkpoint janitor** — `ckpt_janitor.sh`, a background loop that deletes
  old checkpoints so they don't fill the 500 GB volume. Each critic checkpoint
  is ~115 GB (the fp32 Adam optimizer state is most of it), so retention policy
  matters: "keep-1" keeps only the newest, "keep-2-full" the newest two.

| # | Change from previous | What happened, and why |
|---|---|---|
| 1 | initial run (lr 2e-5, no gradient-norm skip) | NaN at step ~6: the gradient norm spiked to 2.7e11 then went non-finite, and the optimizer wrote NaN into the weights. First sign of root cause (a). |
| 2 | compute the loss in fp32 | NaN again at step 7, **bit-for-bit identical** to attempt 1 — proving the change had no effect. miles already casts the critic outputs to fp32 before the loss, so the cast was a no-op. Wrong hypothesis (it was never a loss-precision problem). |
| 3 | raise the `normalize_activation` divide-by-zero floor | NaN again at step 7, again bit-identical. The direction-only loss's gradient is tiny (~0.02), so the `1/|pred|` term I suspected was never the source. Wrong mechanism. |
| 4 | add the non-finite gradient-norm skip; keep lr 2e-5 | No NaN now (bad gradients are discarded), but I **aborted it deliberately**: the prediction norm was drifting up fast (4712 → 6320 over ~28 steps, root cause (b)) and an increasing share of steps were being skipped. Restarted at half the learning rate. |
| 5 | lower the learning rate to 1e-5 | Trained well to a peak **FVE 0.32**, then NaN'd permanently at step 592: the prediction norm kept drifting (→ 7000+) until the activations overflowed bf16 in the *forward* pass — which the gradient skip can't undo. Compounding this, the keep-1 checkpoint janitor (left over from the stable actor run) had already deleted the good earlier checkpoints, so there was **no usable model** and a full rerun was needed. |
| 6 | add the norm-anchor loss term (user "Option A") | NaN at step 87 — *earlier* than before. The norm-anchor did its job (the prediction norm stayed near 5300 the whole run), yet it still NaN'd. This **disproved root cause (b) as the trigger** and exposed root cause (a) clearly: the gradient norm was oscillating between ~0.2 and ~1e19 on alternating batches while the forward pass stayed finite — i.e. the instability is purely in the eager bf16 *backward*. The earlier non-finite-only skip wasn't enough because gradient clipping was rescaling the merely-huge (but finite) gradients down and *applying* them, slowly corrupting the optimizer state. |
| 7 | extend the skip to also drop implausibly-large gradients (norm > 1000) | The training fix **worked** — stable, FVE climbing to 0.25, no NaN. But the *checkpoint save* at step 400 crashed with a disk-quota error: I had switched the janitor to keep-2-full for safety, and two ~115 GB checkpoints plus a third being written exceeded the 500 GB volume. A second self-inflicted disk problem, not a training one. |
| 8 | switch the janitor to keep only the newest checkpoint; rerun from step 0 | **Success.** Full epoch, final **FVE 0.360** (≈ the Qwen baseline 0.375), no NaN, no disk crash. One last issue: the convenience HuggingFace export of the final checkpoint had ~0.14 % NaN values in the value-head readout (a bug in the export's weight-gather), but the underlying FSDP checkpoint's value-head was fully finite, so I repaired the export by copying the value-head from it. |

**Symptom.** Critic-SL ran for steps 0–4 (loss 0.576 → 0.385, FVE
climbing), then `train/grad_norm` spiked to **2.72e11 at step 5**, went
**NaN at step 6**, and `train/loss`/`pred_norm_raw`/`backbone_norm_raw` were
all NaN from step 7 onward. Deterministic and bit-reproducible across reruns.
Separately, the first occurrence also crashed the checkpoint write with
`SafetensorError: Disk quota exceeded` (see C1).
**Two false fixes (recorded so they aren't re-attempted).**
- *fp32 loss cast.* Hypothesis: bf16 loss backward overflow. Casting
  `pred`/`gold` to fp32 in `nla_critic_loss` produced a **bit-identical**
  rerun (`step 0 loss 0.5764217376708984` to the last digit). The tell: that
  value is not bf16-representable, so the loss was *already* fp32 — miles'
  `_train_step` does `values = out.values.float()` before the loss. No-op.
- *normalize_activation relative floor.* Hypothesis: the direction-only MSE
  divides by `|pred|` (`normalize_activation`), whose backward carries
  `1/|pred|`; `clamp_min(1e-12)` makes the scaling `target_scale/floor ≈ 5e13`
  for a near-zero pred. Changed the floor to `target_scale * 1e-3`. Also
  bit-identical — because `∂loss/∂pred ≈ scale/|pred| ≈ 53.7/5600 ≈ 0.02`
  (tiny), normalize is **not** the gradient source, and no sample ever had a
  near-zero norm so the floor never engaged. Kept as a defensive hardening
  anyway.
**Root cause.** Offline gradient localisation on the critic-init checkpoint
showed sane gradients at init (`|g_value_head| ≈ 1.5`, `|g_backbone_out| ≈
3e-4`, max per-layer ≈ 15 at `layers.*.self_attn.q_proj.weight`). The NaN is
not static — it **emerges during training** as the instability documented in
`nla/loss.py`'s own docstring: the direction-only MSE is norm-neutral to first
order in `pred`, but under Adam's scale-invariance the weight-space update
incidentally grows `|pred|` at `~lr·sign(g)` per step (`pred_norm_raw`
observed: 4712 → 5609 → 6320…). As `|pred|`/backbone activations drift up, the
gpt-oss backbone's **bf16 backward** (concentrated in attention `q_proj`, with
the sink term) intermittently amplifies the tiny incoming gradient to a huge
(2.7e11) then non-finite value on specific batches. `clip_grad_norm_(·, 1.0)`
cannot rescue it: with one non-finite grad the total norm is NaN and
`grads.mul_(clip/NaN)` poisons *every* parameter's grad; the optimizer then
writes NaN into the weights, so the next forward's `backbone_last_hidden` is
NaN.
**Fix (two parts).**
1. **miles patch `0004_skip_nonfinite_grad`** — at the FSDP step site
   (`fsdp_utils/actor.py`, after `clip_grad_norm_().full_tensor().item()`),
   guard `if math.isfinite(grad_norm): optimizer.step(); lr_scheduler.step()`
   else log-and-skip. `grad_norm` there is the all-reduced **global** value,
   so every rank takes the same branch (no divergence); the loop-top
   `optimizer.zero_grad(set_to_none=True)` of the next step clears the poisoned
   gradient. Batches with a finite gradient step normally; only the rare non-finite batch is
   dropped. Live on the pod's editable miles; `.patch` recorded for the image
   rebuild (applies after 0001–0003).
2. **Lower critic LR 2e-5 → 1e-5, warmup 50 → 100** — halves the documented
   `~lr·sign(g)` norm-growth rate, the chronic driver. (The Qwen-7B critic was
   stable at 2e-5 because its layer-20 activations are far smaller than
   gpt-oss's layer-17 ~5350-norm residual stream.)
**Result of non-finite-gradient skip + lr-halving (insufficient).** It reached
peak **FVE ≈ 0.32** (step 539–570) but `pred_norm_raw` never stopped drifting
up (6150 → 7060 → …). At **step 592 the weights NaN'd irrecoverably** — not the
earlier intermittent skip (which the guard handles), but a permanent cascade:
once `pred_norm`/the L17 residual stream drifted into the bf16-overflow regime,
a *forward* overflowed → NaN weights, and the non-finite-gradient skip cannot un-poison
weights (it only blocks new NaN-grad updates). Steps 592–967 were all NaN
(~87 % skip in the tail). Compounding operator error: the `ckpt_janitor.sh`
keep-1 retention (set for the *stable* actor run) pruned the finite-FVE
`iter_200/400` saves, leaving only the NaN `iter_967` → **no usable critic
checkpoint, full rerun required.** Lessons: (i) keep-1 retention is unsafe on a
run with known instability — keep periodic + best; (ii) `clip_grad_norm_` +
non-finite-grad-skip is necessary but NOT sufficient — they don't stop the
chronic magnitude runaway that eventually overflows the forward.

**Decision (user, 2026-06-15): Option A — add the norm-anchor term.** The
direction-only MSE is magnitude-invariant, so nothing bounds `|pred|`; the fix
is the docstring's other prescription. Added
`λ·((|pred|-|gold|)/|gold|)²` to `nla_critic_loss` (λ from env
`NLA_CRITIC_NORM_ANCHOR`, default 0.5). Its gradient is purely RADIAL,
orthogonal to the direction-MSE's tangential gradient, so it pins magnitude
without fighting the direction learning FVE measures. This keeps `|pred|` ≈
`|gold|` (~5300), holding the backbone activations out of the bf16-overflow
regime — addressing the *root* cause, not just the symptom. **Trade-off
(accepted):** the objective is no longer pure direction-only MSE, so the
resulting FVE is not strictly apples-to-apples with the Qwen/Gemma/Llama
baselines (which used λ=0); set λ=0 to recover the exact historical objective.
Rerun also switches the janitor to keep-2 + best and adds a NaN watchdog
(kill on sustained non-finite loss) so compute isn't wasted and a good
checkpoint always survives. patch `0004` (non-finite-gradient skip) and the
`normalize_activation` relative floor are retained as defensive depth. Memory
`critic-nan-normalize-floor` updated.

**OUTCOME — Option A was NECESSARY but NOT sufficient; it disproved the
magnitude hypothesis (2026-06-16).** With the norm-anchor on, `pred_norm` was
held at `|gold|` ~5300 *the entire run* (the norm-anchor controlled magnitude as designed) — yet the
critic still NaN'd, and **earlier, at step 87**. That ruled out the
magnitude-runaway / forward-overflow story (the forward was stable, loss flat
~0.29 throughout). The `grad_norm` trace told the real story: it oscillated
**0.2 ↔ 24960 ↔ 1.99e9 ↔ 1.67e17 ↔ 1.82e19 ↔ NaN**, bimodally, from ~step 60 —
i.e. the gpt-oss critic's **eager bf16 BACKWARD produces garbage gradients on
~half the batches**, independent of the forward or `|pred|` (the explosion
concentrates in attention `q_proj` + the sink term). `clip_grad_norm_` rescaled
the huge-but-FINITE ones to norm 1.0 and *stepped* them, which slowly poisoned
the Adam moments → permanent NaN.

**REAL FIX — discard the bad-gradient batches (patch 0004 extended).** Change
the skip condition from "non-finite grad_norm" to **"non-finite OR grad_norm >
`NLA_GRAD_SKIP_THRESHOLD` (default 1000)"** (typical grad_norm ~4, step-0
transient ~118, explosions 1e4+). This *discards* the garbage batches entirely
instead of clipping+stepping them, so Adam is never poisoned. Retained the
norm-anchor (λ=0.5, correctly stabilizes magnitude — harmless) + lr 1e-5.
**Result: a full clean epoch, final FVE 0.360 (≈ Qwen baseline 0.375)**, zero
NaN, zero watchdog fires — at the cost of ~45 % of batches skipped (an inherent
property of the gpt-oss eager bf16 backward, not LR-dependent). The skip rate
caps data efficiency but the surviving ~55 % was enough to match the baseline.
The heavier alternative that would recover the skipped batches — **fp32 critic
backbone** (run the unstable MoE backward in fp32) — was kept in reserve and
not needed. A **dense-critic re-init** (collapse the MoE → a dense MLP,
intermediate = expert_dim × top-k = 11,520, no router) was considered and
**rejected**: the MoE is an input-dependent top-4-of-32 routed mixture that no
fixed dense MLP can reproduce, so there is no function-preserving init — it
would be a distillation project, not an initialization.

---

## C. Infrastructure

### C1. 500 GB network volume filled → checkpoint-write crash
**Symptom.** `safetensors._safetensors_rust.SafetensorError: Error while
serializing: I/O error: Disk quota exceeded (os error 122)` at the step-200
critic save.
**Cause.** FSDP DCP train-state dirs (`model/` + `optimizer/`, ~117 GB each
for the 20.9 B dequantized-bf16 actor) plus converted HF dirs (~78 GB each)
plus the smoke checkpoints exceeded the 500 GB volume. The RL plan's
save-interval-100 × 4000-rollout schedule would have blown it regardless.
**Fix.** `ckpt_janitor.sh` (tmux loop) prunes all but the newest 1–2
`iter_*` dirs every 120 s; deleted redundant DCP train-state once the HF
export existed, plus the smoke HF dir. A volume resize to 1 TB was the cleaner
option but was declined (shared-infra mutation gate); janitor keep-N suffices.

**Recurrence + final fix (2026-06-16).** It crashed AGAIN on the critic rerun's
`iter_400` save — this time because the janitor had been set to **keep-2-full**
(safer for an unstable run) but a *full* critic checkpoint is ~115 GB
(`hf` 29 GB + DCP `model/` ~31 GB + **fp32 Adam `optimizer/` ~55 GB**), so two
of them plus the in-flight third's transient overflowed the 500 GB volume
(~509 GB peak). Training was progressing normally (FVE 0.25, climbing) — only the save
died. **Final janitor = keep-1-full only** (delete all older iters; the newest
is best since FVE climbs monotonically once stable), bounding peak to ~1 full +
1 in-flight full ≈ 451 GB. Lesson: account for the fp32-optimizer DCP (≈2×
params) in checkpoint budgeting, not just the bf16 weights.

### C1b. hf-export `value_head.safetensors` corrupted (DCP is the source of truth)
**Symptom.** The completed critic's final `iter_0000967/hf/value_head.safetensors`
had **11,880 NaN + 93 inf of 8.29 M elements (~0.14 %)** — even though step-966
logged a finite FVE 0.360 (computed *from* the value head) and the backbone
shards were fully finite. The earlier run#3 checkpoint's `hf` value_head was
likewise non-finite.
**Cause.** The convenience `hf/` export in `NLAFSDPActor.save_model`
(`get_model_state_dict(full_state_dict=True, cpu_offload=True)` gather) corrupts
a sliver of the `value_head` on save. It is NOT a training problem — the FSDP
**DCP** `model/value_head.weight` is **100 % finite** (the DCP is written by the
parent's normal checkpoint path, not the NLA gather).
**Fix.** Repair the hf export by copying `value_head` from the DCP: load
`iter_*/model` via `convert_fsdp_to_hf`'s `WrappedStorageReader`, extract
`...value_head.weight` (finite), overwrite `hf/value_head.safetensors`. Verified
the repaired critic loads and predicts finite vectors in the right norm range
(~5400–5700 ≈ gold). **TODO (code, low priority):** fix the `save_model`
value_head gather so the hf export isn't corrupted in the first place.

### C2. Training environment did not exist on the volume venv
**Symptom.** `import miles / sglang / flash_attn` all failed in the volume's
`venv` (only `transformers`/`datasets` were present from datagen).
**Fix.** Built a separate `/workspace/venv_train` (subagent) per `docs/setup.md`
without touching the in-use datagen `venv`: torch 2.9.1+cu128, editable miles
(patches 0001–0003 pre-applied), sglang 0.5.9 (`apply_sglang_patches.sh` + `-e
./python[all]`), `kernels==0.9.0` (0.10–0.15 import-crash vs `huggingface_hub<1.0`),
transformers 4.57.1, editable `nla`. flash-attn: no Dao-AILab wheel for torch
2.9 → used the community prebuilt `flash_attn-2.8.3+cu128torch2.9-cp311` wheel
(mjun0812 releases), import + GPU-kernel verified. MooseFS read-lag on
`/workspace` briefly hid freshly-written site-packages (retry resolved).

### C3. prepare_critic_checkpoint needs CUDA, died on the CPU pod
**Symptom.** The truncated-critic build hung/exited on the CPU pod.
**Cause.** `Mxfp4Config(dequantize=True)` runs the MXFP4→bf16 expert dequant on
CUDA; there is no CPU path. (The CPU pod was also overloaded, load-avg ~122.)
**Fix.** Resequenced critic prep onto the GPU pod after the smoke freed the
GPUs (truncate to K+1=18 layers, strip `lm_head`+final-LN, value head).

### C4. Ray process cleanup / GPU memory retention
**Symptom.** An offline diagnostic OOM'd (`CUDA out of memory… Process … has
57.39 GiB`) while a killed run's workers still held GPU memory; reruns risked
connecting to stale Ray workers.
**Fix.** Tear-down sequence `tmux kill-session` → `ray stop --force` →
`pkill -9 -f train.py` → confirm `nvidia-smi` shows 0 MiB before relaunch.
(`pkill -f train.py` alone left Ray actors holding ~57 GB.) Stale-code was
ruled out: deployed files carried the edits and `nla` resolved to the editable
install; the bit-identical reruns in §B2 were genuine determinism, not caching.

### C5. RunPod capacity / GPU-count mismatch (open)
8×H100 was out of stock in DC US-NE-1 (where the network volume is pinned;
volumes are DC-local and not hot-detachable), so all GPU work ran on a **4×H100**
pod. The plan's Phase-5 RL split `ACTOR=4 CRITIC=3 ROLLOUT=1` assumes 8 GPUs;
RL on 4 GPUs needs a re-derived split (≈2/1/1) or an 8×H100 pod (+ volume
migration) if stock returns. To be settled at the RL 20-step measurement.

---

## D. Expected behavior misread as failure (not bugs)

### D1. eval_av_gap "positional-collapse FAIL" on train rows
The Phase-4 train-row gap eval tripped the automated collapse gate: the
100–150 response-position bucket gap (0.139) fell below 0.5× the 0–50 bucket
(0.354). This is the **gpt-oss W=128 sliding-window attenuation the plan
predicted** — response tokens past ~128 cannot attend directly to the injected
marker. The held-out eval did not trip it (0.179 vs 0.312, ratio 0.57), and
the headline held-out gap (0.378, all buckets positive, 0 % CJK) passed. Plan-
sanctioned remedy: **cap RL responses at 128** (mean explanation ≈126 tokens,
near-free). Not a defect — a known architecture property; the gate threshold is
just conservative on memorised rows.

---

## Outcome (2026-06-16)
Both SFT halves complete and validated on gpt-oss-20b: **actor SFT** held-out
real-vs-random gap **0.378** nats (0 % CJK); **critic SFT** final **FVE 0.360**
(≈ Qwen baseline 0.375). Checkpoints on the volume: actor `av_sft_100k_hf`,
critic `critic_sft_100k/iter_0000967` (hf value_head repaired from DCP). RL is
the next phase, gated on the user (memory `stop-before-rl-directive`).

## Commit trail (branch `gpt-oss-20b`)
- `dcdcb39` stage0 `split_special_tokens=True` (A2)
- `e94c691` `AnthropicBatchProvider` + 100k yaml batch/slice/cache_from (A3)
- eager-attention assert + CLAUDE.md (B1)
- `normalize_activation` relative floor + critic-loss fp32 (B2 defensive)
- norm-anchor term in `nla_critic_loss` (B2, Option A — necessary, insufficient)
- `0004_skip_nonfinite_grad.patch` — extended to skip on grad_norm > threshold,
  the REAL critic fix (B2)
- `plot_train_log` + FVE/norm curves; committed run plots `nla/plots/`
- janitor keep-1-full + DCP-sourced value_head repair are operational
  (`/workspace/*.sh` on the pod), not repo code — see C1/C1b

