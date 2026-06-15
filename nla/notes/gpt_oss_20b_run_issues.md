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
(297 tokens, 130 supervised); (2) DCP→HF conversion — converted tensors
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

### B2. Critic SFT diverges to NaN (norm-growth instability + non-finite-grad)
**Symptom.** Critic-SL ran clean for steps 0–4 (loss 0.576 → 0.385, FVE
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
   gradient. Healthy batches step normally; only the rare non-finite batch is
   dropped. Live on the pod's editable miles; `.patch` recorded for the image
   rebuild (applies after 0001–0003).
2. **Lower critic LR 2e-5 → 1e-5, warmup 50 → 100** — halves the documented
   `~lr·sign(g)` norm-growth rate, the chronic driver. (The Qwen-7B critic was
   stable at 2e-5 because its layer-20 activations are far smaller than
   gpt-oss's layer-17 ~5350-norm residual stream.)
**Result.** Post-warmup (step 94): `pred_norm_raw` equilibrated ~6150 (stable,
not runaway), FVE climbing −1.14 → −0.05 (about to cross 0 into beating
predict-the-mean), non-finite-skip rate a steady ~20 % minority. Committed
(patch `0004`, `normalize_activation` floor, plotter FVE/norm curves); memory
`critic-nan-normalize-floor`. **Caveat carried forward:** the ~20 % dropped
batches are an inherent gpt-oss bf16-backward property, not LR-dependent; if
final FVE lands materially below the Qwen baseline (Critic-SL 37.5 %), the
principled remedy is a loss-level norm-anchoring term (penalise `|pred|`
deviating from `|gold|`) — a critic-objective change with FVE-comparability
implications, to be raised as a research decision, not applied silently.

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

## Commit trail (branch `gpt-oss-20b`)
- `dcdcb39` stage0 `split_special_tokens=True` (A2)
- `e94c691` `AnthropicBatchProvider` + 100k yaml batch/slice/cache_from (A3)
- eager-attention assert + CLAUDE.md (B1)
- `normalize_activation` relative floor + critic-loss fp32 (B2 defensive)
- `0004_skip_nonfinite_grad.patch` (B2 real fix)
- `plot_train_log` + FVE/norm curves (tooling)
</content>
