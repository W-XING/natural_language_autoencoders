# NLA on gpt-oss-20b — RunPod Execution Plan

> **What this is.** The approved execution plan for training a Natural Language
> Autoencoder (NLA) on `openai/gpt-oss-20b` on RunPod. It **corrects/extends**
> the original 639-line ML plan (`gpt_oss_20b_training_plan.md` at repo root)
> with showstoppers a skeptical critique surfaced, and adds the **RunPod
> orchestration layer** the original plan punted on. Send this doc to the RunPod
> control pod (it carries the context pod-Claude needs).

---

## Context

We want to train a Natural Language Autoencoder (NLA) on `openai/gpt-oss-20b`
— the first **MoE** base model in the project. A detailed 639-line ML plan
already exists at repo-root `gpt_oss_20b_training_plan.md`, and the preset,
two datagen configs, and the injection-token cache entry are already committed.
This plan does **not** restate that work; it (1) **corrects/extends** that plan
with the showstoppers a skeptical subagent critique surfaced, and (2) adds the
**RunPod orchestration layer** the original plan punted on ("Confirm the
cluster... SLURM? Ray? bare ssh?").

Goal: two HF checkpoints (`sft_av`, `sft_ar`) + an RL checkpoint with
`nla_meta.yaml` sidecars, and an FVE-vs-baselines table line-comparable to
`configs/TRAINING_NOTES.md` (Qwen analog: Critic-SL FVE 37.5%, RL 0.752).

**Decisions made with the user:** RunPod for all phases (Stage-2 API on a cheap
CPU pod); **spot + single 8×H100 node** throughout (RL split revised
2026-06-11 to 4/3/1 — see §Phase-4/5 memory re-derivation);
**hard stop after Phase −1** — report results + firm cost, then wait for go/no-go
before any billed (100k datagen / SFT / RL) work. **Overarching principle:
maximize hands-off operation** — the user makes a few high-leverage decisions and
supplies credentials once; a deterministic orchestrator runs the billed pipeline
unattended (see Hands-off operating model below).

## Hands-off operating model

Your entire involvement is **four touchpoints**:

1. **One-time setup:** supply credentials (RunPod API key, Anthropic key,
   container registry, SSH key) and rotate the exposed `.env` keys. I produce all
   artifacts — Dockerfiles, a reproducible control-pod bootstrap script, the
   orchestrator, the Phase −1 diagnostics, the eval driver.
2. **Approve this plan.**
3. **One go/no-go** after Phase −1 (the money gate — review the 3 showstopper
   results + firm cost before the billed run).
4. **Up to two research decisions** *only if* Phase −1 surfaces them (frozen
   experts in −1.0; SGLang fallback in −1.A).

> **Adjustment (2026-06-11):** with the API key staying read-only, pod
> create/stop becomes an additional manual touchpoint at each pod boundary
> (~3 create + ~3 stop for the billed run). See §RunPod infrastructure →
> Orchestration → "Read-only-key adjustment".

After go/no-go, the **deterministic orchestrator** (a script on the always-on
control pod, **not** Claude-in-the-loop) runs the entire billed pipeline
(Stage-0 → Stage-2 → Stage-3 → SFT → RL → eval) unattended: provisions each pod,
runs the phase, checkpoints to the network volume, tears the pod down, advances,
and resumes from the last checkpoint on spot preemption. It **notifies you on
completion or failure** (webhook/email) so you never poll. No babysitting, no
per-step approvals, no console clicking. A script is the true set-and-forget
choice (no token cost, no LLM steering); Claude's value is the interactive parts
(this planning, Phase −1 debugging, reading results), done from your Mac whenever
your network allows. Running Claude on the control pod is *optional*, not the
mechanism.

### Why this needs a plan, not a YAML edit

gpt-oss has three properties with no analog in shipped models (Qwen-7B,
Gemma-3-12B/27B, Llama-3.3-70B): an **MoE backbone with MXFP4-packed expert
MLPs**, **sliding-window attention W=128** alternating with full-attn layers
(response cap is 150 > 128), and the **Harmony multi-channel tokenizer** with
attention sinks. Each touches a code path (critic truncation, injection
readout, loss masking, SGLang `input_embeds` serving) that has only ever run
against dense, full-attention, single-channel models.

---

## Current state (correcting the existing plan's stale "to create" list)

> **Correction (2026-06-10, control pod):** verification FAILED — none of the
> items below were on `origin/main` or this clone; the original commits were
> never pushed from the local machine. All were **recreated from this plan's
> spec** on branch `gpt-oss-20b` (user-approved). Reconcile against the local
> commits when they're pushed.

Already committed — **verify, don't create**:
- `nla/datagen/model_presets.py` — `gpt_oss_20b` preset (num_layers=24,
  d_model=2880, batch_size=4, device_map="cuda:0", turn_marker="<|start|>").
- `configs/datagen/gpt_oss_20b_ultrafineweb_{1k,100k}.yaml` — both pin
  `layer_index: 17`. **Neither has `stage2.cache_from`** → the 100k run is a
  full ~500k-call Sonnet run, *not* the discounted reuse the original plan
  claims. Either add `cache_from` pointing at the 1k smoke's
  `*_explained.parquet`, or accept full cost (see §Cost).
- `nla/datagen/injection_token_cache.yaml` — gpt-oss → char **"㎡"** (U+33A1),
  **token_id 83806**. The original plan body refers to "㊗" in places — that's
  stale; the committed cache is authoritative.

Still to do at the repo level: `nla/notes/gpt_oss_20b_training_plan.md` (publish
the canonical plan), `nla/scripts/eval_gpt_oss_20b.py` (Phase-6 driver), plus
the **conditional fixes** that Phase −1 may force (below).

---

## Phase −1 — Showstopper triage (1×H100, ~1 day, read-only/diagnostic)

**This is the only phase we run before the user's go/no-go.** All on a single
spot H100. If any of the three gates below fails, name the fallback here and
stop. The original plan's Phase −1.A/B/C are folded in and **hardened** per the
critique.

### −1.0 (NEW, top priority) Actor expert trainability under MXFP4

The original plan audits only *critic* truncation; it never checks whether the
**actor's expert MLPs train at all**. A repo-wide grep finds zero
`mxfp4|quantiz|dequant|requires_grad` handling in `nla/`; the actor loads via
plain `AutoModelForCausalLM.from_pretrained` (`nla/train_actor.py:233`). MXFP4
expert weights typically dequant to bf16 as **buffers with
`requires_grad=False`** → SFT/RL would silently train only attention / router /
embeddings while all 32 experts stay frozen. The Phase-2 smoke gates would NOT
catch this; the result would be research-invalid.

**Diagnostic (~1 H100-hr):** load gpt-oss-20b as the actor does, iterate
`model.named_parameters()`, and report which expert tensors have
`requires_grad=True` vs are dtype-quantized buffers; confirm whether FSDP2 can
shard/gather them. **Gate:** if experts are frozen, this is a research decision
for the user — (a) accept "train attn/router/embed only" as the intended NLA-on-
MoE design, (b) add an explicit dequant-to-trainable-bf16 step (extends the
loader, more memory), or (c) abort. Also re-do the Phase-4 memory accounting,
which currently models a dense, fully-trainable 20B.

### −1.A SGLang + gpt_oss + `input_embeds` — prove embeds are *consumed*

The repo already ships a model-specific `gemma3_mm.py` patch
(`patches/apply_sglang_patches.sh`) routing `input_embeds` because the generic
path drops them — **there is no `gpt_oss` analog**. An HTTP-200 from the spike
is insufficient: it can pass while embeds are ignored → RL trains on the literal
"㎡" marker → headline FVE is noise.

**Hardened spike:** launch `python -m sglang.launch_server --model
openai/gpt-oss-20b --disable-radix-cache`; post **two** `input_embeds` payloads
at identical token positions but different embed values; assert the outputs
**differ**. **Gate / fallbacks (lock in here):** works → proceed; embeds ignored
→ write a `gpt_oss` routing patch (model-specific, ~the gemma3 pattern) or fall
back to Miles HF-generate rollouts (~5× slower RL); accepts-but-garbage →
dtype/scale mismatch, debug vs a known-good Qwen call. Also confirm MoE
**weight-sync** (`update_weights` + `/dev/shm` embedding dump) round-trips for
an MoE — the critique flags this as unverified.

### −1.B Harmony loss-mask + channels (extends original −1.C)

The `generic` `_turn_close_ids` (patch `0001:906-929`) asserts assistant
content is a contiguous subsequence right after the prompt head; Harmony's
`<|channel|>final<|message|>…<|return|>` breaks that → the assert likely fires
and training won't start. Separately, the **`analysis` reasoning channel** can
hijack `<explanation>` extraction (`nla/schema.py:44`) and blow the 150-token
cap → mass TRUNCATED→FAILED samples → RL starvation
(`_truncate_to_cross_rank_min`). The original plan hand-waves this as "~30 LOC"
and never mentions channels.

**Dry run:** exercise `MultiTurnLossMaskGenerator(tok, "generic")` on a Harmony
(user, assistant) pair; inspect `close_ids`. **Gate:** add a
`tokenizer_type="harmony"` branch that (a) anchors on the `final` channel, (b)
strips `<|channel|>/<|message|>` control tokens from the appended close
sequence, and (c) ensures the actor prompt renders correctly through
`apply_chat_template`. Confirm `<explanation>` extraction survives a response
that contains an `analysis` channel.

### −1.B′ MoE critic truncation audit (original −1.B, unchanged)

Run the config/state-dict enumeration from the original plan against gpt-oss to
confirm `nla/models.py:_truncate_config_layers` covers every per-layer config
attr and that `prepare_critic_checkpoint.py` carries **router + all 32 expert
tensors** per kept layer. Confirm `_no_split_modules` contains the MoE block
class (else FSDP wraps at leaf granularity → NCCL storm).

### −1.D (NEW) Injection-char / neighbor round-trip through Harmony

Cheapest catastrophic-bug catch. Validate that `compute_canonical_neighbors`
and the marker (char "㎡", token 83806) survive `apply_chat_template` with the
Harmony tokenizer — left/right neighbor IDs must match what the injection hook
(`nla/injection.py`) checks at runtime. This is exactly the silent token-drift
class CLAUDE.md warns about (wrong position → model emits Chinese).

### Phase 0 calibrations (run alongside −1, no code changes)

- **INJ_SCALE:** do NOT reuse the Qwen `median×1.7` constant (the docs
  themselves contradict that ratio). Forward ~32 UltraFineWeb prompts, measure
  the residual-norm distribution **at the extracted positions**, and set
  INJ_SCALE from the p50–p90 of *that* distribution. Record in `phase0_notes.md`.
- **Layer K:** keep the base-model logit-lens KL probe over K∈{15,17,19}, but
  treat it as advisory — the *real* SWA-vs-150 test is bucketing the Phase-2
  smoke loss-gap by response position (free; see Phase 2). Smoke default K=17.

**End of Phase −1: STOP.** Report the three gates + calibrations + a firm cost
estimate; wait for the user's go/no-go.

## Phase −1 RESULTS (run 2026-06-10, 1×H100 SXM on-demand, ~2 GPU-h ≈ $7)

Scripts: `scripts/phase_minus1_gpt_oss/` (branch `gpt-oss-20b`). Verdict JSONs:
control pod `~/phase_minus1_results/`, pod volume `/workspace/logs/`.

| Gate | Result |
|---|---|
| −1.0 expert trainability | **FAIL — decision pending** |
| −1.A SGLang input_embeds | **PASS** |
| −1.B Harmony loss-mask/channels | **FAIL — fix designed, decision pending** |
| −1.B′ MoE critic truncation | **PASS** |
| −1.D marker/neighbor round-trip | **PASS** |
| Phase-0 INJ_SCALE | **measured: 5447–5901 @ K=17** |

- **−1.0:** as the actor loads it (plain `from_pretrained`, Hopper, kernels
  installed), the ~19B MXFP4 expert weights are `triton_kernels` custom tensors
  — in NEITHER `named_parameters` nor `named_buffers`; only 1.8B/20.9B params
  optimizer-visible. Backward probe: experts AND **router get zero grad** (the
  packed MoE forward is non-differentiable end-to-end) — so option (a)
  "train attn/router/embed" is really "attention+embeddings only, router
  frozen". Only path that trains the MoE: **(b) dequant-to-bf16 at actor load**
  (~42GB weights; ≈42GB/GPU optimizer-state share on 8×H100 — redo the Phase-4
  memory accounting). Packed-load peak was 17.3 GiB on 1 GPU.
- **−1.A:** stock SGLang 0.5.9 **consumes** input_embeds for gpt_oss — two
  payloads differing only at the ㎡ position (perturbation norm 5500) gave
  completely different greedy outputs. **No gpt_oss routing patch needed**; the
  HF-generate fallback is dead. Still untested: MoE `update_weights` round-trip
  (RL prerequisite, test at Phase-5 bring-up).
- **−1.B:** confirmed both ways. The `generic` `_turn_close_ids` assert fires
  (Harmony inserts `<|channel|>final<|message|>` = tokens [200005, 17196,
  200008] between gen-prompt and content; close is `<|return|>` = 200002).
  `<explanation>` extraction breaks even NON-adversarially: the analysis
  channel merely *mentioning* the tag corrupts the first-match regex across
  channels. Fix options (decision pending): **(A, recommended) forced-final
  prefill** — append `<|channel|>final<|message|>` to the generation prompt in
  SFT data + rollouts, analysis channel structurally impossible, ~30–60 LOC,
  CPU-verifiable; (B) native two-channel with channel-aware mask/extraction —
  2–4× RL token cost, SFT targets have no analysis content; (C) learned
  suppression (channel header in SFT target) — same mask fix as A but RL can
  drift back. Harden `extract_explanation` (final-channel anchor) regardless.
- **−1.B′:** truncation covers all gpt-oss per-layer config attrs; truncated
  critic keeps layers 0..17 with router + 32-expert tensors uniform per layer;
  `GptOssDecoderLayer` in `_no_split_modules`.
- **−1.D:** ㎡/83806 + neighbors round-trip `apply_chat_template` exactly; the
  injection hook finds exactly 1 position and agrees with the sidecar.
- **Phase-0:** residual norms at extracted positions (32 UltraFineWeb docs):
  p50–p90 = **5447–5901 @ K=17** (K=15 ≈ 3068, K=19 ≈ 9960 — steeply
  depth-dependent). Logit-lens KL non-monotonic (4.19 / 4.84 / 3.78 for
  15/17/19) — advisory only, per plan. Smoke default stays K=17.
- **Env pins discovered (bake into the image):** `kernels==0.9.0` — versions
  0.10–0.15 crash at import against `huggingface_hub<1.0` (which
  transformers 4.57 requires). Stack: torch 2.9.1+cu128, transformers 4.57.1,
  sglang 0.5.9. Also: the MXFP4 quantizer runs CUDA packing even under
  `device_map="cpu"` — never `from_pretrained` the model on a GPU another
  process owns (read tensors via safetensors instead).

**Go/no-go status (updated 2026-06-11):** the two research decisions are
**user-confirmed** — −1.0 = dequantize-to-bf16 (`Mxfp4Config(dequantize=True)`),
−1.B = `forced_final` (see the decision sections below). Balance top-up:
promised by the user, not yet landed (was $199.70). RunPod API key: **stays
read-only for now** (user decision 2026-06-11) — pods can be listed/polled but
not created/stopped via API, so pod lifecycle is manual; the operating model is
adjusted accordingly (see §Orchestration → "Read-only-key adjustment").
RL step-time remains the dominant cost unknown until the 20-step measurement.

---

## MXFP4 actor trainability (−1.0)

**Problem.** ~19B of gpt-oss's 20.9B params are the MXFP4-packed expert MLPs.
Under the default load they are not trainable, so SFT/RL would update only the
attention and embedding parameters and leave the experts unchanged. This differs
from the dense baselines, where all parameters are fine-tuned.

**What −1 tells us.** On hardware: the experts load as `triton_kernels` custom
tensors (present in neither `named_parameters` nor `named_buffers`), the packed
MoE forward has no autograd backward, so both experts and router receive zero
gradient, and 1.8B of 20.9B params are optimizer-visible. Training the experts
requires dequantizing to bf16 at load. Packed-load peak was 17.3 GiB.

**Decision (user-confirmed 2026-06-11) & why.** Load with `Mxfp4Config(dequantize=True)` at the two training
load sites, which makes the experts trainable bf16 parameters and matches the
all-parameter fine-tuning used for the other models. Cost: ~42 GB bf16 weights
plus optimizer/grad state — see §"Phase-4/5 memory re-derivation" (done
2026-06-11: ~10 B/param sharded; SFT fits on 8 GPUs, RL split revised to
4/3/1). OOM ladder: reduce micro-batch, then gradient-checkpointing, then
offload.

**Alternatives & trade-offs.**
- **Frozen experts** (default load): no expert gradients, optimizer state, or
  gradient communication, and experts stay 4-bit, so lower memory and faster
  steps. Trains attention and embeddings only (router also frozen), which is not
  comparable to the dense full-fine-tune baselines.
- **LoRA / partial expert training:** lower memory than full dequantization, at
  the cost of added adapter machinery and reduced comparability with the dense
  baselines. A fallback if full dequantization does not fit the 4/2/2 RL split.
- **Do not train on gpt-oss:** an option if the Phase-4 re-derivation shows RL
  cannot fit even with the OOM ladder.

## Harmony actor (−1.B)

**Problem.** gpt-oss uses Harmony channels (`analysis` for reasoning, `final` for
the answer). The `generic` loss-mask assumes the response tokens are contiguous
immediately after the prompt; the channel control tokens break that assumption,
so the assert fails and SFT does not start. Separately, the `analysis` channel can
interfere with `<explanation>` extraction.

**What −1 tells us.** Both issues reproduce. The assert fails on
`<|channel|>final<|message|>` = `[200005, 17196, 200008]` (close token
`<|return|>` = 200002). Extraction fails without an adversarial input: an
`analysis` channel that contains the `<explanation>` string causes the
whole-string `EXPLANATION_RE` match to return the wrong span.

**Decision (user-confirmed 2026-06-11) & why.** Default to `forced_final`: prefix `<|channel|>final<|message|>`
onto the generation prompt so the model produces the explanation directly and no
`analysis` channel is generated. This addresses both issues, is the smallest
change (~30–60 LOC), and produces an explanation with no reasoning content.
Implement it as a configurable mode (`actor_reasoning_mode`) so Phase 2 can
compare it against `free_analysis` on measured FVE. Components: (1) prompt prefix
`[200005, 17196, 200008]` — the marker stays in the user content
(`<concept>㎡</concept>` / 83806, confirmed by −1.D); (2) a `harmony` loss-mask
branch in Miles patch `0001` with close = `<|return|>` (requires an image
rebuild); (3) harden `extract_explanation` to isolate the final channel before
matching (in-repo); (4) record the mode in the sidecar and assert data-mode equals
runtime-mode. Resolve all token IDs from the live tokenizer rather than
hardcoding, and use the same prompt construction in SFT, RL, and
`compute_canonical_neighbors`.

**Alternatives & trade-offs.**
- **`free_analysis`** (the actor may generate an `analysis` channel; the monitor
  reads only `final`): allows reasoning before the explanation, at ~2–4× RL
  tokens. The `analysis` channel is unsupervised during SFT. A long `analysis`
  moves the explanation past `marker_pos + 128`, where the W=128 sliding-window
  layers no longer attend directly to the marker, which may reduce signal. Use as
  a Phase-2 comparison arm; assess by bucketing FVE by `analysis` length.
- **`bounded_analysis`** (cap `analysis` at ~32–64 tokens): retains short
  reasoning while limiting the sliding-window effect, but requires generation-time
  length control. Consider only if the smoke shows higher FVE at short `analysis`
  lengths.
- **Native two-channel mask / learned suppression** (run options B/C): more
  implementation, and RL can revert toward emitting `analysis`.

---

## Phase-4/5 memory re-derivation (dequantized bf16 actor, 2026-06-11)

**Basis (read from the patched miles source, not assumed):** FSDP2 stores
params **bf16** (patch 0001 `torch_dtype=bfloat16`), `MixedPrecisionPolicy
(param_dtype=bf16, reduce_dtype=fp32)` → sharded grads **fp32**, optimizer is
plain `torch.optim.AdamW` on the bf16 DTensors → `exp_avg`/`exp_avg_sq`
allocated **bf16** (`zeros_like(p)`; the patch-0001 comment claiming "Adam
keeps fp32 master copies" is wrong for torch AdamW on bf16 params). Sharded
train state ≈ **2 (param) + 4 (grad) + 4 (Adam) = 10 B/param**, split across
the group's GPUs. Transients on top: ~1.65 GB per-layer bf16 all-gather
(0.823 B params/layer × 2 B) × prefetch, activations (short sequences,
micro-batch 1–4), ~4–6 GB CUDA ctx + NCCL.

**Param counts:** actor 20.9 B (≈41.8 GB bf16). Critic (K=17 → 18 layers kept,
lm_head stripped, + d² value head): 18 × 0.823 B/layer + 0.579 B embed ≈
**15.4 B** (≈30.9 GB bf16). RL ref model (KL loss, `--ref-load`): +41.8 GB
bf16, FSDP-sharded over the actor group.

| Phase / group | state | GPUs | GB/GPU | verdict |
|---|---|---|---|---|
| SFT actor | 209 GB | 8 | 26.1 | fits (worst-case fp32-Adam 41.8 → still fits) |
| SFT critic | 154 GB | 8 | 19.3 | fits |
| RL actor + ref | 209+42 GB | 4 | 62.7 | tight — grad-ckpt on; 52.3 without KL/ref |
| RL critic @ 4/2/2 | 154 GB | 2 | **77.0** | **effectively OOM** (<3 GB left for transients) |
| RL critic @ 4/3/1 | 154 GB | 3 | 51.3 | fits |
| RL rollout, 1 GPU | 41.8 GB wts | 1 | ~42+KV | fits (GQA 8 kv-heads + SWA → small KV; `--sglang-mem-fraction-static ≈ 0.85`) |

**Conclusion: the plan's 4/2/2 RL split fails at the critic.** Default for
gpt-oss becomes **`ACTOR_GPUS=4 CRITIC_GPUS=3 ROLLOUT_GPUS=1`** (cost: half
the rollout throughput vs 2 engines — shows up in the 20-step timing).
Alternative keeping 2 rollout engines: 4/2/2 with `CPUOffloadPolicy` on the
critic only (miles `fsdp_cpu_offload`; optimizer step on CPU, slower critic
steps; pod needs ≥~200 GB free host RAM).

**Bring-up gate (same pod session as the 20-step timing):** print actual
`bytes/param = (param+grad+optim bytes) / N` after step 1. ≈10 B confirms the
table; ≈16 B (fp32 Adam states — e.g. a miles upstream change) means RL needs
CPU offload for BOTH trainers regardless of split (actor 334/4 = 83.5 GB/GPU),
and SFT still fits. OOM ladder unchanged: micro-batch ↓ → gradient
checkpointing → `fsdp_cpu_offload`.

---

## Phases 2–6 (billed; only after go/no-go) — deltas from the existing plan

Follow `gpt_oss_20b_training_plan.md` Phases 2–6 as written, with these
corrections:

- **Phase 2 (1k smoke):** after the 200-step AV-SFT, **bucket the real-vs-random
  loss gap by response position** (0–50 / 50–100 / 100–150). A gap that
  collapses past position 128 is the SWA readout failure — decide cap-at-128 vs
  different K *here*, cheaply, instead of trusting the base-model KL probe.
- **Phase 3 (100k datagen):** if `cache_from` is wired from the smoke outputs,
  Stage 2 is partially cached; otherwise budget the full ~500k Sonnet calls.
- **Phase 4 (SFT):** single 8×H100 node (`actor_sft.sh`/`critic_sft.sh` are
  already `--*-num-nodes 1 --*-num-gpus-per-node 8`). Memory re-derived for the
  dequantized bf16 actor — see "Phase-4/5 memory re-derivation" below. SFT fits
  on 8 GPUs even in the worst (fp32-optimizer) case. OOM order: drop
  `--micro-batch-size`, then `--gradient-checkpointing`.
- **Phase 5 (RL):** single node, **`ACTOR_GPUS=4 CRITIC_GPUS=3 ROLLOUT_GPUS=1`**
  (re-derived 2026-06-11; the original 4/2/2 OOMs at the critic — see
  §Phase-4/5 memory re-derivation, incl. the 4/2/2+critic-offload alternative);
  `--shm-size=8g` + `NLA_EMBED_DUMP_DIR=/dev/shm/nla`; keep
  `--sglang-disable-radix-cache` (required). **Measure bytes/param and
  step-time in the first ~20 steps** and extrapolate to 4000 rollouts before
  letting it run — MoE step-time is the single largest cost unknown.
- **Phase 6.4 (routing eval):** measure routing at the **extraction/AV layer**
  (where expert-collapse would actually hurt the NLA), not only the critic's
  kept blocks as the original plan states.

---

## RunPod infrastructure layer (the new piece)

**Single 8×H100-80GB node in Secure Cloud for every GPU phase; a cheap CPU pod
for the API-only Stage 2; one Secure-Cloud network volume as the cross-phase
source of truth; a tiny always-on CPU "control" pod running the orchestrator so
the billed phases advance unattended.** Provisioned via the RunPod **REST API v1**
(`POST https://rest.runpod.io/v1/pods`) / `runpodctl`. Spot where the DC offers
it, **on-demand fallback** for the 8×H100 pods when spot capacity is dry. MXFP4
expert kernels require Hopper/Blackwell — pin **H100-80GB** (H200/B200 if
cheaper), never A100.

**Why Secure Cloud (verified June 2026):** network volumes are **Secure Cloud
only** — Community Cloud (the cheapest spot) has no network volumes. Since the
cross-phase storage strategy depends on a persistent volume, all GPU/CPU pods run
in Secure Cloud. Secure is only ~10–30% over Community, and Secure-Cloud spot
still saves big vs on-demand, so the cost impact is small and the zero-storage-
code simplicity is worth it. **Resilience to your network interruptions:** every
long job runs in `tmux` on the pod and the orchestrator runs on the control pod —
so neither your laptop dropping nor the local Claude session going offline stops
work; the pod keeps running and reconnect picks up where it left off.

### Image (build once, reuse all run)

Build a Docker image **once**, push to a registry, use as the pod template for
the whole multi-day run — do not rebuild the heavy stack per pod. Critical
subtlety: **Miles and SGLang are source checkouts that patches modify in place**,
not pip packages — bake them as editable installs with `.git` retained.
Encode `docs/setup.md` exactly:
- Miles: clone → `git checkout $(cat $NLA_REPO/nla/miles_patches/UPSTREAM_PIN
  | cut -d@ -f2)` → `build_conda.sh` → `uv pip install -e .` → `git apply
  $NLA_REPO/nla/miles_patches/*.patch` (checkout-before-apply is what makes the
  patch clean).
- `flash-attn --no-build-isolation`.
- SGLang: clone → `apply_sglang_patches.sh` → `uv pip install -e
  ./sglang/python[all]` (intentionally shadows the conda wheel). **If −1.A needs
  a `gpt_oss` `input_embeds` patch, it lands here.**
- This repo: `uv pip install -e .`. Build-time `python -c "import miles, sglang,
  nla"` so a broken image fails the build, not the H100 pod.
- **Do NOT** bake weights or secrets. gpt-oss-20b is **ungated** (no HF_TOKEN).
- Separate stripped CPU image for Stage 2/1/3 (just `anthropic + pyarrow +
  pyyaml + httpx + orjson`).

### Storage

One **Secure-Cloud network volume ≥1TB** (checkpoints at save-interval 100 over
4000 rollouts blow past 500GB), mounted `/data` on every pod,
`HF_HOME=/data/hf_cache` (40GB weights downloaded once). **Constraint (verified):
volumes must be attached at pod-create and can't be hot-detached** — fine for us,
since the volume persists independently of pods, and each phase creates a fresh
pod with it attached, then terminates. Create the volume in a DC that has **both
8×H100 and network volumes** (the volume and pod must share that DC). Keep
`NLA_EMBED_DUMP_DIR` on `/dev/shm` (scratch), never the volume.

### Orchestration (unattended billed run)

An **always-on CPU control pod — 8 vCPU / 16 GB / 3 GHz** (cents/hr) holds the
RunPod API key (as a pod secret) and runs a **deterministic python/bash
orchestrator** that, for the **approved billed phases only**, chains: provision
GPU/CPU pod → run the phase → poll to completion → tear the pod down → provision
the next phase's pod. It lives on RunPod, so it survives your laptop and the local
Claude session going offline — the whole point. **Phase −1 is NOT auto-chained**
(its go/no-go gates need a human); the orchestrator takes over only after your
go/no-go, for Stage-0 → Stage-2 → Stage-3 → SFT → RL → eval. Each phase tees logs
to `/data/logs/`, the orchestrator **notifies on completion/failure** (webhook or
email), and RL resume (`--load` from the volume) caps any preemption loss to ~100
rollouts.

#### Read-only-key adjustment (2026-06-11)

The user is not providing a read/write RunPod API key for now. The current key
supports `GET` only (list pods, poll status); it cannot create, stop, or
terminate pods, so the orchestrator **cannot provision or tear down pods**.
Adjusted model:

- **Pod lifecycle is manual.** At each pod boundary the orchestrator (or
  Claude, interactively) sends an ntfy notification containing the exact pod
  spec to deploy (GPU type/count, image, network volume, `--shm-size`, spot vs
  on-demand) and the user creates/stops the pod in the RunPod console.
  Everything *inside* a pod stays fully automated (`tmux` + phase script + ntfy
  on completion/failure), and progress monitoring still works via the
  read-only key.
- **Pack phases to minimize touchpoints.** Three pod sessions instead of five:
  (1) 8×H100 — Stage 0 + Stage 1; (2) CPU — Stage 2 + Stage 3 (+ shuffle);
  (3) 8×H100 — SFT → RL 20-step measurement → (cost check) → RL → eval,
  chained in one script. ≈3 create + ≈3 stop actions for the whole billed run.
- **Idle-billing risk shifts to the user**: a finished phase bills until the
  pod is stopped. Mitigations: the completion notification fires immediately;
  **verify on the next pod** whether `runpodctl stop pod $RUNPOD_POD_ID` works
  from inside the pod with whatever pod-scoped credentials RunPod injects — if
  it does, every phase script ends with self-stop and the idle risk disappears
  (only *creation* stays manual). Until verified, assume manual stop.
- **Spot auto-resume is gone.** The orchestrator can't recreate a preempted
  pod. For the multi-day RL run either (a) accept manual
  recreate-on-preemption (ntfy → user recreates the pod, `--load` resumes from
  the volume, ≤100 rollouts lost), or (b) run RL **on-demand** (+~50–80% on
  that line only) to remove preemption babysitting entirely. Recommendation:
  spot for the short pods (Stage 0, SFT), decide spot-vs-on-demand for RL at
  the 20-step cost checkpoint.
- The control pod keeps the read-only key for monitoring only; no write
  credential is stored anywhere.

The control pod is **disposable and reproducible**: a `bootstrap.sh` (in a
private config repo) installs Claude (optional), sets the auth token, clones the
NLA repo, and copies `~/.claude/CLAUDE.md` — so moving/recreating it is a few
scripted minutes, and the data/checkpoints (on the network volume) never move.
16 GB is sufficient for the orchestrator + Stage-2 API fan-out; if the one-time
Stage-3 parquet build on ~1M rows needs more, the orchestrator bursts a temporary
larger CPU pod for that single step rather than oversizing the always-on pod.

### Runbook (never hold idle H100s)

1. **Once:** build+push both images (heavy GPU image + stripped CPU image);
   create the ≥1TB **Secure-Cloud** volume in a DC with 8×H100; create the GPU
   pod template (`--shm-size=8g`, volume at `/data`) and CPU template; launch the
   **always-on control pod** with the RunPod API key as a secret and deploy the
   orchestrator (it drives steps 3–7 unattended once you give go/no-go after
   Phase −1).
2. **Phase −1/0 + 1k smoke:** 1×H100 spot pod, run all triage/diagnostics in
   `tmux`; **stop the pod**; report; await go/no-go.
3. **Stage 0 (100k extract):** 8×H100 spot pod, `run_pipeline --stages 0` in
   tmux (shells `stage0_multigpu.sh`, shard skip-on-resume); `--stages 1`; **stop.**
4. **Stage 2 (API) + Stage 3:** CPU pod, same volume, `ANTHROPIC_API_KEY` as a
   runtime secret, `--stages 2` then `--stages 3` (+shuffle) in tmux; **stop.**
5. **Phase 4 SFT:** 8×H100 spot pod; `prepare_critic_checkpoint`; run
   `actor_sft.sh` + `critic_sft.sh`; watch logs for the CJK injection-failure
   smell; **stop.**
6. **Phase 5 RL:** 8×H100 spot pod, `--shm-size=8g`, 4/3/1 split, in tmux,
   `… 2>&1 | tee /data/logs/rl.log`. Spot preemption mitigated by
   `SAVE_INTERVAL=100` + `--load` resume from volume. Measure step-time first;
   stop ~4000 rollouts.
7. **Phase 6 eval** on the same/idle pod; download final checkpoints; delete the
   volume to stop storage charges.

### Cost (Secure-Cloud spot, single-node; on-demand fallback adds ~50–80%)

Secure-Cloud spot H100 ≈ $1.6–2.2/GPU-hr (vs Community spot $1.3–1.6, vs Secure
on-demand $2.7–4). Bands below assume spot; if 8×H100 spot capacity is dry and
we fall to on-demand for a pod, that line rises ~50–80%.

| Phase | GPU | $ |
|---|---|---|
| **−1/0 + 1k smoke (interactive, ~1 day if green; 2–4 days if fixes needed)** | 1×H100 ~8–12 active h | $30–60 |
| 100k extract | 8×H100 ~2–4h | $60–130 |
| Stage 2 (API, no GPU) | CPU pod | API **~$1,100** (see correction below) + ~$5 |
| SFT (AV+AR) | 8×H100 ~4–10h | $90–280 |
| **RL → 4000 rollouts** | 8×H100, **step-time TBD** | **$1,100–3,200** |
| Storage 1TB + control pod | run-days | $50–160 |

**Total ≈ $1.5–4.2k (Secure spot), RL is 70–85%** and the only order-of-magnitude
unknown — retired by the ~20-step measurement before the full RL run (~$10 of
H100 to de-risk thousands). On-demand-fallback worst case pushes total toward
~$6–7k.

### Stage-2 cost correction + decision (2026-06-12, measured on the 1k smoke)

The plan's original **$150–400** Stage-2 line was wrong. Measured on the
completed 1k smoke (`count_tokens` on real prompts; real explanation lengths):
**836 input + 126 output tokens/call mean** → ~500k Sonnet 4.6 calls =
**$2,200 standard** ($3/$15 per Mtok). Prompt caching is not applicable: the
fixed instruction prefix is 372 tokens, under Sonnet 4.6's 2,048-token
cacheable minimum. **User decision: Message Batches API (−50%) → ~$1,100.**
Implemented as `nla.datagen.providers.AnthropicBatchProvider` (same
`complete()` contract and drop semantics; errored/expired requests retried in
follow-up batches), wired into
`configs/datagen/gpt_oss_20b_ultrafineweb_100k.yaml` together with
`cache_from` pointing at the smoke's `*_explained.parquet` (~1% free reuse).
GPT-5.5 was considered and rejected: $5/$30 per Mtok (≥2× Sonnet) plus
reasoning tokens billed as output, and it would break explainer-model
comparability with the shipped Qwen/Gemma/Llama datasets. Haiku 4.5 batch
(~$367) remains the cheaper fallback if budget demands it, gated on a quality
comparison. Wall-clock: each 65,536-prompt chunk is one batch round, usually
<1 h; ~8 chunks across av+ar → same-day completion on the CPU pod.

---

## Code sync & git workflow

Code runs on the pod but is edited locally (where Claude runs); **git is both the
transport to the pod and the safety net.** No commit → the pod never sees the
change and it's one crash from gone.

- Work on a **feature branch off `main`** (e.g. `gpt-oss-20b`), not `main`
  directly.
- **Commit the existing groundwork first** — the uncommitted `model_presets.py`
  `gpt_oss_20b` preset, `injection_token_cache.yaml` entry, `CLAUDE.md` edits, and
  the untracked `configs/datagen/gpt_oss_20b_*.yaml` — so it's preserved and the
  pod can pull it.
- Phase −1 fixes (Harmony loss-mask branch, any `gpt_oss` SGLang patch, eval
  driver, Dockerfiles, orchestrator) → commit to the branch → push → pod
  `git pull`s.
- **NLA repo is `-e` (editable) on the pod**, so `git pull` takes effect on the
  next run with no reinstall (unless deps/entry-points change). **Miles/SGLang
  patches are baked into the image** — changing those needs an image rebuild (or
  re-apply on the running pod); NLA-repo changes do not.
- The **Docker image is built from a committed SHA** — push fixes *before*
  building/rebuilding the image so the image contains them.
- **Never commit `.env` or secrets** — confirm `.env` is gitignored; inject keys
  as pod secrets at runtime.

## What I need from you

- ~~**RunPod API key (read/write)**~~ — **declined for now (2026-06-11)**: the
  key stays read-only, pod lifecycle is manual via the console (see
  "Read-only-key adjustment"). Revisit only if manual pod create/stop becomes
  a bottleneck.
- **A Secure-Cloud DC that has both 8×H100 and network volumes** — confirm in the
  deploy UI (capacity fluctuates; check 1×H100 for Phase −1 and 8×H100 single-node
  for the rest). The network volume is created in that same DC. Accept on-demand
  fallback for the 8×H100 pods if spot is dry.
- **Spending limit / credits** for the ~$1.5–4.2k spot envelope (up to ~$6–7k if
  on-demand fallback is used heavily).
- **Container registry** (Docker Hub/GHCR) + push creds — the heavy image is
  built once and reused.
- **SSH public key** added to your RunPod account (for tmux/debug attach).
- **Anthropic API key** with a **tier high enough for 500k calls @ concurrency
  100** (low tiers stretch Stage 2 to days). Inject as a runtime secret.
- **Rotate the exposed keys in repo-root `.env`** (`ANTHROPIC_/OPENAI_/
  OPENROUTER_/GEMINI_`, incl. the plaintext `sk-ant-…`) and confirm `.env` is
  gitignored — anything that touched a commit or shared pod is burned.
- ~~**Two research decisions**~~ — **resolved (user-confirmed 2026-06-11)**:
  −1.0 → dequantize-to-bf16 at the training load sites; −1.B → `forced_final`
  Harmony mode (−1.A passed, no SGLang fallback needed).

I can produce the Dockerfiles, the RunPod pod templates, the control-pod
orchestrator, the Phase −1 diagnostic scripts, and the Phase-6 eval driver; you
provide the account/keys/quota above and run the commands (or grant me the RunPod
CLI/creds to drive it).

## tmux (network-drop resilience)

Principle: **long jobs run inside `tmux` on the pod (the remote machine), never a
bare SSH shell and never local tmux.** tmux lives on the pod and owns the job's
shell, so an SSH/wifi drop doesn't kill the job — you re-`attach` after
reconnecting. Mandatory for the billed phases (extract/SFT/RL); optional for
Phase −1's short diagnostics. With the control-pod orchestrator driving the
billed phases, tmux is mainly for *your* manual attach/observe — the orchestrator
launches each phase detached and tees logs to `/data/logs/`.

**Setup (on the pod, after SSH in):**
```bash
apt-get update && apt-get install -y tmux   # if missing
tmux new -s <phase>                          # e.g. rl
<run the phase command>  2>&1 | tee /data/logs/<phase>.log
# detach (job keeps running): Ctrl-b then d
# reattach after a drop:      tmux attach -t <phase>
# list sessions:              tmux ls
```

---

## Verification

- **Phase −1 gates (hard):** (−1.0) actor expert tensors' `requires_grad` state
  reported and a trainability decision made; (−1.A) two differing `input_embeds`
  payloads produce differing SGLang outputs; (−1.B) Harmony loss-mask returns
  clean `close_ids` and `<explanation>` survives an `analysis`-channel response;
  (−1.B′) critic truncation carries router + 32 experts/layer, `_no_split_modules`
  correct; (−1.D) marker/neighbor IDs round-trip through `apply_chat_template`.
- **Phase 2 smoke:** CJK-leak <1% by step 200; real-vs-random gap >0.1 **and**
  not collapsing past response position 128; critic step-0 MSE within 10% of
  predict-the-mean.
- **Sidecar contract:** `av_sft.parquet`'s `nla_meta.yaml` has d_model=2880 and
  `injection_token_id` matching the cached Harmony entry (83806).
- **End:** FVE-vs-baselines table in TRAINING_NOTES.md format; CJK-leak ~0%;
  per-layer routing entropy at the AV layer ≥3.5 bits (uniform-32 = 5).
