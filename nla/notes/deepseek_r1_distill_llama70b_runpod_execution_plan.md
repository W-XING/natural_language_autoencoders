# Plan: Train NLAs on DeepSeek-R1-Distill-Llama-70B (RunPod)

> On approval, copy to
> `nla/notes/deepseek_r1_distill_llama70b_runpod_execution_plan.md`
> (CLAUDE.md: plans live in the codebase), mirroring
> `nla/notes/qwen3_32b_execution_plan.md` /
> `nla/notes/gpt_oss_20b_runpod_execution_plan.md`.
>
> This plan was rewritten after an adversarial subagent critique
> (per [[critique-plans-with-subagent]]) caught that the first draft's central
> mechanism was unnecessary. The finding is recorded below.

## Context

Add **DeepSeek-R1-Distill-Llama-70B** as a new NLA datagen + training target on
RunPod. The repo already trains NLAs on `llama70b` (Llama-3.3-70B-Instruct),
which is **architecturally identical** (80 layers, d_model 8192, GQA 64/8). The
model differs in two ways — a DeepSeek tokenizer/chat format, and always-on
`<think>` reasoning that cannot be disabled — but **both are already handled by
installed code**, so this is a small, mostly-config change, not new plumbing.

## Headline finding (from the plan critique)

The forced `<｜Assistant｜><think>\n` that DeepSeek's chat template appends is
**already absorbed by the installed miles loss-mask**. `mask_utils.py:36-44`
explicitly names DeepSeek-R1 and splits the prompt at the
`add_generation_prompt` divergence point; `gen_multi_turn_loss_mask_distill_qwen`
(`mask_utils.py:182-196`) puts `…<｜Assistant｜><think>\n` on the **masked prompt
side** and supervises only the explanation. `get_loss_mask:240-244` **auto-routes**
`--loss-mask-type qwen` → `distill_qwen` whenever `<｜Assistant｜>` is in the added
vocab (it is, for DeepSeek).

Consequences for the design:
- **No new `actor_reasoning_mode`, no new miles `.patch`, no prefill machinery.**
  Use `actor_reasoning_mode="default"` and `--loss-mask-type generic` (explicit;
  `qwen` also auto-routes to the same `distill_qwen` function).
- The first draft's "reuse the harmony forced-prefix split" is **impossible**:
  `_harmony_affixes` (`mask_utils.py:74-78`, mirrored `nla/schema.py:74-91`)
  asserts a non-single-channel template and DeepSeek is single-channel — it
  raises by design, telling you to use `generic`.
- The `model_type=="qwen3"` `enable_thinking` gate at `nla_generate.py:129`
  staying a no-op for DeepSeek is **correct, not a bug** — suppression happens at
  the loss-mask, not the template kwarg.

## Verified external facts (primary sources — see Sources)

**Architecture (`config.json`):** `model_type=llama`, `LlamaForCausalLM`;
num_hidden_layers **80**, hidden_size **8192**, heads 64 / KV 8 (GQA),
intermediate 28672, vocab 128256, max_pos 131072, rope_theta 5e5, **bf16**, bos
128000, eos [128001,128008,128009]; **71B ≈ 142 GB bf16**. → `default_layer =
(2*80)//3 = 53` (verified live default; `llama70b` configs do not override
`layer_index` — `model_presets.py:114-130`).

**Tokenizer (`tokenizer_config.json`):** DeepSeek BOS/EOS strings; turn markers
`<｜User｜>` / `<｜Assistant｜>`; `<think>`/`</think>` are **plain text** (no token
IDs); template **force-appends `<｜Assistant｜><think>\n`** on
`add_generation_prompt=True`; **no** `enable_thinking` switch; recommended temp
0.6, **no system prompt**.

**RunPod (runpod.io/pricing, fetched 2026-06-19):** on-demand $/hr — A100 80GB
**$1.39**, H100 80GB **$2.89** PCIe / **$3.29** SXM, H200 **$4.39**, B200
**$5.89**; storage network <1TB **$0.07**/GB/mo, volume 0.10 active / 0.20 idle.
Spot/Community-Cloud not on this page — **re-check before quoting a total.**
142 GB weights → minimum 2× 80 GB GPUs to hold; extraction adds activations + KV.

## Concrete changes (small)

### 1. Preset — `nla/datagen/model_presets.py` (add to `MODELS`)
```python
"deepseek_r1_70b": ModelPreset(
    hf_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    num_layers=80, d_model=8192,            # default_layer = 53
    extractor_kwargs={"batch_size": 1, "max_length": 4096, "device_map": "auto"},
    turn_marker="<｜Assistant｜>",
    accepts_system_role=False,              # DeepSeek: no system prompt
),
```
**`batch_size: 1`** — matches the shipped `llama70b` preset (`model_presets.py:73-75`):
`device_map="auto"` + larger batch on residual memory triggers accelerate's
estimator into CPU offload → meta-tensor crash (the documented flap,
`model_presets.py:37-42`). `turn_marker`/`accepts_system_role` are **inert** for
UltraFineWeb (raw-text stage0 path; they only matter for chat-corpus extractors,
`model_presets.py:23-26`) but set correctly anyway.

### 2. Datagen configs — `configs/datagen/deepseek_r1_70b_ultrafineweb_{1k,100k_gpt55}.yaml`
Base on `llama70b_ultrafineweb_{1k,100k}.yaml` for stage0/1/3, but take the
**stage-2 block from `qwen3_32b_ultrafineweb_100k_gpt55.yaml`** — i.e. the
**GPT-5.5 explainer** (user directive, replacing Sonnet):
```yaml
stage2:
  provider_cls: nla.datagen.providers.OpenAIBatchProvider   # already wired
  provider_kwargs:
    model: gpt-5.5
    reasoning_effort: low          # gpt-5.5 rejects 'minimal'; 'low' = least non-zero
    max_completion_tokens: 2048    # reasoning(low)+output can exceed 1024
    batch_max_requests: 20000
  chunk_size: 65536
```
Set `model: deepseek_r1_70b` + `base_model`. Keep `stage0.multigpu: false`,
`device_map: auto`, **`batch_size: 1`**, `stage3.actor_reasoning_mode: default`
(NOT forced_final). Requires `OPENAI_API_KEY` and the `openai` package
(lazy-imported in `OpenAIBatchProvider`).
**100k throughput:** `device_map="auto"` pipelines one 70B instance across the
visible GPUs → single-instance-bound. The per-doc keyed RNG (a documented
bit-reproducibility invariant) makes data-parallel sharding safe — prefer N
independent 2×H100 stage-0 shards for the 100k run over one 8-GPU pipeline.
Decide at Phase 1.

### 3. Training invocation — no code change
`actor_sft.sh` with `INSTRUCT_MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-70B`,
**`LOSS_MASK_TYPE=generic`** (explicit; `qwen` auto-routes to the same path),
`INJ_SCALE=<calibrated>`. Critic init: `prepare_critic_checkpoint --num-layers 53`.

### 4. License — `release/model_cards/license_stanzas.py`
Add a `DEEPSEEK` stanza + `BY_PRESET["deepseek_r1_70b"]` (MIT distill; Llama-3.3
base license applies — confirm at release).

## RunPod execution phases

**Phase −1 — triage (CPU / 1×H100):**
- −1.A **Loss-mask split (CPU, the key test). ✅ PASSED 2026-06-19** on a
  throwaway 1×H100 US-GA-2 pod (~$0.7), live DeepSeek tokenizer, via
  `nla/scripts/phase_minus1a_loss_mask_check.py`. Result: generation prompt
  forces `…<｜Assistant｜><think>\n`; `<｜Assistant｜>` is added-vocab (auto-route
  fires); mask = 16 masked prompt + 14 supervised response; masked tail ends
  `…<｜Assistant｜><think>\n`, supervised head starts `<explanation>`. Confirms
  the headline finding — no new reasoning_mode/patch; `--loss-mask-type generic`,
  `actor_reasoning_mode=default`.
  - **Gotcha (env pin):** miles `mask_utils._turn_close_ids` assumes
    `apply_chat_template(tokenize=True)` returns `list[int]`. **transformers 5.x
    returns a `BatchEncoding` dict** → `full[lcp]` indexing breaks (`lcp=0`,
    AssertionError). The test PASSES under **transformers 4.55.0**. The training
    env must pin **transformers <5** (the SGLang/miles tested 4.x), else every
    `generic`/`distill_qwen`/`harmony` loss-mask silently breaks at SFT.
- −1.B **Injection marker.** No manual regeneration needed —
  `find_injection_token` (`injection_tokens.py:57-86`) auto-picks a single-token
  CJK char on cache miss and writes `injection_token_cache.yaml` (DeepSeek has no
  entry yet). Just **confirm** the auto-picked char round-trips + neighbors match
  via the existing `config.py:237-257` asserts and the CJK-grep smoke (CLAUDE.md).
- −1.C **SGLang + DeepSeek + input_embeds.** Confirm the injection input-embeds
  path (`nla_input_embeds` patch) loads this checkpoint.

**Phase 0 — calibration (1×–2×H100):** measure **INJ_SCALE** for layer 53 of
*this* model. Ballpark is llama70b's **~30** (`nla_generate.py:254` comment),
NOT gpt-oss's 5447–5901 ([[phase-minus1-results]]) — recalibrate, don't reuse.
Optional logit-lens layer sweep to confirm 53.

**Phase 1–3 — datagen:** 1k smoke (`…_1k.yaml`), gate on CJK-free output, then
100k. Stage-2 explanations via **GPT-5.5** (`OpenAIBatchProvider`, Batch API,
reasoning_effort `low`) — user directive, replacing Sonnet.

**Phase 4–5 — Actor SFT + Critic SFT (8×H100):** reuse the FA2 + no-ckpt +
micro=16 baseline from `TRAINING_NOTES.md`; ~66 GB bf16 weights/GPU like
llama70b; tune micro down if OOM. Held-out AV-gap gate (llama70b precedent 0.378).

**Phase 6 — RL (GRPO):** **deferred.** Per [[stop-before-rl-directive]], stop
after critic SFT and report. At RL time, sanity-check that the actor — prompted
with `…<think>\n` — emits a short `<explanation>` rather than a long live trace;
only if it doesn't is a `</think>` rollout prefill worth revisiting (and even
then it's a `nla_generate` prefill, not the harmony patch).

## Cost (rate inputs; flagged)
- 8×H100 SXM = **$26.3/hr** on-demand (spot/Community unverified — re-check).
- 1 TB network volume ≈ **$50–70/mo** (142 GB weights + 100k activation parquets).
- Stage-2 API: **GPT-5.5** Batch over ~500k av+ar calls ≈ **~$2.2k** (the
  qwen3 gpt55 config's measured estimate: $5/$30 per Mtok, reasoning billed as
  output, halved by Batch; ~2× Sonnet). Confirm OpenAI Batch quota/limits.

## Verification / gates
- **−1.A loss-mask split** passes on DeepSeek tokenizer (the core gate).
- 1k-smoke generated text is **CJK-free** (injection works).
- **Sidecar contract:** IDs/templates/scales/d_model in `nla_meta.yaml`
  assert-match the live tokenizer at startup.
- Packed-vs-standalone per-sample NLL agree (the [[gpt-oss-eager-attention-bug]]
  diagnostic) — rules out a masking/attention bug at 70B.
- 1k end-to-end datagen→SFT, eval_av_gap on held-out doc_ids.

## Open decisions for the user
1. **100k stage-0:** single 8-GPU pipeline vs N×2-GPU data-parallel shards.
2. **Stage-2:** GPT-5.5 via `OpenAIBatchProvider` (per directive) — confirm
   `OPENAI_API_KEY` + Batch quota are set on the pod.
3. **Layer:** trust the 2/3 rule (53) or run a logit-lens sweep first.
4. **GPU tier / spot:** confirm Secure-Cloud spot availability for the cost model.

## Sources
- https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/blob/main/config.json
- https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/blob/main/tokenizer_config.json
- https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B (model card)
- https://www.runpod.io/pricing
- Installed miles: `/tmp/miles/miles/utils/mask_utils.py` (loss-mask routing)
