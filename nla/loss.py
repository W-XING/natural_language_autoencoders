"""NLA critic loss: MSE (optionally scale-normalized) at the last-token position.

Signature matches miles' custom_loss protocol (training_utils/loss.py:889):
    fn(args, parallel_state, batch, logits, sum_of_sample_mean) -> (loss, metrics)

`logits` here is the value-head output, not token logits. Layout varies by backend:
  - FSDP NLACriticModel: [1, T_packed, d_model]
  - Megatron LinearForLastLayer: [T_packed, 1, d_model] (seq-first, and .float()'d)

Extraction: the critic prompt template ends with a fixed suffix (e.g.
"</text> <summary>") so the last real token IS the extraction point.
No scanning — just offset + len - 1 per sample in the packed stream.
The suffix verification is a one-time check at dataset load
(nla.config.verify_critic_suffix), not here.
"""

import os

import torch
import torch.nn.functional as F

from nla.schema import MM_ACTIVATION_KEY, MM_MSE_SCALE_KEY, normalize_activation


def _get_gold_activation(batch: dict) -> torch.Tensor:
    """Read gold activation from batch, handling both FSDP and Megatron data paths.

    FSDP: NLAFSDPActor._get_model_inputs_args pops from multimodal_train_inputs
    and moves to batch[MM_ACTIVATION_KEY] before model forward.

    Megatron: forward_step is a closure inside model.py with no interception
    point — multimodal_train_inputs stays in batch (the pre-hook on GPTModel
    only pops from the copied kwargs, not the batch dict). data.py:244-255
    already concatenated per-sample [1, d] into [B, d].
    """
    if MM_ACTIVATION_KEY in batch:
        return batch[MM_ACTIVATION_KEY]
    mm = batch.get("multimodal_train_inputs")
    assert mm is not None and MM_ACTIVATION_KEY in mm, (
        f"gold activation not found: neither batch[{MM_ACTIVATION_KEY!r}] nor "
        f"batch['multimodal_train_inputs'][{MM_ACTIVATION_KEY!r}] is set"
    )
    return mm[MM_ACTIVATION_KEY]


def nla_critic_loss(args, parallel_state, batch, values, sum_of_sample_mean):
    """Direction-only MSE + norm-anchor between critic prediction and gold
    activation at the last-token position.

    mse_scale (from args.nla_mse_scale, set by the actor's init()) controls
    normalization: if a float, BOTH pred and gold are L2-normalized to that
    norm — direction-only MSE. If None, raw MSE.

    A norm-anchor penalty λ·((|pred|-|gold|)/|gold|)² (λ from env
    NLA_CRITIC_NORM_ANCHOR, default 0.5) is added to pin |pred| to |gold| and
    stop the magnitude runaway that destabilises gpt-oss training (see the
    inline comment + nla/notes/gpt_oss_20b_run_issues.md §B2). Set λ=0 to
    recover the historical pure direction-only objective (Qwen/Gemma/Llama
    baselines).
    """
    unconcat_tokens = batch["unconcat_tokens"]
    gold = _get_gold_activation(batch)
    mse_scale = getattr(args, "nla_mse_scale", None)
    if mse_scale is None:
        mse_scale = batch.get(MM_MSE_SCALE_KEY)
    B = len(unconcat_tokens)

    if B == 0:
        loss = 0.0 * values.sum()
        return loss, {"loss": loss.detach()}

    # FSDP: [1, T_packed, d]. Megatron: [T_packed, 1, d] (seq-first).
    # Either way the batch dim is 1 in thd packing — squeeze is safe.
    assert values.ndim == 3 and 1 in values.shape[:2], (
        f"unexpected values layout {tuple(values.shape)} — expected one of the "
        f"first two dims to be 1 (thd packing with batch=1)"
    )
    values_flat = values.squeeze(0) if values.shape[0] == 1 else values.squeeze(1)
    last_idx = torch.empty(B, dtype=torch.long, device=values_flat.device)
    offset = 0
    for i, tokens in enumerate(unconcat_tokens):
        last_idx[i] = offset + tokens.shape[0] - 1
        offset += tokens.shape[0]
    pred = values_flat[last_idx]

    gold = gold.to(pred.device)
    # Compute in fp32 (defensive — _train_step already .float()'s values, so
    # this is usually a no-op, but keeps the path fp32 if that ever changes).
    # The actual NaN root-cause fix lives in normalize_activation's RELATIVE
    # gradient floor: the direction-only loss divides by |pred|, whose backward
    # scaling is bounded by scale/floor — an absolute 1e-12 floor made that
    # ~5e13 and overflowed for a near-zero-norm outlier pred (gpt-oss critic
    # SFT: grad_norm 2.7e11 at step 5 → NaN at step 6, grad-clip can't rescue
    # an already-NaN gradient). See normalize_activation docstring.
    pred_f, gold_f = pred.float(), gold.float()
    loss_per_sample = F.mse_loss(
        normalize_activation(pred_f, mse_scale),
        normalize_activation(gold_f, mse_scale),
        reduction="none",
    ).mean(dim=-1)

    # Norm-anchor term (user decision 2026-06-15, Option A). The direction-only
    # MSE above is magnitude-invariant in `pred`, so under Adam's scale-
    # invariance the weight update incidentally grows |pred| ~lr·sign(g) per
    # step (the "backbone norm grows ~linearly" instability the docstring
    # warned about). On gpt-oss that runaway drove the L17 residual stream into
    # the bf16-overflow regime → an UNRECOVERABLE NaN at step 592 of the first
    # 100k critic run (peak FVE 0.32 lost; patch-0004 skip-guard + lr-halving
    # only delayed it). This term pins |pred| to |gold| via a dimensionless
    # RELATIVE penalty ((|p|-|g|)/|g|)², whose gradient is purely RADIAL (⊥ the
    # direction-MSE's tangential gradient) — it controls magnitude without
    # fighting the direction learning that FVE measures. λ from env
    # NLA_CRITIC_NORM_ANCHOR (default 0.5). NOTE: this makes the objective no
    # longer pure direction-only MSE, so FVE is not strictly apples-to-apples
    # with the Qwen/Gemma/Llama baselines (see nla/notes/gpt_oss_20b_run_issues.md
    # §B2 and the execution plan critic decision).
    lam = float(os.environ.get("NLA_CRITIC_NORM_ANCHOR", "0.5"))
    norm_anchor = pred_f.new_zeros(())
    if lam > 0.0 and mse_scale is not None:
        pred_norm = pred_f.norm(dim=-1)
        gold_norm = gold_f.norm(dim=-1).clamp_min(1e-6)
        norm_anchor_per_sample = ((pred_norm - gold_norm) / gold_norm) ** 2
        loss_per_sample = loss_per_sample + lam * norm_anchor_per_sample
        norm_anchor = (lam * norm_anchor_per_sample).sum().detach()

    # Miles' loss_function wrapper (training_utils/loss.py:912) rescales by
    # `/global_batch_size * dp_size`, expecting a per-rank SUM (matching
    # sum_of_sample_mean semantics used by policy_loss/sft_loss). .mean()
    # here would pre-divide by B → grads B× too small.
    loss = loss_per_sample.sum()

    # Miles' aggregator (log_utils.py:372) divides every metric by num_samples,
    # expecting per-microbatch SUMS. Keep all entries on the same device —
    # loss.py:922 packs them into one tensor; CPU+CUDA mix is version-fragile.
    backbone_h = batch.get("_nla_backbone_last_hidden")
    dev = pred.device
    log = {
        "loss": loss.detach(),
        "pred_norm_raw": pred.norm(dim=-1).sum().detach(),
        "gold_norm_raw": gold.norm(dim=-1).sum().detach(),
        "mse_scale": torch.tensor(float(B) * (mse_scale if mse_scale is not None else -1.0), device=dev),
    }
    log["norm_anchor"] = norm_anchor.to(dev)
    if backbone_h is not None:
        log["backbone_norm_raw"] = backbone_h[last_idx].norm(dim=-1).sum().detach()
    mean_loss = loss_per_sample.mean().detach()
    b_rv = getattr(args, "nla_baseline_rawvar", None)
    if b_rv is not None and b_rv > 0:
        log["fve_nrm"] = (1.0 - mean_loss / b_rv) * B
    return loss, log
