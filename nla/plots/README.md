# gpt-oss-20b training-run plots

Training-curve PNGs rendered by `tools/plot_train_log.py` from the miles run
logs. Companion to `../notes/gpt_oss_20b_run_issues.md` (postmortem) and
`../notes/gpt_oss_20b_decisions.md` (decisions). One dir per run; each has
`loss / grad_norm / lr / response_len / reward.png`, plus extras noted below.

| Dir | Run | Verdict | Key plots |
|---|---|---|---|
| `sft_smoke/` | Phase-2 AV-SFT smoke, **first attempt (FA2)** | **INVALID** — non-causal packed FA2 (run-issues §B1); loss fell to ~0.05 by copy-forward, not learning | `loss.png` (the fake fast drop) |
| `sft_smoke_eager/` | Phase-2 AV-SFT smoke, **eager rerun** | valid; gates passed | `loss.png` |
| `sft_100k/` | **Phase-4 actor SFT** (100k, eager) | ✅ loss 9.37→1.495; held-out real-vs-random gap **0.378** nats, 0% CJK | `loss.png`, `loss_log.png` (log-y) |
| `critic_sft/` | **Critic SFT** (100k, norm-anchor + threshold-skip) | ✅ final **FVE 0.360** (≈ Qwen baseline 0.375) | `fve.png`, `norms.png` (pred vs gold norm) |

## Notes
- **`critic_sft/` is partial — rendered up to ~step 356 (FVE ~0.25).** The run
  completed cleanly at step 966 with **final FVE 0.360**; the source log to
  step 966 lives on the network volume
  (`/workspace/logs/critic_sft_100k.log`), not snapshotted here. The committed
  `fve.png` shows the characteristic climb from −1.2 through 0; extrapolate the
  same slope to ~0.36 at step 966.
- The two `sft_smoke*` dirs are kept side-by-side deliberately: the invalid FA2
  run vs the corrected eager run — the visual diff (fake-fast vs honest descent)
  is the clearest illustration of the non-causal-packing bug.
- `critic_sft/fve.png` and `norms.png` are the diagnostic pair for the critic
  saga: FVE climbing while pred-norm stays pinned to gold (the norm-anchor
  working) — see run-issues §B2.

Regenerate any plot from a log with:
`python tools/plot_train_log.py --log <run.log> --out nla/plots/<dir> --title "<title>"`
