"""Plot training curves from a miles run log.

Parses the `log_utils` step/rollout dict lines that miles emits, e.g.
    ... step 25: {'train/loss': 6.78, 'train/grad_norm': 13.1, 'train/lr-pg_0': 1e-05, ...}
    ... rollout 25: {'rollout/response_lengths': 121.2, 'rollout/rewards': 0.0, ...}
and writes one PNG per metric group. Works on partial logs of a running job.

Usage:
    python tools/plot_train_log.py --log sft_smoke.log --out ~/plots/sft_smoke \
        [--title "Phase-2 AV-SFT smoke"]
"""

import argparse
import ast
import re
import traceback
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_LINE_RE = re.compile(r"\b(step|rollout) (\d+): (\{.*\})\s*$")


def parse_log(path: Path) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for line in path.read_text(errors="replace").splitlines():
        m = _LINE_RE.search(line)
        if not m:
            continue
        step = int(m.group(2))
        try:
            d = ast.literal_eval(m.group(3))
        except (ValueError, SyntaxError):
            continue  # truncated/garbled dict line (e.g. log cut mid-write)
        for k, v in d.items():
            if isinstance(v, (int, float)):
                series[k].append((step, float(v)))
    return series


# metric key -> (group png, y-label, log-scale)
_GROUPS = {
    "train/loss": ("loss", "train loss (nats)", False),
    "train/grad_norm": ("grad_norm", "grad norm", True),
    "train/lr-pg_0": ("lr", "learning rate", False),
    "rollout/rewards": ("reward", "rollout reward", False),
    "rollout/raw_reward": ("reward", "rollout reward", False),
    "train/fve_nrm": ("fve", "FVE (normalized)", False),
    "train/pred_norm_raw": ("norms", "raw L2 norm", False),
    "train/gold_norm_raw": ("norms", "raw L2 norm", False),
    "rollout/response_lengths": ("response_len", "response length (tokens)", False),
    "perf/step_time": ("step_time", "step time (s)", False),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", required=True)
    p.add_argument("--out", required=True, help="output directory for PNGs")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    series = parse_log(Path(args.log))
    if not series:
        print(f"no step/rollout metric lines found in {args.log}")
        return
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    by_png: dict[str, list[str]] = defaultdict(list)
    for key in series:
        if key in _GROUPS:
            by_png[_GROUPS[key][0]].append(key)

    written = []
    for png, keys in by_png.items():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for key in keys:
            pts = sorted(series[key])
            ax.plot([s for s, _ in pts], [v for _, v in pts],
                    label=key, linewidth=1.2)
        _, ylabel, logy = _GROUPS[keys[0]]
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if len(keys) > 1:
            ax.legend(fontsize=8)
        ax.set_title(args.title or Path(args.log).stem, fontsize=10)
        path = out / f"{png}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))

    n_steps = max((s for pts in series.values() for s, _ in pts), default=0)
    print(f"parsed {len(series)} metric series up to step {n_steps}")
    for w in written:
        print("wrote", w)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
