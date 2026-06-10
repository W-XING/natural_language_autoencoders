#!/usr/bin/env bash
# Bootstrap a fresh runpod/pytorch pod for Phase −1 gpt-oss-20b diagnostics.
#
# Light setup, NOT the full training stack: Phase −1 needs torch + transformers
# (MXFP4 path) + sglang (−1.A server) + this repo + a patched miles CHECKOUT
# on PYTHONPATH (−1.B imports miles.utils.mask_utils only — no build_conda).
# The full docs/setup.md stack is for the billed phases / the baked image.
#
# Expects the NLA repo at /workspace/natural_language_autoencoders (rsync'd
# from the control pod — the gpt-oss-20b branch is not on origin).
#
# Usage (on the GPU pod): bash natural_language_autoencoders/scripts/phase_minus1_gpt_oss/pod_setup.sh
set -euxo pipefail
cd /workspace
NLA_REPO=/workspace/natural_language_autoencoders
[ -d "$NLA_REPO" ] || { echo "NLA repo not found at $NLA_REPO — rsync it first"; exit 1; }

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq tmux git rsync curl

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv /workspace/venv --python 3.11
source /workspace/venv/bin/activate

# sglang first — it pins torch/transformers to its tested versions (expected,
# per docs/setup.md). kernels/accelerate: MXFP4 load path on Hopper.
# kernels==0.9.0: newer kernels (0.10–0.15) crash at import against
# huggingface_hub<1.0 (StrictDataclassFieldValidationError on `str | None`),
# and transformers 4.57 requires hub<1.0 — 0.9.0 is the latest that imports.
uv pip install "sglang[all]>=0.5.6"
uv pip install accelerate "kernels==0.9.0"
uv pip install -e "$NLA_REPO"

# miles checkout at the pin + integration patches — PYTHONPATH import only.
if [ ! -d /workspace/miles ]; then
    git clone https://github.com/radixark/miles.git /workspace/miles
    cd /workspace/miles
    git checkout "$(cut -d@ -f2 "$NLA_REPO/nla/miles_patches/UPSTREAM_PIN")"
    git apply "$NLA_REPO"/nla/miles_patches/*.patch
    cd /workspace
fi
grep -q 'PYTHONPATH=/workspace/miles' ~/.bashrc || cat >> ~/.bashrc <<'RC'
export PYTHONPATH=/workspace/miles:$PYTHONPATH
source /workspace/venv/bin/activate
RC

mkdir -p /data/logs   # diagnostics default --out here (no network volume in Phase −1)

PYTHONPATH=/workspace/miles python - <<'PY'
import torch, transformers, sglang
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "| transformers", transformers.__version__, "| sglang", sglang.__version__)
import nla.schema
from miles.utils import mask_utils
print("nla + miles.utils.mask_utils import OK")
PY
echo "pod_setup complete"
