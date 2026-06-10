#!/usr/bin/env bash
# Phase −1 driver — run all gpt-oss-20b showstopper diagnostics, collect
# verdicts, keep going on failure (we want ALL gate results for the go/no-go).
#
# −1.A needs an SGLang server up separately:
#   tmux new -s sglang
#   python -m sglang.launch_server --model openai/gpt-oss-20b \
#       --disable-radix-cache --port 30000
# Pass SKIP_SGLANG=1 to defer it.
#
# Usage: bash scripts/phase_minus1_gpt_oss/run_all.sh [logdir]
set -u -o pipefail  # without pipefail, `| tee` masks the python exit code → false PASS
LOGDIR="${1:-/data/logs}"
mkdir -p "$LOGDIR"
cd "$(dirname "$0")/../.."

declare -A RESULTS
run() {
    local name="$1"; shift
    echo -e "\n========== $name =========="
    if "$@" --out "$LOGDIR/${name}.json" 2>&1 | tee "$LOGDIR/${name}.log"; then
        RESULTS[$name]=PASS
    else
        RESULTS[$name]=FAIL
    fi
}

# CPU-only checks first — fail fast before touching the GPU.
run diag_m1_D    python scripts/phase_minus1_gpt_oss/diag_m1_D_neighbor_roundtrip.py
run diag_m1_B    python scripts/phase_minus1_gpt_oss/diag_m1_B_harmony_lossmask.py

# GPU checks.
run diag_m1_0      python scripts/phase_minus1_gpt_oss/diag_m1_0_expert_trainability.py
run diag_m1_Bprime python scripts/phase_minus1_gpt_oss/diag_m1_Bprime_critic_truncation.py --device-map cuda:0
run diag_phase0    python scripts/phase_minus1_gpt_oss/diag_phase0_inj_scale.py

if [ "${SKIP_SGLANG:-0}" != "1" ]; then
    run diag_m1_A python scripts/phase_minus1_gpt_oss/diag_m1_A_sglang_embeds.py
else
    RESULTS[diag_m1_A]=SKIPPED
fi

echo -e "\n================ Phase −1 summary ================"
for k in diag_m1_0 diag_m1_A diag_m1_B diag_m1_Bprime diag_m1_D diag_phase0; do
    printf "%-16s %s\n" "$k" "${RESULTS[$k]:-NOT_RUN}"
done
echo "verdicts + details: $LOGDIR/*.json"
echo "Phase −1 ends here — STOP for go/no-go."
