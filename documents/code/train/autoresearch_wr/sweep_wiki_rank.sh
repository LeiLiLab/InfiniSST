#!/bin/bash
# Greedy sweep: submit wiki_rank experiments as dependency chain on taurus 2-GPU.
# Each experiment modifies train.sh's WIKI_RANK, submits via sbatch, and chains
# the next experiment to run after the previous finishes.
#
# Usage: bash sweep_wiki_rank.sh
# Results: check wandb + slurm logs in /mnt/gemini/data1/jiaxuanluo/logs/autoresearch/

set -euo pipefail

# ======Configuration=====
TRAIN_SH="$(dirname "$0")/train.sh"
WIKI_RANKS=(250000 500000 1000000 2000000)
LOG_DIR="/mnt/gemini/data1/jiaxuanluo/logs/autoresearch"
# ======Configuration=====

mkdir -p "${LOG_DIR}"

assert_file_exists() {
    if [ ! -f "$1" ]; then
        echo "ERROR: $1 not found" >&2
        exit 1
    fi
}
assert_file_exists "${TRAIN_SH}"

PREV_JOB=""
echo "=== Wiki_rank Greedy Sweep ==="
echo "Values: ${WIKI_RANKS[*]}"
echo ""

for WR in "${WIKI_RANKS[@]}"; do
    WR_K=$((WR / 1000))

    # Patch WIKI_RANK in train.sh
    sed -i "s/^WIKI_RANK=.*/WIKI_RANK=${WR}/" "${TRAIN_SH}"
    echo "[SWEEP] Set WIKI_RANK=${WR} (${WR_K}k)"

    # Build sbatch command
    SBATCH_CMD="sbatch --parsable"
    if [ -n "${PREV_JOB}" ]; then
        SBATCH_CMD="${SBATCH_CMD} --dependency=afterany:${PREV_JOB}"
    fi
    SBATCH_CMD="${SBATCH_CMD} --job-name=wr_${WR_K}k ${TRAIN_SH}"

    # Submit
    JOB_ID=$(eval "${SBATCH_CMD}")
    echo "[SWEEP] Submitted wr=${WR_K}k → job ${JOB_ID} (dep: ${PREV_JOB:-none})"

    PREV_JOB="${JOB_ID}"
done

echo ""
echo "=== All ${#WIKI_RANKS[@]} experiments submitted as chain ==="
echo "Monitor: squeue -u \$USER -p taurus"
echo "Logs:    ls ${LOG_DIR}/"
echo "WandB:   qwen3_rag_autoresearch project"
