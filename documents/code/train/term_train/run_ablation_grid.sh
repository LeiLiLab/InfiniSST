#!/bin/bash
# Submits the 4 ablation variants to the cluster, balanced across Aries
# and Gemini partitions so at most 2 jobs run concurrently per partition
# (each variant takes a full 8-GPU node).
#
# Variants:
#   1. baseline      -> aries   (reference run)
#   2. hardneg_k128  -> gemini  (A1: +hard-neg scale)
#   3. mfa_smallest  -> aries   (A2a: tightest covering window)
#   4. mfa_logsumexp -> gemini  (A2b: all covering via LSE)
#
# Usage:
#   bash run_ablation_grid.sh
#
# This prints a table of (variant, partition, job_id) for later tracking.

set -euo pipefail

# ======Configuration=====
LAUNCHER="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_hardneg_ablation_aries.sh"
LOG_DIR="/mnt/gemini/data1/jiaxuanluo/logs"
# Variant -> partition assignment.  Alternating between aries/gemini so that
# within a single partition the two variants run sequentially (partition can
# only host one 8-GPU job at a time).
# NOTE: Gemini currently has stale GPU processes consuming memory even when SLURM
# reports it idle (zombie instructpro processes). Until that is cleaned, queue
# all 4 variants to aries sequentially. SLURM will FIFO them 1-at-a-time.
VARIANT_BASELINE_PARTITION="aries"
VARIANT_K128_PARTITION="aries"
VARIANT_SMALLEST_PARTITION="aries"
VARIANT_LOGSUMEXP_PARTITION="aries"
# ======Configuration=====

if [ ! -x "${LAUNCHER}" ]; then
    echo "Launcher not executable or missing: ${LAUNCHER}" >&2
    exit 1
fi
mkdir -p "${LOG_DIR}"

submit_variant() {
    local variant="$1"
    local partition="$2"
    local jobname="q3_abl_${variant}"
    local out_line
    out_line="$(sbatch \
        --partition="${partition}" \
        --job-name="${jobname}" \
        --export=ALL,ABLATION_VARIANT="${variant}" \
        "${LAUNCHER}")"
    # Expected format: "Submitted batch job <id>"
    local job_id
    job_id="$(echo "${out_line}" | awk '{print $NF}')"
    echo "${variant}|${partition}|${job_id}"
}

echo "[GRID] Submitting 4 ablation variants..."

R1="$(submit_variant baseline      "${VARIANT_BASELINE_PARTITION}")"
R2="$(submit_variant hardneg_k128  "${VARIANT_K128_PARTITION}")"
R3="$(submit_variant mfa_smallest  "${VARIANT_SMALLEST_PARTITION}")"
R4="$(submit_variant mfa_logsumexp "${VARIANT_LOGSUMEXP_PARTITION}")"

echo ""
echo "VARIANT            | PARTITION | JOB_ID"
echo "-------------------+-----------+--------"
printf "%-18s | %-9s | %s\n" $(echo "${R1}" | tr '|' ' ')
printf "%-18s | %-9s | %s\n" $(echo "${R2}" | tr '|' ' ')
printf "%-18s | %-9s | %s\n" $(echo "${R3}" | tr '|' ' ')
printf "%-18s | %-9s | %s\n" $(echo "${R4}" | tr '|' ' ')
echo ""
echo "[GRID] Monitor with: squeue -o '%.10i %.9P %.14j %.8u %.2t %.10M %.6D %R'"
echo "[GRID] Logs in: ${LOG_DIR}/<JOB_ID>_q3_ablation_<variant>.out"
