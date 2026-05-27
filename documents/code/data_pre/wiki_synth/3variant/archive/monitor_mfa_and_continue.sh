#!/bin/bash
set -euo pipefail

# Monitor MFA job 43387 and run normalize → train when complete.
# TTS and merge already done. MFA resubmitted after fixing CLI args.

# ======Configuration=====
MFA_JOB_ID="43446"
POLL_INTERVAL=300

CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"

NORMALIZE_SCRIPT="${REPO_ROOT}/documents/code/data_pre/wiki_synth/3variant/build_train_3variant.py"
TRAIN_SCRIPT="${REPO_ROOT}/documents/code/train/term_train/run_3variant_1m_aries.sh"
# ======Configuration=====

export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "================================================================"
echo "[MONITOR] Waiting for MFA job ${MFA_JOB_ID} to complete..."
echo "[MONITOR] Start: $(date)"
echo "================================================================"

while true; do
    RUNNING=$(squeue -j "${MFA_JOB_ID}" --noheader 2>/dev/null | wc -l)
    if [ "${RUNNING}" -eq 0 ]; then
        FAILED=$(sacct -j "${MFA_JOB_ID}" --format=State --noheader --parsable2 2>/dev/null | grep -ciE "FAILED|CANCELLED|TIMEOUT" || true)
        if [ "${FAILED}" -gt 0 ]; then
            echo "[ERROR] MFA job ${MFA_JOB_ID} has failed tasks!"
            sacct -j "${MFA_JOB_ID}" --format=JobID%-15,State%-15,ExitCode,Elapsed
            exit 1
        fi
        echo "[MONITOR] MFA job ${MFA_JOB_ID} completed at $(date)"
        break
    fi
    echo "  [$(date +%H:%M:%S)] MFA still running (${RUNNING} tasks in queue)..."
    sleep "${POLL_INTERVAL}"
done

echo ""
echo "[Step 1] Running build_train_3variant.py (normalize)..."
python "${NORMALIZE_SCRIPT}"
echo "[Step 1] Normalization completed at $(date)."

echo ""
echo "[Step 2] Submitting training job on aries..."
TRAIN_JOB_ID=$(sbatch --parsable "${TRAIN_SCRIPT}")
echo "[Step 2] Training job submitted: ${TRAIN_JOB_ID}"

echo ""
echo "================================================================"
echo "[PIPELINE] Normalize + train completed!"
echo "[PIPELINE] Training job: ${TRAIN_JOB_ID}"
echo "[PIPELINE] End: $(date)"
echo "================================================================"
