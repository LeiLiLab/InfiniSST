#!/bin/bash
set -euo pipefail

# Full 3-variant pipeline: merge TTS → submit MFA → (wait) → normalize → submit train
# Run this AFTER all TTS shards (43281) have completed.

# ======Configuration=====
CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"

MERGE_SCRIPT="${REPO_ROOT}/documents/code/data_pre/wiki_synth/3variant/merge_tts_3variant.py"
MFA_SCRIPT="${REPO_ROOT}/documents/code/data_pre/wiki_synth/3variant/run_mfa_3variant_aries.sh"
NORMALIZE_SCRIPT="${REPO_ROOT}/documents/code/data_pre/wiki_synth/3variant/build_train_3variant.py"
TRAIN_SCRIPT="${REPO_ROOT}/documents/code/train/term_train/run_3variant_1m_aries.sh"

TTS_SHARD_DIR="/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/3variant"
TOTAL_TTS_SHARDS=8
# ======Configuration=====

export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "================================================================"
echo "[PIPELINE] 3-variant: merge → MFA → normalize → train"
echo "[PIPELINE] Start: $(date)"
echo "================================================================"

# ---------- Step 1: Verify all TTS shards exist ----------
echo ""
echo "[Step 1] Verifying TTS shards..."
for i in $(seq 0 $((TOTAL_TTS_SHARDS - 1))); do
    SHARD_PATH="${TTS_SHARD_DIR}/wiki_synth_3variant_with_tts_shard${i}.jsonl"
    if [ ! -f "${SHARD_PATH}" ]; then
        echo "[ERROR] Missing TTS shard ${i}: ${SHARD_PATH}"
        echo "[ERROR] TTS job (43281) may not have completed. Aborting."
        exit 1
    fi
    LINE_COUNT=$(wc -l < "${SHARD_PATH}")
    echo "  Shard ${i}: ${LINE_COUNT} lines"
done
echo "[Step 1] All ${TOTAL_TTS_SHARDS} TTS shards verified."

# ---------- Step 2: Merge TTS shards ----------
echo ""
echo "[Step 2] Running merge_tts_3variant.py..."
python "${MERGE_SCRIPT}"
echo "[Step 2] Merge completed."

# ---------- Step 3: Submit MFA job ----------
echo ""
echo "[Step 3] Submitting MFA job on aries..."
MFA_JOB_ID=$(sbatch --parsable "${MFA_SCRIPT}")
echo "[Step 3] MFA job submitted: ${MFA_JOB_ID}"
echo "[Step 3] Waiting for MFA to complete (this may take hours)..."

# Poll until MFA completes
while true; do
    STATES=$(sacct -j "${MFA_JOB_ID}" --format=State --noheader --parsable2 2>/dev/null | sort -u | tr '\n' ',')
    if echo "${STATES}" | grep -qiE "FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY"; then
        echo "[ERROR] MFA job ${MFA_JOB_ID} has failed state(s): ${STATES}"
        exit 1
    fi
    RUNNING=$(squeue -j "${MFA_JOB_ID}" --noheader 2>/dev/null | wc -l)
    if [ "${RUNNING}" -eq 0 ]; then
        ALL_COMPLETED=$(sacct -j "${MFA_JOB_ID}" --format=State --noheader --parsable2 2>/dev/null | grep -v "^$" | sort -u)
        if echo "${ALL_COMPLETED}" | grep -qi "COMPLETED"; then
            echo "[Step 3] MFA job ${MFA_JOB_ID} completed."
            break
        fi
    fi
    echo "  [$(date +%H:%M:%S)] MFA still running... (states: ${STATES})"
    sleep 300
done

# ---------- Step 4: Normalize (build training JSONL) ----------
echo ""
echo "[Step 4] Running build_train_3variant.py (normalize)..."
python "${NORMALIZE_SCRIPT}"
echo "[Step 4] Normalization completed."

# ---------- Step 5: Submit training job ----------
echo ""
echo "[Step 5] Submitting training job on aries..."
TRAIN_JOB_ID=$(sbatch --parsable "${TRAIN_SCRIPT}")
echo "[Step 5] Training job submitted: ${TRAIN_JOB_ID}"

echo ""
echo "================================================================"
echo "[PIPELINE] All steps completed."
echo "[PIPELINE] MFA job: ${MFA_JOB_ID}"
echo "[PIPELINE] Training job: ${TRAIN_JOB_ID}"
echo "[PIPELINE] End: $(date)"
echo "================================================================"
