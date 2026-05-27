#!/bin/bash
set -euo pipefail

# Monitor TTS job 43281 and automatically trigger the merge → MFA → normalize → train pipeline.
# Run this in a tmux/screen session.

# ======Configuration=====
TTS_JOB_ID="43281"
POLL_INTERVAL=300

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
echo "[MONITOR] Waiting for TTS job ${TTS_JOB_ID} to complete..."
echo "[MONITOR] Start: $(date)"
echo "================================================================"

# ---------- Phase 1: Wait for TTS job to complete ----------
while true; do
    RUNNING=$(squeue -j "${TTS_JOB_ID}" --noheader 2>/dev/null | wc -l)
    if [ "${RUNNING}" -eq 0 ]; then
        FAILED=$(sacct -j "${TTS_JOB_ID}" --format=State --noheader --parsable2 2>/dev/null | grep -ciE "FAILED|CANCELLED|TIMEOUT" || true)
        if [ "${FAILED}" -gt 0 ]; then
            echo "[ERROR] TTS job ${TTS_JOB_ID} has failed tasks!"
            sacct -j "${TTS_JOB_ID}" --format=JobID%-15,State%-15,ExitCode,Elapsed
            exit 1
        fi
        echo "[MONITOR] TTS job ${TTS_JOB_ID} all tasks completed at $(date)"
        break
    fi
    echo "  [$(date +%H:%M:%S)] TTS still running (${RUNNING} tasks in queue)..."
    sleep "${POLL_INTERVAL}"
done

# ---------- Phase 2: Verify all TTS shards ----------
echo ""
echo "[Step 1] Verifying TTS shards..."
for i in $(seq 0 $((TOTAL_TTS_SHARDS - 1))); do
    SHARD_PATH="${TTS_SHARD_DIR}/wiki_synth_3variant_with_tts_shard${i}.jsonl"
    if [ ! -f "${SHARD_PATH}" ]; then
        echo "[ERROR] Missing TTS shard ${i}: ${SHARD_PATH}"
        exit 1
    fi
    LINE_COUNT=$(wc -l < "${SHARD_PATH}")
    echo "  Shard ${i}: ${LINE_COUNT} lines"
done
echo "[Step 1] All ${TOTAL_TTS_SHARDS} TTS shards verified."

# ---------- Phase 3: Merge TTS shards ----------
echo ""
echo "[Step 2] Running merge_tts_3variant.py..."
python "${MERGE_SCRIPT}"
echo "[Step 2] Merge completed at $(date)."

# ---------- Phase 4: Submit MFA job ----------
echo ""
echo "[Step 3] Submitting MFA job on aries..."
MFA_JOB_ID=$(sbatch --parsable "${MFA_SCRIPT}")
echo "[Step 3] MFA job submitted: ${MFA_JOB_ID}"
echo "[Step 3] Waiting for MFA to complete..."

while true; do
    RUNNING=$(squeue -j "${MFA_JOB_ID}" --noheader 2>/dev/null | wc -l)
    if [ "${RUNNING}" -eq 0 ]; then
        FAILED=$(sacct -j "${MFA_JOB_ID}" --format=State --noheader --parsable2 2>/dev/null | grep -ciE "FAILED|CANCELLED|TIMEOUT" || true)
        if [ "${FAILED}" -gt 0 ]; then
            echo "[ERROR] MFA job ${MFA_JOB_ID} has failed tasks!"
            sacct -j "${MFA_JOB_ID}" --format=JobID%-15,State%-15,ExitCode,Elapsed
            exit 1
        fi
        echo "[Step 3] MFA job ${MFA_JOB_ID} completed at $(date)."
        break
    fi
    echo "  [$(date +%H:%M:%S)] MFA still running (${RUNNING} tasks in queue)..."
    sleep "${POLL_INTERVAL}"
done

# ---------- Phase 5: Normalize (build training JSONL) ----------
echo ""
echo "[Step 4] Running build_train_3variant.py..."
python "${NORMALIZE_SCRIPT}"
echo "[Step 4] Normalization completed at $(date)."

# ---------- Phase 6: Submit training job ----------
echo ""
echo "[Step 5] Submitting training job on aries..."
TRAIN_JOB_ID=$(sbatch --parsable "${TRAIN_SCRIPT}")
echo "[Step 5] Training job submitted: ${TRAIN_JOB_ID}"

echo ""
echo "================================================================"
echo "[PIPELINE] All steps completed!"
echo "[PIPELINE] MFA job: ${MFA_JOB_ID}"
echo "[PIPELINE] Training job: ${TRAIN_JOB_ID}"
echo "[PIPELINE] End: $(date)"
echo "================================================================"
