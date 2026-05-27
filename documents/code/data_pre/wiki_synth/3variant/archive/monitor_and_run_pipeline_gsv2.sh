#!/bin/bash
set -euo pipefail

# v2 (GigaSpeech voice pool, CLEAN-ONLY) 3variant pipeline chainer.
# Monitors TTS job -> merges 32 shards -> submits MFA (20 shards) -> builds
# TRAIN_JSONL -> submits A1 retriever training.  Run in tmux/screen.

# ======Configuration=====
TTS_JOB_ID="${TTS_JOB_ID:-43871}"
POLL_INTERVAL="${POLL_INTERVAL:-300}"

CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"

MERGE_SCRIPT="${REPO_ROOT}/documents/code/data_pre/wiki_synth/3variant/merge_tts_3variant.py"
MFA_SCRIPT="${REPO_ROOT}/documents/code/data_pre/wiki_synth/3variant/run_mfa_3variant_gigaspeech_taurus.sh"
NORMALIZE_SCRIPT="${REPO_ROOT}/documents/code/data_pre/wiki_synth/3variant/build_train_3variant.py"
TRAIN_SCRIPT="${REPO_ROOT}/documents/code/train/term_train/run_voice_pool_a1_clean_taurus.sh"

# WAVs go to gemini/home (big); JSONL shards land next to the *input* JSONL per
# rag_tts_multispeaker_noise.py (dirname(args.data)).
TTS_WAV_DIR="/mnt/gemini/home/jiaxuanluo/wiki_synth_tts_3variant_gigaspeech_full"
TTS_JSONL_DIR="/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/3variant"
TTS_SHARD_PREFIX="wiki_synth_3variant_gs_v2_clean_with_tts"
TOTAL_TTS_SHARDS=32
MERGED_JSONL="${TTS_WAV_DIR}/wiki_synth_3variant_gs_v2_clean_dual.jsonl"

MFA_OUTPUT_DIR="/mnt/gemini/home/jiaxuanluo/MFA/3variant_gsv2/output"
TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_clean_mfa.jsonl"
# ======Configuration=====

export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "================================================================"
echo "[MONITOR][GSV2] Chaining TTS(${TTS_JOB_ID}) -> merge -> MFA -> build -> train"
echo "[MONITOR][GSV2] Start: $(date)"
echo "================================================================"

wait_job() {
    local job_id="$1"
    local label="$2"
    while true; do
        RUNNING=$(squeue -j "${job_id}" --noheader 2>/dev/null | wc -l)
        if [ "${RUNNING}" -eq 0 ]; then
            FAILED=$(sacct -j "${job_id}" --format=State --noheader --parsable2 2>/dev/null | grep -ciE "FAILED|CANCELLED|TIMEOUT|NODE_FAIL" || true)
            if [ "${FAILED}" -gt 0 ]; then
                echo "[ERROR] ${label} job ${job_id} has failed tasks!"
                sacct -j "${job_id}" --format=JobID%-15,State%-15,ExitCode,Elapsed
                exit 1
            fi
            echo "[MONITOR] ${label} job ${job_id} completed at $(date)"
            return 0
        fi
        echo "  [$(date +%H:%M:%S)] ${label} still running (${RUNNING} tasks)..."
        sleep "${POLL_INTERVAL}"
    done
}

# ---------- Phase 1: Wait for TTS ----------
wait_job "${TTS_JOB_ID}" "TTS"

# ---------- Phase 2: Verify TTS shards ----------
echo ""
echo "[Step 1] Verifying ${TOTAL_TTS_SHARDS} TTS shards..."
for i in $(seq 0 $((TOTAL_TTS_SHARDS - 1))); do
    SHARD_PATH="${TTS_JSONL_DIR}/${TTS_SHARD_PREFIX}_shard${i}.jsonl"
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
echo "[Step 2] Merging 32 TTS shards -> ${MERGED_JSONL} ..."
mkdir -p "$(dirname "${MERGED_JSONL}")"
python "${MERGE_SCRIPT}" \
    --shard_dir "${TTS_JSONL_DIR}" \
    --shard_prefix "${TTS_SHARD_PREFIX}" \
    --total_shards "${TOTAL_TTS_SHARDS}" \
    --output "${MERGED_JSONL}"
echo "[Step 2] Merge completed at $(date)."

# ---------- Phase 4: Submit MFA ----------
echo ""
echo "[Step 3] Submitting MFA job (taurus)..."
MFA_JOB_ID=$(sbatch --parsable "${MFA_SCRIPT}")
echo "[Step 3] MFA job submitted: ${MFA_JOB_ID}"
wait_job "${MFA_JOB_ID}" "MFA"

# ---------- Phase 5: Build training JSONL (normalize + GS merge) ----------
echo ""
echo "[Step 4] Building ${TRAIN_JSONL} (v2 MFA + GigaSpeech) ..."
python "${NORMALIZE_SCRIPT}" \
    --mfa-dir "${MFA_OUTPUT_DIR}" \
    --output-train "${TRAIN_JSONL}"
echo "[Step 4] TRAIN_JSONL built at $(date)."

# ---------- Phase 6: Submit training ----------
echo ""
echo "[Step 5] Submitting A1 voice-pool training (taurus, 6 GPUs)..."
TRAIN_JOB_ID=$(sbatch --parsable "${TRAIN_SCRIPT}")
echo "[Step 5] Training submitted: ${TRAIN_JOB_ID}"

echo ""
echo "================================================================"
echo "[PIPELINE] Chain complete.  MFA=${MFA_JOB_ID}  TRAIN=${TRAIN_JOB_ID}"
echo "[PIPELINE] End: $(date)"
echo "================================================================"
