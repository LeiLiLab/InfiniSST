#!/bin/bash
#SBATCH --job-name=tts_gs_v2_full
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=2-12:00:00
#SBATCH --array=0-31
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%A-%a_tts_gs_v2_full.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%A-%a_tts_gs_v2_full.err
#SBATCH --chdir=/tmp

# Full-scale 3M wiki_synth TTS regenerate using v2 GigaSpeech voice pool
# (9,989 unique-opus prompts vs v1's 6,206 effective speakers).
# Diffs vs run_tts_3variant_gigaspeech_poc_taurus.sh:
#   1. DATA_PATH: full 3M utterances (not 5k POC).
#   2. SPEAKER_DIR: /mnt/gemini/home/jiaxuanluo/gigaspeech_speaker_prompts (v2).
#   3. NOISE_DIR: "" (CLEAN-ONLY).  Pre-rendered noisy WAVs proved unhelpful
#      (see notes_voicepool_a1.md).  Noise injection, if ever, will happen
#      downstream at MFA-chunk time or on-the-fly in training.
#   4. OUTPUT_DIR: gemini/home (taurus/data and aries/data* near full).
#   5. TOTAL_SHARDS=32 (was 6 / 8 previously): ~94k utterances/shard.
#   6. output_jsonl_prefix bumped so shards don't collide with v1 output.
#
# Expected throughput: v1 ran ~40 ut/s per GPU; 32 GPU-shards × 40 ut/s
# = 1280 ut/s global → 3M / 1280 ≈ 39 min wall-time per shard. Use
# time=2d-12h as generous ceiling; actual should be <6 h per shard.

set -euo pipefail

COSYVOICE_ROOT="/mnt/gemini/home/jiaxuanluo/CosyVoice"
CONDA_ENV="/mnt/gemini/home/jiaxuanluo/miniconda3/envs/cosyvoice_vllm"

SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/tts/different_variants/rag_tts_multispeaker_noise.py"
DATA_PATH="/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/3variant/wiki_synth_utterances_3variant.jsonl"
OUTPUT_DIR="/mnt/gemini/home/jiaxuanluo/wiki_synth_tts_3variant_gigaspeech_full"
MODEL_DIR="${COSYVOICE_ROOT}/pretrained_models/Fun-CosyVoice3-0.5B"

SPEAKER_DIR="/mnt/gemini/home/jiaxuanluo/gigaspeech_speaker_prompts"
NOISE_DIR=""

TOTAL_SHARDS=32
BATCH_SIZE=16
SNR_LOW=5
SNR_HIGH=25
OUTPUT_JSONL_PREFIX="wiki_synth_3variant_gs_v2_clean_with_tts"

export PATH="${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export COSYVOICE_ROOT="${COSYVOICE_ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

LOCAL_TMP_DIR="/tmp/${USER}_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}/pytorch_tmp"
mkdir -p "${LOCAL_TMP_DIR}"
export TMPDIR="${LOCAL_TMP_DIR}"
export TMP="${LOCAL_TMP_DIR}"
export TEMP="${LOCAL_TMP_DIR}"
trap '[ -n "${LOCAL_TMP_DIR:-}" ] && rm -rf "${LOCAL_TMP_DIR}"' EXIT

SHARD_ID="${SLURM_ARRAY_TASK_ID}"

echo "================================================================"
echo "[TTS][GS-V2-FULL] Shard ${SHARD_ID}/${TOTAL_SHARDS}"
echo "[TTS][GS-V2-FULL] DATA       = ${DATA_PATH}"
echo "[TTS][GS-V2-FULL] SPEAKER    = ${SPEAKER_DIR}"
echo "[TTS][GS-V2-FULL] NOISE      = '${NOISE_DIR}' (empty = CLEAN-ONLY)"
echo "[TTS][GS-V2-FULL] OUTPUT     = ${OUTPUT_DIR}"
echo "[TTS][GS-V2-FULL] start at $(date)"
echo "================================================================"

mkdir -p "${OUTPUT_DIR}"

if [ -z "${NOISE_DIR}" ]; then
    python "${SCRIPT_PATH}" \
        --data "${DATA_PATH}" \
        --output-dir "${OUTPUT_DIR}" \
        --model-dir "${MODEL_DIR}" \
        --speaker-dir "${SPEAKER_DIR}" \
        --noise-dir "" \
        --shard-id "${SHARD_ID}" \
        --num-shards "${TOTAL_SHARDS}" \
        --batch-size "${BATCH_SIZE}" \
        --snr-low "${SNR_LOW}" \
        --snr-high "${SNR_HIGH}" \
        --no_dedup \
        --output_jsonl_prefix "${OUTPUT_JSONL_PREFIX}"
else
    python "${SCRIPT_PATH}" \
        --data "${DATA_PATH}" \
        --output-dir "${OUTPUT_DIR}" \
        --model-dir "${MODEL_DIR}" \
        --speaker-dir "${SPEAKER_DIR}" \
        --noise-dir "${NOISE_DIR}" \
        --shard-id "${SHARD_ID}" \
        --num-shards "${TOTAL_SHARDS}" \
        --batch-size "${BATCH_SIZE}" \
        --snr-low "${SNR_LOW}" \
        --snr-high "${SNR_HIGH}" \
        --no_dedup \
        --output_jsonl_prefix "${OUTPUT_JSONL_PREFIX}"
fi

echo "[TTS][GS-V2-FULL] shard ${SHARD_ID} done at $(date)"
