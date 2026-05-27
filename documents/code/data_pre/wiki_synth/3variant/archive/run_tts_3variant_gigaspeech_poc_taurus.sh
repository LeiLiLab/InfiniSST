#!/bin/bash
#SBATCH --job-name=tts_gs_voice_poc
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00:00
#SBATCH --array=0-5
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%A-%a_tts_gs_voice_poc.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%A-%a_tts_gs_voice_poc.err
#SBATCH --chdir=/tmp

# 5k-sample POC of CosyVoice zero-shot TTS using GigaSpeech voice references
# instead of VCTK (109 UK studio speakers).  Validates pipeline + spot-check
# audio quality before committing to the full 500k-3M regenerate.
#
# Only 2 diffs vs run_tts_3variant_taurus_helper.sh:
#   1. SPEAKER_DIR now points at the 10k gigaspeech_speaker_prompts pool
#      (built via build_gigaspeech_voice_pool.py).
#   2. DATA_PATH is the 5k POC subset.
# Same CosyVoice script, same model, same shard scheme.

set -euo pipefail

# ======Configuration=====
COSYVOICE_ROOT="/mnt/gemini/home/jiaxuanluo/CosyVoice"
CONDA_ENV="/mnt/gemini/home/jiaxuanluo/miniconda3/envs/cosyvoice_vllm"

SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/tts/different_variants/rag_tts_multispeaker_noise.py"
DATA_PATH="/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/3variant_gigaspeech_poc/wiki_synth_utterances_poc5k.jsonl"
OUTPUT_DIR="/mnt/aries/data6/jiaxuanluo/wiki_synth_tts_3variant_gigaspeech_poc"
MODEL_DIR="${COSYVOICE_ROOT}/pretrained_models/Fun-CosyVoice3-0.5B"

SPEAKER_DIR="/mnt/taurus/data/jiaxuanluo/gigaspeech_speaker_prompts"
NOISE_DIR="/mnt/taurus/data/siqiouyang/datasets/wham_wav"

# 6 shards on 6 GPUs = full taurus allocation.  5000 / 6 ≈ 833 per shard.
TOTAL_SHARDS=6
BATCH_SIZE=16
SNR_LOW=5
SNR_HIGH=25
OUTPUT_JSONL_PREFIX="wiki_synth_3variant_gigaspeech_poc_with_tts"
# ======Configuration=====

export PATH="${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export COSYVOICE_ROOT="${COSYVOICE_ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Use /tmp — not /dev/shm which has tmpfs-cleaner races on taurus/aries.
LOCAL_TMP_DIR="/tmp/${USER}_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}/pytorch_tmp"
mkdir -p "${LOCAL_TMP_DIR}"
export TMPDIR="${LOCAL_TMP_DIR}"
export TMP="${LOCAL_TMP_DIR}"
export TEMP="${LOCAL_TMP_DIR}"
trap '[ -n "${LOCAL_TMP_DIR:-}" ] && rm -rf "${LOCAL_TMP_DIR}"' EXIT

SHARD_ID="${SLURM_ARRAY_TASK_ID}"

echo "================================================================"
echo "[TTS][GS-POC] Shard ${SHARD_ID}/${TOTAL_SHARDS}"
echo "[TTS][GS-POC] DATA       = ${DATA_PATH}"
echo "[TTS][GS-POC] SPEAKER    = ${SPEAKER_DIR}"
echo "[TTS][GS-POC] OUTPUT     = ${OUTPUT_DIR}"
echo "[TTS][GS-POC] start at $(date)"
echo "================================================================"

mkdir -p "${OUTPUT_DIR}"

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

echo "[TTS][GS-POC] shard ${SHARD_ID} done at $(date)"
