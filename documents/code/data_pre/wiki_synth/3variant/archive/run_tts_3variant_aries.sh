#!/bin/bash
#SBATCH --job-name=tts_3var
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --array=0-7
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%A-%a_tts_3var.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%A-%a_tts_3var.err
#SBATCH --chdir=/tmp

set -euo pipefail

# ======Configuration=====
COSYVOICE_ROOT="/mnt/gemini/home/jiaxuanluo/CosyVoice"
CONDA_ENV="/mnt/gemini/home/jiaxuanluo/miniconda3/envs/cosyvoice_vllm"

SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/tts/different_variants/rag_tts_multispeaker_noise.py"
DATA_PATH="/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/3variant/wiki_synth_utterances_3variant.jsonl"
OUTPUT_DIR="/mnt/aries/data6/jiaxuanluo/wiki_synth_tts_3variant"
MODEL_DIR="${COSYVOICE_ROOT}/pretrained_models/Fun-CosyVoice3-0.5B-aries"

SPEAKER_DIR="/mnt/taurus/data/siqiouyang/datasets/vctk_speaker_prompts"
NOISE_DIR="/mnt/taurus/data/siqiouyang/datasets/wham_wav"

TOTAL_SHARDS=8
BATCH_SIZE=16
SNR_LOW=5
SNR_HIGH=25
OUTPUT_JSONL_PREFIX="wiki_synth_3variant_with_tts"
# ======Configuration=====

export PATH="${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export COSYVOICE_ROOT="${COSYVOICE_ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

LOCAL_TMP_DIR="/dev/shm/${USER}/pytorch_tmp"
mkdir -p "${LOCAL_TMP_DIR}"
export TMPDIR="${LOCAL_TMP_DIR}"
export TMP="${LOCAL_TMP_DIR}"
export TEMP="${LOCAL_TMP_DIR}"

SHARD_ID="${SLURM_ARRAY_TASK_ID}"

echo "================================================================"
echo "[TTS] 3-variant top-1M multi-speaker + noise - aries"
echo "[TTS] Shard ${SHARD_ID}/${TOTAL_SHARDS}"
echo "[TTS] Data: ${DATA_PATH}"
echo "[TTS] Output: ${OUTPUT_DIR}"
echo "[TTS] Model: ${MODEL_DIR}"
echo "[TTS] Speakers: ${SPEAKER_DIR}"
echo "[TTS] Noise: ${NOISE_DIR} (SNR=[${SNR_LOW},${SNR_HIGH}]dB)"
echo "[TTS] Start: $(date)"
echo "================================================================"

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

echo "[TTS] Shard ${SHARD_ID} completed at $(date)"
