#!/bin/bash
#SBATCH --job-name=tts_gs_v2_poc
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0-03:00:00
#SBATCH --array=0-5
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%A-%a_tts_gs_v2_poc.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%A-%a_tts_gs_v2_poc.err
#SBATCH --chdir=/tmp

# POC for v2 GigaSpeech voice pool (9,989 unique-opus prompts) + CLEAN-ONLY.
# Double role:
#   (a) sanity check that rag_tts_multispeaker_noise.py handles --noise-dir ""
#       (= no noisy branch) without crashing;
#   (b) produce 5k clean WAVs off the new voice pool for spot-check listening
#       and A/B vs the v1 5k POC.
# Diffs vs run_tts_3variant_gigaspeech_poc_taurus.sh:
#   - SPEAKER_DIR  -> gemini/home/gigaspeech_speaker_prompts (v2).
#   - NOISE_DIR    -> "" (no noisy branch).
#   - OUTPUT_DIR   -> gemini/home.
#   - output_jsonl_prefix bumped to _v2_clean for disjoint writes.

set -euo pipefail

COSYVOICE_ROOT="/mnt/gemini/home/jiaxuanluo/CosyVoice"
CONDA_ENV="/mnt/gemini/home/jiaxuanluo/miniconda3/envs/cosyvoice_vllm"

SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/tts/different_variants/rag_tts_multispeaker_noise.py"
DATA_PATH="/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/3variant_gigaspeech_poc/wiki_synth_utterances_poc5k.jsonl"
OUTPUT_DIR="/mnt/gemini/home/jiaxuanluo/wiki_synth_tts_3variant_gigaspeech_poc_v2"
MODEL_DIR="${COSYVOICE_ROOT}/pretrained_models/Fun-CosyVoice3-0.5B"

SPEAKER_DIR="/mnt/gemini/home/jiaxuanluo/gigaspeech_speaker_prompts"
NOISE_DIR=""

TOTAL_SHARDS=6
BATCH_SIZE=16
OUTPUT_JSONL_PREFIX="wiki_synth_3variant_gs_v2_clean_poc_with_tts"

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
echo "[TTS][GS-V2-POC] Shard ${SHARD_ID}/${TOTAL_SHARDS} (CLEAN-ONLY, v2 pool)"
echo "[TTS][GS-V2-POC] DATA       = ${DATA_PATH}"
echo "[TTS][GS-V2-POC] SPEAKER    = ${SPEAKER_DIR}"
echo "[TTS][GS-V2-POC] OUTPUT     = ${OUTPUT_DIR}"
echo "[TTS][GS-V2-POC] start at $(date)"
echo "================================================================"

mkdir -p "${OUTPUT_DIR}"

python "${SCRIPT_PATH}" \
    --data "${DATA_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-dir "${MODEL_DIR}" \
    --speaker-dir "${SPEAKER_DIR}" \
    --noise-dir "" \
    --shard-id "${SHARD_ID}" \
    --num-shards "${TOTAL_SHARDS}" \
    --batch-size "${BATCH_SIZE}" \
    --no_dedup \
    --output_jsonl_prefix "${OUTPUT_JSONL_PREFIX}"

echo "[TTS][GS-V2-POC] shard ${SHARD_ID} done at $(date)"
