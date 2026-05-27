#!/bin/bash
#SBATCH --job-name=mfa_gsv2_p020
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-19
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%A-%a_mfa_gsv2_p020.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%A-%a_mfa_gsv2_p020.err
#SBATCH --chdir=/tmp

# MFA alignment + chunk cutting for the partial GSV2 clean-only TTS set.
# Input uses local completed TTS shards 0-20 only; outputs are isolated from
# the later full 32-shard GSV2 pipeline.

set -euo pipefail

CONDA_ENV="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/mfa"
SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/wiki_synth/align_and_cut_wiki_synth.py"
MFA_MODEL_DIR="/mnt/taurus/home/jiaxuanluo/Documents/MFA"

DATA_JSONL="/mnt/gemini/home/jiaxuanluo/wiki_synth_tts_3variant_gigaspeech_partial0_20/wiki_synth_3variant_gs_v2_clean_partial0_20.jsonl"
WORK_DIR="/mnt/gemini/home/jiaxuanluo/MFA/3variant_gsv2_partial0_20/work"
OUTPUT_AUDIO_DIR="/mnt/gemini/home/jiaxuanluo/MFA/3variant_gsv2_partial0_20/chunks"
OUTPUT_JSONL_DIR="/mnt/gemini/home/jiaxuanluo/MFA/3variant_gsv2_partial0_20/output"
NOISE_DIR=""

NUM_SHARDS=20

export PATH="${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

SHARD_ID="${SLURM_ARRAY_TASK_ID}"

export MFA_ROOT_DIR="/mnt/gemini/home/jiaxuanluo/MFA/mfa_root_gsv2_p020_${SHARD_ID}"
export TMPDIR="/mnt/gemini/home/jiaxuanluo/MFA/tmp_gsv2_p020_${SHARD_ID}"
export HOME="/mnt/gemini/home/jiaxuanluo/MFA/mfa_home_gsv2_p020_${SHARD_ID}"
mkdir -p "${MFA_ROOT_DIR}" "${TMPDIR}" "${HOME}"

echo "================================================================"
echo "[MFA][GSV2-PARTIAL0_20] 3-variant alignment (clean-only) - taurus"
echo "[MFA][GSV2-PARTIAL0_20] Shard ${SHARD_ID}/${NUM_SHARDS}"
echo "[MFA][GSV2-PARTIAL0_20] Input      = ${DATA_JSONL}"
echo "[MFA][GSV2-PARTIAL0_20] Work dir   = ${WORK_DIR}"
echo "[MFA][GSV2-PARTIAL0_20] Chunks     = ${OUTPUT_AUDIO_DIR}"
echo "[MFA][GSV2-PARTIAL0_20] JSONL dir  = ${OUTPUT_JSONL_DIR}"
echo "[MFA][GSV2-PARTIAL0_20] Noise dir  = '${NOISE_DIR}' (empty = CLEAN-ONLY)"
echo "[MFA][GSV2-PARTIAL0_20] Start: $(date)"
echo "================================================================"

if [ ! -f "${DATA_JSONL}" ]; then
    echo "[MFA][ERROR] Missing merged partial JSONL: ${DATA_JSONL}" >&2
    exit 1
fi

mkdir -p "${WORK_DIR}" "${OUTPUT_AUDIO_DIR}" "${OUTPUT_JSONL_DIR}"

# Keep this filename pattern aligned with build_train_3variant.py defaults.
OUTPUT_JSONL="${OUTPUT_JSONL_DIR}/wiki_synth_train_shard_$(printf '%02d' "${SHARD_ID}").jsonl"

python "${SCRIPT_PATH}" \
    --data "${DATA_JSONL}" \
    --work-dir "${WORK_DIR}" \
    --output-audio-dir "${OUTPUT_AUDIO_DIR}" \
    --output-jsonl "${OUTPUT_JSONL}" \
    --shard-id "${SHARD_ID}" \
    --num-shards "${NUM_SHARDS}" \
    --mfa-conda-prefix "${CONDA_ENV}" \
    --noise-dir "${NOISE_DIR}" \
    --mfa-model-dir "${MFA_MODEL_DIR}"

echo "[MFA][GSV2-PARTIAL0_20] Shard ${SHARD_ID} completed at $(date)"
