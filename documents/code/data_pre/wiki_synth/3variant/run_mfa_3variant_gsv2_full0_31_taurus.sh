#!/bin/bash
#SBATCH --job-name=mfa_gsv2_f031
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-31
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%A-%a_mfa_gsv2_f031.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%A-%a_mfa_gsv2_f031.err
#SBATCH --chdir=/tmp

# MFA alignment + chunk cutting for full GSV2 clean-only TTS shards 0-31.

set -euo pipefail

CONDA_ENV="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/mfa"
SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/wiki_synth/align_and_cut_wiki_synth.py"
MFA_MODEL_DIR="/mnt/taurus/home/jiaxuanluo/Documents/MFA"

DATA_JSONL="/mnt/gemini/home/jiaxuanluo/wiki_synth_tts_3variant_gigaspeech_full0_31/wiki_synth_3variant_gs_v2_clean_full0_31.jsonl"
WORK_DIR="/mnt/gemini/home/jiaxuanluo/MFA/3variant_gsv2_full0_31/work"
OUTPUT_AUDIO_DIR="/mnt/gemini/home/jiaxuanluo/MFA/3variant_gsv2_full0_31/chunks"
OUTPUT_JSONL_DIR="/mnt/gemini/home/jiaxuanluo/MFA/3variant_gsv2_full0_31/output"
NOISE_DIR=""

NUM_SHARDS=32

export PATH="${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

SHARD_ID="${SLURM_ARRAY_TASK_ID}"

export MFA_ROOT_DIR="/mnt/gemini/home/jiaxuanluo/MFA/mfa_root_gsv2_f031_${SHARD_ID}"
export TMPDIR="/mnt/gemini/home/jiaxuanluo/MFA/tmp_gsv2_f031_${SHARD_ID}"
export HOME="/mnt/gemini/home/jiaxuanluo/MFA/mfa_home_gsv2_f031_${SHARD_ID}"
mkdir -p "${MFA_ROOT_DIR}" "${TMPDIR}" "${HOME}"

echo "================================================================"
echo "[MFA][GSV2-FULL0_31] 3-variant alignment (clean-only) - taurus"
echo "[MFA][GSV2-FULL0_31] Shard ${SHARD_ID}/${NUM_SHARDS}"
echo "[MFA][GSV2-FULL0_31] Input      = ${DATA_JSONL}"
echo "[MFA][GSV2-FULL0_31] Work dir   = ${WORK_DIR}"
echo "[MFA][GSV2-FULL0_31] Chunks     = ${OUTPUT_AUDIO_DIR}"
echo "[MFA][GSV2-FULL0_31] JSONL dir  = ${OUTPUT_JSONL_DIR}"
echo "[MFA][GSV2-FULL0_31] Noise dir  = '${NOISE_DIR}' (empty = CLEAN-ONLY)"
echo "[MFA][GSV2-FULL0_31] Start: $(date)"
echo "================================================================"

if [ ! -f "${DATA_JSONL}" ]; then
    echo "[MFA][ERROR] Missing merged full JSONL: ${DATA_JSONL}" >&2
    exit 1
fi

mkdir -p "${WORK_DIR}" "${OUTPUT_AUDIO_DIR}" "${OUTPUT_JSONL_DIR}"

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

echo "[MFA][GSV2-FULL0_31] Shard ${SHARD_ID} completed at $(date)"
