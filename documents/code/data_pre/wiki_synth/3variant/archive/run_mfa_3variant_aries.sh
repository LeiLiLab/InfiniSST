#!/bin/bash
#SBATCH --job-name=mfa_3var
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-19
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%A-%a_mfa_3var.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%A-%a_mfa_3var.err
#SBATCH --chdir=/tmp

set -euo pipefail

# ======Configuration=====
CONDA_ENV="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/mfa"
SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/wiki_synth/align_and_cut_wiki_synth.py"
MFA_MODEL_DIR="/mnt/taurus/home/jiaxuanluo/Documents/MFA"

DATA_JSONL="/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/3variant/wiki_synth_3variant_dual.jsonl"
WORK_DIR="/mnt/aries/data4/jiaxuanluo/MFA/3variant/work"
OUTPUT_AUDIO_DIR="/mnt/aries/data4/jiaxuanluo/MFA/3variant/chunks"
OUTPUT_JSONL_DIR="/mnt/aries/data4/jiaxuanluo/MFA/3variant/output"
NOISE_DIR="/mnt/taurus/data/siqiouyang/datasets/wham_wav"

NUM_SHARDS=20
# ======Configuration=====

export PATH="${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

SHARD_ID="${SLURM_ARRAY_TASK_ID}"

# aries root partition (/) is 100% full; redirect all temp/cache to data4
# Each shard gets its own MFA_ROOT_DIR to avoid concurrent YAML corruption
export MFA_ROOT_DIR="/mnt/data4/jiaxuanluo/MFA/mfa_root_${SHARD_ID}"
export TMPDIR="/mnt/data4/jiaxuanluo/MFA/tmp_${SHARD_ID}"
export HOME="/mnt/data4/jiaxuanluo/mfa_home_${SHARD_ID}"
mkdir -p "${MFA_ROOT_DIR}" "${TMPDIR}" "${HOME}"

echo "================================================================"
echo "[MFA] 3-variant alignment - aries"
echo "[MFA] Shard ${SHARD_ID}/${NUM_SHARDS}"
echo "[MFA] Input: ${DATA_JSONL}"
echo "[MFA] Work dir: ${WORK_DIR}"
echo "[MFA] Output chunks: ${OUTPUT_AUDIO_DIR}"
echo "[MFA] Output JSONL dir: ${OUTPUT_JSONL_DIR}"
echo "[MFA] Noise dir: ${NOISE_DIR}"
echo "[MFA] MFA models: ${MFA_MODEL_DIR}"
echo "[MFA] Start: $(date)"
echo "================================================================"

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

echo "[MFA] Shard ${SHARD_ID} completed at $(date)"
