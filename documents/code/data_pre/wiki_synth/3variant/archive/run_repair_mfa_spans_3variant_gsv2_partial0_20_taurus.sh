#!/bin/bash
#SBATCH --job-name=repair_gsv2_p020
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-19
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%A-%a_repair_gsv2_p020.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%A-%a_repair_gsv2_p020.err
#SBATCH --chdir=/tmp

# Rebuild partial GSV2 MFA JSONLs from existing TextGrids after adding
# mfa_term_*_in_chunk fields. This skips forced alignment and reuses the
# completed /work/shard_XX/mfa_output directories from job 43946.

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

export MFA_ROOT_DIR="/mnt/gemini/home/jiaxuanluo/MFA/mfa_root_gsv2_p020_repair_${SHARD_ID}"
export TMPDIR="/mnt/gemini/home/jiaxuanluo/MFA/tmp_gsv2_p020_repair_${SHARD_ID}"
export HOME="/mnt/gemini/home/jiaxuanluo/MFA/mfa_home_gsv2_p020_repair_${SHARD_ID}"
mkdir -p "${MFA_ROOT_DIR}" "${TMPDIR}" "${HOME}"

echo "================================================================"
echo "[REPAIR][GSV2-PARTIAL0_20] Rebuild shard ${SHARD_ID}/${NUM_SHARDS} with MFA spans"
echo "[REPAIR][GSV2-PARTIAL0_20] Input     = ${DATA_JSONL}"
echo "[REPAIR][GSV2-PARTIAL0_20] Work dir  = ${WORK_DIR}"
echo "[REPAIR][GSV2-PARTIAL0_20] JSONL dir = ${OUTPUT_JSONL_DIR}"
echo "[REPAIR][GSV2-PARTIAL0_20] Start: $(date)"
echo "================================================================"

OUTPUT_JSONL="${OUTPUT_JSONL_DIR}/wiki_synth_train_shard_$(printf '%02d' "${SHARD_ID}").jsonl"
TEXTGRID_DIR="${WORK_DIR}/shard_$(printf '%02d' "${SHARD_ID}")/mfa_output"
if [ ! -d "${TEXTGRID_DIR}" ]; then
    echo "[REPAIR][ERROR] Missing existing MFA output: ${TEXTGRID_DIR}" >&2
    exit 1
fi

python "${SCRIPT_PATH}" \
    --data "${DATA_JSONL}" \
    --work-dir "${WORK_DIR}" \
    --output-audio-dir "${OUTPUT_AUDIO_DIR}" \
    --output-jsonl "${OUTPUT_JSONL}" \
    --shard-id "${SHARD_ID}" \
    --num-shards "${NUM_SHARDS}" \
    --mfa-conda-prefix "${CONDA_ENV}" \
    --noise-dir "${NOISE_DIR}" \
    --mfa-model-dir "${MFA_MODEL_DIR}" \
    --skip-mfa

echo "[REPAIR][GSV2-PARTIAL0_20] Shard ${SHARD_ID} completed at $(date)"
