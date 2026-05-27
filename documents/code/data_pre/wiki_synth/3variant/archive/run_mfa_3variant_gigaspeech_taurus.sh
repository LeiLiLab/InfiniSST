#!/bin/bash
#SBATCH --job-name=mfa_3var_gsv2
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-19
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%A-%a_mfa_3var_gsv2.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%A-%a_mfa_3var_gsv2.err
#SBATCH --chdir=/tmp

# MFA alignment + 1.92s-chunk cutting for the v2 GigaSpeech voice pool
# regenerated TTS.  Clean-only variant (--noise-dir "") -- no pre-rendered
# noisy chunks produced.  Shards=20 to match the aries baseline's job
# layout; output paths under gemini/home (taurus/data and aries/* near
# full).
#
# Diffs vs run_mfa_3variant_aries.sh:
#   1. Partition taurus (aries busy).
#   2. DATA_JSONL: new v2 merged jsonl (produced by merge_tts_3variant.py
#      after TTS completes).
#   3. WORK_DIR / OUTPUT_AUDIO_DIR / OUTPUT_JSONL_DIR all under gemini/home.
#   4. NOISE_DIR="" (no WHAM mixing; pre-rendered noisy chunks are not
#      produced for the A1 experiment).
#   5. MFA root dir per-shard under gemini/home to avoid YAML races.

set -euo pipefail

CONDA_ENV="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/mfa"
SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/wiki_synth/align_and_cut_wiki_synth.py"
MFA_MODEL_DIR="/mnt/taurus/home/jiaxuanluo/Documents/MFA"

DATA_JSONL="/mnt/gemini/home/jiaxuanluo/wiki_synth_tts_3variant_gigaspeech_full/wiki_synth_3variant_gs_v2_clean_dual.jsonl"
WORK_DIR="/mnt/gemini/home/jiaxuanluo/MFA/3variant_gsv2/work"
OUTPUT_AUDIO_DIR="/mnt/gemini/home/jiaxuanluo/MFA/3variant_gsv2/chunks"
OUTPUT_JSONL_DIR="/mnt/gemini/home/jiaxuanluo/MFA/3variant_gsv2/output"
NOISE_DIR=""

NUM_SHARDS=20

export PATH="${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

SHARD_ID="${SLURM_ARRAY_TASK_ID}"

# Per-shard MFA root / TMP / HOME directory: avoids concurrent YAML corruption.
export MFA_ROOT_DIR="/mnt/gemini/home/jiaxuanluo/MFA/mfa_root_gsv2_${SHARD_ID}"
export TMPDIR="/mnt/gemini/home/jiaxuanluo/MFA/tmp_gsv2_${SHARD_ID}"
export HOME="/mnt/gemini/home/jiaxuanluo/MFA/mfa_home_gsv2_${SHARD_ID}"
mkdir -p "${MFA_ROOT_DIR}" "${TMPDIR}" "${HOME}"

echo "================================================================"
echo "[MFA][GSV2] 3-variant alignment (clean-only) - taurus"
echo "[MFA][GSV2] Shard ${SHARD_ID}/${NUM_SHARDS}"
echo "[MFA][GSV2] Input      = ${DATA_JSONL}"
echo "[MFA][GSV2] Work dir   = ${WORK_DIR}"
echo "[MFA][GSV2] Chunks     = ${OUTPUT_AUDIO_DIR}"
echo "[MFA][GSV2] JSONL dir  = ${OUTPUT_JSONL_DIR}"
echo "[MFA][GSV2] Noise dir  = '${NOISE_DIR}' (empty = CLEAN-ONLY)"
echo "[MFA][GSV2] Start: $(date)"
echo "================================================================"

mkdir -p "${WORK_DIR}" "${OUTPUT_AUDIO_DIR}" "${OUTPUT_JSONL_DIR}"

OUTPUT_JSONL="${OUTPUT_JSONL_DIR}/wiki_synth_train_gsv2_shard_$(printf '%02d' "${SHARD_ID}").jsonl"

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

echo "[MFA][GSV2] Shard ${SHARD_ID} completed at $(date)"
