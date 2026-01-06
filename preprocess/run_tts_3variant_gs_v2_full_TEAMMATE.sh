#!/usr/bin/env bash

##SBATCH --nodelist=babel-4-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:L40S:1
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
##SBATCH --dependency=afterok:job_id
#SBATCH --job-name=tts_giga_spk
##SBATCH --array=22-31
#SBATCH --array=23-29
##SBATCH --account=siqiouya
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH -e slurm_logs/%A-%a.err
#SBATCH -o slurm_logs/%A-%a.out

# ---------------------------------------------------------------------------
# Teammate-side sbatch for 43871 back-half (shards 22-31).
#
# Assumes teammate already has a working CosyVoice + vLLM env. You must have
# `python` point at the right env BEFORE calling sbatch (e.g. submit from
# inside the activated env, or add `source activate <env>` below).
#
# Derived from run_tts_3variant_gigaspeech_full_taurus.sh (jiaxuan's 43871).
# Key differences:
#   - No conda/PATH manipulation (use your own env).
#   - All data paths come from TEAMMATE_* env vars.
#   - --array defaults to 22-31 (10 shards). shard_id MUST equal the array
#     idx — sharding is `range(shard_id, total, 32)`, don't remap to 0-9.
#   - --no-load-trt passed because CosyVoice ships TRT plans that are
#     GPU-arch-specific. If you want TRT, remove that flag and let it
#     recompile (~15-25 min first run).
# ---------------------------------------------------------------------------
#
# REQUIRED env vars:
#   COSYVOICE_ROOT         path to your CosyVoice checkout (worker does
#                          `sys.path.append(COSYVOICE_ROOT)` internally)
#   TEAMMATE_MODEL_DIR     path to Fun-CosyVoice3-0.5B/ (typically
#                          ${COSYVOICE_ROOT}/pretrained_models/Fun-CosyVoice3-0.5B)
#   TEAMMATE_DATA          path to wiki_synth_utterances_3variant.jsonl
#                          (from jiaxuan; 343 MB; shard*.jsonl WILL LAND in
#                          `dirname(TEAMMATE_DATA)` — see handoff §6)
#   TEAMMATE_SPEAKER_DIR   path to gigaspeech_speaker_prompts/ (from jiaxuan's
#                          HF dataset; 2.3 GB, contains speaker_index.json +
#                          9989 wav)
#   TEAMMATE_OUTPUT_DIR    path to write ~150-200 GB of clean/ wavs
#                          (MUST have >=250 GB free; SSD preferred)
#
# OPTIONAL env var:
#   TEAMMATE_WORKER        path to rag_tts_multispeaker_noise.py.
#                          Defaults to `rag_tts_multispeaker_noise.py`
#                          sitting *next to this sbatch* — i.e. you can just
#                          drop both files into the same directory and run.
#
# Submit (from wherever you put these two files):
#   sbatch -p <your_partition> -o <log>/%A-%a.out -e <log>/%A-%a.err \
#          run_tts_3variant_gs_v2_full_TEAMMATE.sh
#
# Smoke-test a single shard first (recommended):
#   sbatch --array=31 ... run_tts_3variant_gs_v2_full_TEAMMATE.sh
# ---------------------------------------------------------------------------

set -euo pipefail

source /home/siqiouya/miniconda3/bin/activate cosyvoice_vllm 

COSYVOICE_ROOT=/home/siqiouya/code/CosyVoice/
TEAMMATE_MODEL_DIR=/data/user_data/siqiouya/ckpts/pretrained/tts/Fun-CosyVoice3-0.5B/
TEAMMATE_DATA=/home/siqiouya/code/infinisst-omni/preprocess/wiki_synth_utterances_3variant.jsonl
TEAMMATE_SPEAKER_DIR=/data/group_data/li_lab/siqiouya/datasets/gigaspeech/gigaspeech_speaker_prompts_v2/gigaspeech_speaker_prompts/
TEAMMATE_OUTPUT_DIR=/data/group_data/li_lab/siqiouya/datasets/gigaspeech/wiki_synth_utterances_tts_gigaspk_22-31
TEAMMATE_WORKER=/home/siqiouya/code/infinisst-omni/preprocess/rag_tts_multispeaker_noise.py

# : "${COSYVOICE_ROOT:?set COSYVOICE_ROOT to your CosyVoice checkout}"
# : "${TEAMMATE_MODEL_DIR:?set TEAMMATE_MODEL_DIR to Fun-CosyVoice3-0.5B/}"
# : "${TEAMMATE_DATA:?set TEAMMATE_DATA to wiki_synth_utterances_3variant.jsonl}"
# : "${TEAMMATE_SPEAKER_DIR:?set TEAMMATE_SPEAKER_DIR to gigaspeech_speaker_prompts/}"
# : "${TEAMMATE_OUTPUT_DIR:?set TEAMMATE_OUTPUT_DIR to a path with >=250 GB free}"

# Locate the worker script: explicit env var wins; otherwise look next to
# this sbatch file (BASH_SOURCE survives under `sbatch` because SLURM copies
# the script to the job's working dir, but we handle both cases).
if [ -n "${TEAMMATE_WORKER:-}" ]; then
    SCRIPT_PATH="${TEAMMATE_WORKER}"
else
    _SELF="${BASH_SOURCE[0]:-$0}"
    _SELF_DIR="$(cd "$(dirname "${_SELF}")" && pwd)"
    SCRIPT_PATH="${_SELF_DIR}/rag_tts_multispeaker_noise.py"
fi
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "[FATAL] Worker script not found: ${SCRIPT_PATH}"
    echo "[FATAL] Either place rag_tts_multispeaker_noise.py next to this sbatch,"
    echo "[FATAL] or export TEAMMATE_WORKER=/abs/path/to/rag_tts_multispeaker_noise.py"
    exit 3
fi

TOTAL_SHARDS=32
BATCH_SIZE=16
SNR_LOW=5
SNR_HIGH=25
OUTPUT_JSONL_PREFIX="wiki_synth_3variant_gs_v2_clean_with_tts"

export COSYVOICE_ROOT
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

LOCAL_TMP_DIR="/tmp/${USER}_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}/pytorch_tmp"
mkdir -p "${LOCAL_TMP_DIR}"
export TMPDIR="${LOCAL_TMP_DIR}"
export TMP="${LOCAL_TMP_DIR}"
export TEMP="${LOCAL_TMP_DIR}"
trap '[ -n "${LOCAL_TMP_DIR:-}" ] && rm -rf "${LOCAL_TMP_DIR}"' EXIT

SHARD_ID="${SLURM_ARRAY_TASK_ID}"

if [ "${SHARD_ID}" -lt 22 ] || [ "${SHARD_ID}" -gt 31 ]; then
    echo "[FATAL] Shard ${SHARD_ID} outside 22-31. jiaxuan owns 14-21."
    exit 2
fi

echo "================================================================"
echo "[TTS][GS-V2-TEAMMATE] Shard ${SHARD_ID}/${TOTAL_SHARDS}"
echo "[TTS][GS-V2-TEAMMATE] WORKER   = ${SCRIPT_PATH}"
echo "[TTS][GS-V2-TEAMMATE] DATA     = ${TEAMMATE_DATA}"
echo "[TTS][GS-V2-TEAMMATE] SPEAKER  = ${TEAMMATE_SPEAKER_DIR}"
echo "[TTS][GS-V2-TEAMMATE] MODEL    = ${TEAMMATE_MODEL_DIR}"
echo "[TTS][GS-V2-TEAMMATE] OUTPUT   = ${TEAMMATE_OUTPUT_DIR}"
echo "[TTS][GS-V2-TEAMMATE] JSONL to = $(dirname "${TEAMMATE_DATA}") (quirk — see handoff §6)"
echo "[TTS][GS-V2-TEAMMATE] start at $(date)"
echo "================================================================"

mkdir -p "${TEAMMATE_OUTPUT_DIR}"

python "${SCRIPT_PATH}" \
    --data "${TEAMMATE_DATA}" \
    --output-dir "${TEAMMATE_OUTPUT_DIR}" \
    --model-dir "${TEAMMATE_MODEL_DIR}" \
    --speaker-dir "${TEAMMATE_SPEAKER_DIR}" \
    --noise-dir "" \
    --shard-id "${SHARD_ID}" \
    --num-shards "${TOTAL_SHARDS}" \
    --batch-size "${BATCH_SIZE}" \
    --snr-low "${SNR_LOW}" \
    --snr-high "${SNR_HIGH}" \
    --no-load-trt \
    --no_dedup \
    --output_jsonl_prefix "${OUTPUT_JSONL_PREFIX}"

echo "[TTS][GS-V2-TEAMMATE] shard ${SHARD_ID} done at $(date)"
