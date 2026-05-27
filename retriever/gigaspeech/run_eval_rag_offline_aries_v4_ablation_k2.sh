#!/bin/bash
#SBATCH --job-name=eval_rag_k2_ablation_v4
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --array=0-7
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_eval_rag_k2_ablation_v4.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_eval_rag_k2_ablation_v4.err

set -euo pipefail

# 环境注入
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="$CONDA_PREFIX/bin:/mnt/taurus/home/jiaxuanluo/miniconda3/condabin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/.local/lib/python3.10/site-packages:/mnt/taurus/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

echo "[INFO] Running Offline RAG Evaluation V4: K2 (FAISS top-k per window) ablation..."

# ------------------------- Fixed settings -------------------------
# Fix to m2 retriever (aligned with chunk_size=1.92s in main setup)
MODEL_PATH="/mnt/gemini/data2/jiaxuanluo/final_main_result_model_v1.pt"

GLOSSARIES=(
  "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary_with_translations.json"
  "/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060.json"
)

# Fix recall top-K (final max-pool truncation)
TOP_K=10

# K2 sweep
K2S=(3 5 10 20)

# No thresholding (upper bound diagnostic)
SCORE_THRESHOLD="0.0"

# Fixed sliding window
CHUNK_SIZE="1.92"
HOP_SIZE="0.48"

MERGE_PLURAL_TERMS=1
TEXT_LORA_R=16

NUM_GLOSSARIES=${#GLOSSARIES[@]}
NUM_K2=${#K2S[@]}
TOTAL_TASKS=$((NUM_GLOSSARIES * NUM_K2))

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if [ "${TASK_ID}" -ge "${TOTAL_TASKS}" ]; then
  echo "[ERR] SLURM_ARRAY_TASK_ID (${TASK_ID}) >= TOTAL_TASKS (${TOTAL_TASKS})" >&2
  exit 1
fi

G_IDX=$((TASK_ID / NUM_K2))
K2_IDX=$((TASK_ID % NUM_K2))

GLOSSARY_PATH="${GLOSSARIES[$G_IDX]}"
K2="${K2S[$K2_IDX]}"

echo "[INFO] Task mapping: task_id=${TASK_ID} glossary_idx=${G_IDX}/${NUM_GLOSSARIES} k2_idx=${K2_IDX}/${NUM_K2}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] GLOSSARY_PATH=${GLOSSARY_PATH}"
echo "[INFO] TOP_K=${TOP_K}"
echo "[INFO] K2=${K2}"
echo "[INFO] SCORE_THRESHOLD=${SCORE_THRESHOLD}"
echo "[INFO] CHUNK_SIZE=${CHUNK_SIZE}"
echo "[INFO] HOP_SIZE=${HOP_SIZE}"
echo "[INFO] MERGE_PLURAL_TERMS=${MERGE_PLURAL_TERMS}"

# Strip accidental surrounding quotes
strip_surrounding_quotes() {
  local s="$1"
  s="${s%\"}"; s="${s#\"}"
  s="${s%\'}"; s="${s#\'}"
  printf '%s' "$s"
}
MODEL_PATH="$(strip_surrounding_quotes "${MODEL_PATH}")"
GLOSSARY_PATH="$(strip_surrounding_quotes "${GLOSSARY_PATH}")"

# Validate inputs early
if [ ! -f "${MODEL_PATH}" ]; then
  echo "[ERR] model_path not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [ ! -f "${GLOSSARY_PATH}" ]; then
  echo "[ERR] glossary_path not found: ${GLOSSARY_PATH}" >&2
  exit 1
fi

# Build / resolve index path (match run_simuleval_rag_aries_v4.sh)
MODEL_TAG="$(basename "${MODEL_PATH}" .pt)"
GLOSSARY_TAG="$(basename "${GLOSSARY_PATH}" .json)"
INDEX_CACHE_DIR="/mnt/gemini/data2/jiaxuanluo/index_cache_v4"
mkdir -p "${INDEX_CACHE_DIR}"
MODEL_TAG="${MODEL_TAG// /_}"
GLOSSARY_TAG="${GLOSSARY_TAG// /_}"
INDEX_PATH="${INDEX_CACHE_DIR}/${MODEL_TAG}__${GLOSSARY_TAG}__tr${TEXT_LORA_R}.pkl"

# Build index if missing (locked to avoid races)
if [ ! -f "${INDEX_PATH}" ]; then
  echo "[INFO] Index not found. Building: ${INDEX_PATH}"
  LOCK_FILE="${INDEX_PATH}.lock"
  (
    exec 201>"${LOCK_FILE}"
    flock 201
    if [ ! -f "${INDEX_PATH}" ]; then
      MODEL_PATH="${MODEL_PATH}" \
      GLOSSARY_PATH="${GLOSSARY_PATH}" \
      OUTPUT_PATH="${INDEX_PATH}" \
      TARGET_LANG_CODE="zh" \
      bash retriever/gigaspeech/run_build_index_v4.sh
    else
      echo "[INFO] Index already built by another process: ${INDEX_PATH}"
    fi
  )
else
  echo "[INFO] Using existing index: ${INDEX_PATH}"
fi

if [ ! -f "${INDEX_PATH}" ]; then
  echo "[ERR] Index build failed or missing: ${INDEX_PATH}" >&2
  exit 3
fi

WAV_DIR="/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/segmented_wavs/gold"
TXT_PATH="/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt"
MAX_SAMPLES=0

python /mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/eval_rag_offline_qwen3_acl6060_v4_sentence_maxpool.py \
  --model_path "${MODEL_PATH}" \
  --index_path "${INDEX_PATH}" \
  --glossary_path "${GLOSSARY_PATH}" \
  --wav_dir "${WAV_DIR}" \
  --txt_path "${TXT_PATH}" \
  --rag_chunk_size "${CHUNK_SIZE}" \
  --rag_hop_size "${HOP_SIZE}" \
  --score_threshold "${SCORE_THRESHOLD}" \
  --top_k "${TOP_K}" \
  --rag_voting_k "${K2}" \
  --max_samples "${MAX_SAMPLES}" \
  --debug_print_limit 0 \
  --debug_miss_limit 10 \
  --rag_lora_r 32 \
  --rag_text_lora_r 16 \
  --device cuda:0 \
  $([ "${MERGE_PLURAL_TERMS}" -eq 1 ] && echo "--merge_plural_terms")

echo "[INFO] K2 ablation DONE (k2=${K2})."


