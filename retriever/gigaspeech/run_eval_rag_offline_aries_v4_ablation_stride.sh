#!/bin/bash
#SBATCH --job-name=eval_rag_sweep_v4
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --array=0-15
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_eval_rag_sweep_v4.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_eval_rag_sweep_v4.err

set -euo pipefail

# 环境注入
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="$CONDA_PREFIX/bin:/mnt/taurus/home/jiaxuanluo/miniconda3/condabin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/.local/lib/python3.10/site-packages:/mnt/taurus/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

echo "[INFO] Running Offline RAG Evaluation V4 (Tuned Text Encoder)..."

# Stride ablation: fix chunk_size=1.92s and sweep hop_size.
# We intentionally remove threshold filtering by fixing score_threshold=0.0 for an easy-to-interpret upper bound diagnostic.
MODELS=(
  # Fix to m2 retriever (aligned with chunk_size=1.92s)
  "/mnt/gemini/data2/jiaxuanluo/final_main_result_model_v1.pt"
)
GLOSSARIES=(
  "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary_with_translations.json"
  "/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060.json"
)
TOP_KS=(5 10)
# Hop sizes to sweep (seconds)
HOP_SIZES=(0.96 0.48 0.24 0.12)
MERGE_PLURAL_TERMS=1  # set to 1 to merge plural variants like "text/texts" during eval

NUM_MODELS=${#MODELS[@]}
NUM_GLOSSARIES=${#GLOSSARIES[@]}
NUM_TOPKS=${#TOP_KS[@]}
NUM_HOPS=${#HOP_SIZES[@]}
TOTAL_TASKS=$((NUM_MODELS * NUM_GLOSSARIES * NUM_HOPS * NUM_TOPKS))

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if [ "${TASK_ID}" -ge "${TOTAL_TASKS}" ]; then
  echo "[ERR] SLURM_ARRAY_TASK_ID (${TASK_ID}) >= TOTAL_TASKS (${TOTAL_TASKS})" >&2
  exit 1
fi

K_IDX=$((TASK_ID % NUM_TOPKS))
TMP=$((TASK_ID / NUM_TOPKS))
H_IDX=$((TMP % NUM_HOPS))
TMP=$((TMP / NUM_HOPS))
G_IDX=$((TMP % NUM_GLOSSARIES))
M_IDX=$((TMP / NUM_GLOSSARIES))

MODEL_PATH="${MODELS[$M_IDX]}"
GLOSSARY_PATH="${GLOSSARIES[$G_IDX]}"
TOP_K="${TOP_KS[$K_IDX]}"
SCORE_THRESHOLD="0.0"
HOP_SIZE="${HOP_SIZES[$H_IDX]}"
CHUNK_SIZE="1.92"

echo "[INFO] Task mapping: task_id=${TASK_ID} model_idx=${M_IDX}/${NUM_MODELS} glossary_idx=${G_IDX}/${NUM_GLOSSARIES} hop_idx=${H_IDX}/${NUM_HOPS} topk_idx=${K_IDX}/${NUM_TOPKS}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] GLOSSARY_PATH=${GLOSSARY_PATH}"
echo "[INFO] SCORE_THRESHOLD=${SCORE_THRESHOLD}"
echo "[INFO] TOP_K=${TOP_K}"
echo "[INFO] CHUNK_SIZE=${CHUNK_SIZE}"
echo "[INFO] HOP_SIZE=${HOP_SIZE}"
echo "[INFO] MERGE_PLURAL_TERMS=${MERGE_PLURAL_TERMS}"

TEXT_LORA_R=16

# Strip accidental surrounding quotes to avoid FileNotFoundError (e.g. MODEL_PATH="'/path/file.pt'")
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

# Auto-generate / cache index path based on MODEL_PATH + GLOSSARY_PATH (match run_simuleval_rag_aries_v4.sh)
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

# Sentence-level sliding window max-pool diagnostic (easy-to-interpret offline upper bound)
K2=20
MAX_SAMPLES=0 # 0 表示评估所有样本

# 运行评估脚本：整句滑窗 + max-pool，看 GT 命中次数
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

echo "[INFO] Evaluation finished. Threshold: ${SCORE_THRESHOLD}"

