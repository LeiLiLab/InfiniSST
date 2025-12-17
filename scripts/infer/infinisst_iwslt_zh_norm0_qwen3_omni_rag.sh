#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2
#SBATCH --partition=aries
#SBATCH --array=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jaxanluo@gmail.com
#SBATCH -e /mnt/taurus/home/jiaxuanluo/InfiniSST/scripts/infer/logs/infer_infinisst_omni_%A_%a.err
#SBATCH -o /mnt/taurus/home/jiaxuanluo/InfiniSST/scripts/infer/logs/infer_infinisst_omni_%A_%a.out
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/scripts/infer

set -e  # 遇到错误立即退出

map_path() { [[ "$1" = /* ]] && echo "/mnt/taurus$1" || echo "$1"; }

# 用环境里的绝对路径，避免依赖 conda activate
PY="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/infinisst/bin/python"
export PATH="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/infinisst/bin:$PATH"

ensure_python_module() {
  local module_name="$1"
  if ! "${PY}" -c "import ${module_name}" >/dev/null 2>&1; then
    echo "[INFO] Installing python module: ${module_name}"
    "${PY}" -m pip install --quiet --upgrade "${module_name}"
  fi
}

ensure_numpy_faiss() {
  if ! "${PY}" -c "import numpy as np; import faiss" >/dev/null 2>&1; then
    echo "[INFO] Reinstalling numpy<2 and faiss-cpu for RAG compatibility"
    "${PY}" -m pip install --quiet --upgrade "numpy<2" "faiss-cpu>=1.7.4"
  fi
}

# ===================== 配置参数 =====================

# 激活环境
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate infinisst
# —— Conda 环境激活（容错）——
# 尝试 3 种来源：/mnt/taurus 上的 conda.sh、本机 $HOME、系统 anaconda 模块
if [ -f /mnt/taurus/home/jiaxuanluo/miniconda3/etc/profile.d/conda.sh ]; then
  source /mnt/taurus/home/jiaxuanluo/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
  source ~/miniconda3/etc/profile.d/conda.sh
elif command -v module >/dev/null 2>&1; then
  module load anaconda || module load anaconda3 || true
  # 有些集群需要这句把 conda 注入 shell
  eval "$(conda shell.bash hook)" 2>/dev/null || true
fi

SIMULEVAL_BIN="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/infinisst/bin/simuleval"

# source /home/jiaxuanluo/miniconda3/etc/profile.d/conda.sh
# conda activate infinisst

export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:/home/jiaxuanluo/InfiniSST:${PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# Data / model resources
ROOT="/mnt/taurus/data/siqiouyang/datasets/acl6060"
LANG_CODE=zh
LANG=Chinese
TOKENIZER=zh
LATENCY_UNIT=char

DEFAULT_MODEL_NAME="/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct-lora/v4-20251114-122213-hf"
MODEL_NAME="${MODEL_NAME_OVERRIDE:-${DEFAULT_MODEL_NAME}}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://taurus.cs.ucsb.edu:8000/v1}"

RAG_INDEX_PATH="/mnt/gemini/data2/jiaxuanluo/indices/qwen2_audio_term_index_acl6060.pkl"
RAG_MODEL_PATH="/mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_level_modal_v2_best.pt"

OUTPUT_DIR="/mnt/gemini/data2/jiaxuanluo/infinisst_omni_acl6060"
mkdir -p "${OUTPUT_DIR}"

# Streaming parameters
SRC_SEGMENT_SIZE=$((SLURM_ARRAY_TASK_ID * 960))
LATENCY_MULTIPLIER=${SLURM_ARRAY_TASK_ID}
MAX_LATENCY_MULTIPLIER=12
BEAM=1
NO_REPEAT_NGRAM_LOOKBACK=100
NO_REPEAT_NGRAM_SIZE=5
MAX_NEW_TOKENS=$((SLURM_ARRAY_TASK_ID * 10))
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
MIN_START_SEC=0

RUN_TAG="omni_seg${SRC_SEGMENT_SIZE}_lm${LATENCY_MULTIPLIER}_maxnt${MAX_NEW_TOKENS}_tp${TOP_P}_tk${TOP_K}"
OUTPUT_PATH="${OUTPUT_DIR}/${RUN_TAG}"

cd "${ROOT}"

TMP_DATA_DIR="${SLURM_TMPDIR:-/tmp/${USER}/infinisst_eval_${SLURM_JOB_ID:-$$}}"
mkdir -p "${TMP_DATA_DIR}"
SOURCE_LIST="${TMP_DATA_DIR}/dev.source"
TARGET_LIST="${TMP_DATA_DIR}/dev.target.${LANG_CODE}"
cp dev.source "${SOURCE_LIST}"
cp "dev.target.${LANG_CODE}" "${TARGET_LIST}"

# 修正数据列表里的绝对路径，确保挂载前缀一致
if grep -q "/mnt/data/siqiouyang" "${SOURCE_LIST}"; then
  sed -i 's|/mnt/data/siqiouyang|/mnt/taurus/data/siqiouyang|g' "${SOURCE_LIST}"
fi

ensure_python_module openai
ensure_numpy_faiss

"${PY}" "${SIMULEVAL_BIN}" \
    --agent /mnt/taurus/home/jiaxuanluo/InfiniSST/agents/infinisst_omni.py \
    --agent-class agents.InfiniSSTOmni \
    --source-segment-size ${SRC_SEGMENT_SIZE} \
    --source-lang English \
    --target-lang ${LANG} \
    --min-start-sec ${MIN_START_SEC} \
    --source "${SOURCE_LIST}" \
    --target "${TARGET_LIST}" \
    --output ${OUTPUT_PATH} \
    \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --beam ${BEAM} \
    --no-repeat-ngram-lookback ${NO_REPEAT_NGRAM_LOOKBACK} \
    --no-repeat-ngram-size ${NO_REPEAT_NGRAM_SIZE} \
    --temperature ${TEMPERATURE} \
    --top-p ${TOP_P} \
    --top-k ${TOP_K} \
    \
    --use-vllm 1 \
    --vllm-base-url ${VLLM_BASE_URL} \
    --model-name ${MODEL_NAME} \
    \
    --quality-metrics BLEU \
    --eval-latency-unit ${LATENCY_UNIT} \
    --sacrebleu-tokenizer ${TOKENIZER} \
    \
    --rag-enabled \
    --rag-index-path ${RAG_INDEX_PATH} \
    --rag-model-path ${RAG_MODEL_PATH} \
    --rag-base-model Qwen/Qwen2-Audio-7B-Instruct \
    --rag-device cuda:0 \
    --rag-top-k 5 \
    --rag-target-lang zh \
    --rag-lora-r 16 \
    --rag-lora-alpha 32 \
    --rag-lora-dropout 0.0