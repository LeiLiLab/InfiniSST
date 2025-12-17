#!/usr/bin/env bash
# 测试脚本：只使用 vLLM，不启用 RAG
# 用于排查 vLLM 初始化问题

#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2
#SBATCH --partition=aries
#SBATCH --array=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jaxanluo@gmail.com
#SBATCH -e /mnt/taurus/home/jiaxuanluo/InfiniSST/scripts/infer/logs/test_vllm_only_%A_%a.err
#SBATCH -o /mnt/taurus/home/jiaxuanluo/InfiniSST/scripts/infer/logs/test_vllm_only_%A_%a.out
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/scripts/infer

set -e

PY="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/infinisst/bin/python"
export PATH="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/infinisst/bin:$PATH"

if [ -f /mnt/taurus/home/jiaxuanluo/miniconda3/etc/profile.d/conda.sh ]; then
  source /mnt/taurus/home/jiaxuanluo/miniconda3/etc/profile.d/conda.sh
fi

SIMULEVAL_BIN="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/infinisst/bin/simuleval"

export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:/home/jiaxuanluo/InfiniSST:${PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=0
export VLLM_NO_USAGE_STATS=1
export VLLM_USE_V1=0  # Force use v0 engine instead of v1 (more stable)
export CUDA_VISIBLE_DEVICES=0,1
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

echo "[TEST] Testing vLLM without RAG"
echo "[TEST] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

ROOT="/mnt/taurus/data/siqiouyang/datasets/acl6060"
LANG_CODE=zh
LANG=Chinese
TOKENIZER=zh
LATENCY_UNIT=char

MODEL_NAME="/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct-lora/v4-20251114-122213-hf"
OUTPUT_DIR="/mnt/gemini/data2/jiaxuanluo/test_vllm_only"
mkdir -p "${OUTPUT_DIR}"

SRC_SEGMENT_SIZE=960
MAX_NEW_TOKENS=10
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
MIN_START_SEC=0
MAX_CACHE_CHUNKS=120
KEEP_CACHE_CHUNKS=60

OUTPUT_PATH="${OUTPUT_DIR}/test_run"

cd "${ROOT}"

TMP_DATA_DIR="/tmp/${USER}/test_vllm_$$"
mkdir -p "${TMP_DATA_DIR}"
SOURCE_LIST="${TMP_DATA_DIR}/dev.source"
TARGET_LIST="${TMP_DATA_DIR}/dev.target.${LANG_CODE}"
cp dev.source "${SOURCE_LIST}"
cp "dev.target.${LANG_CODE}" "${TARGET_LIST}"

if grep -q "/mnt/data/siqiouyang" "${SOURCE_LIST}"; then
  sed -i 's|/mnt/data/siqiouyang|/mnt/taurus/data/siqiouyang|g' "${SOURCE_LIST}"
fi

echo "[TEST] Starting simuleval without RAG..."

"${PY}" "${SIMULEVAL_BIN}" \
    --agent /mnt/taurus/home/jiaxuanluo/InfiniSST/agents/infinisst_omni_vllm_rag.py \
    --agent-class agents.InfiniSSTOmniVLLMRAG \
    --source-segment-size ${SRC_SEGMENT_SIZE} \
    --source-lang English \
    --target-lang ${LANG} \
    --min-start-sec ${MIN_START_SEC} \
    --source "${SOURCE_LIST}" \
    --target "${TARGET_LIST}" \
    --output ${OUTPUT_PATH} \
    \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --beam 1 \
    --temperature ${TEMPERATURE} \
    --top-p ${TOP_P} \
    --top-k ${TOP_K} \
    \
    --use-vllm 1 \
    --model-name ${MODEL_NAME} \
    --max-cache-chunks ${MAX_CACHE_CHUNKS} \
    --keep-cache-chunks ${KEEP_CACHE_CHUNKS} \
    \
    --quality-metrics BLEU \
    --eval-latency-unit ${LATENCY_UNIT} \
    --sacrebleu-tokenizer ${TOKENIZER}
    # NOTE: No --rag-enabled flag, testing vLLM only

echo "[TEST] Test completed"

