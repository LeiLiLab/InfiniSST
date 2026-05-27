#!/bin/bash
#SBATCH --job-name=simuleval_rag_sweep
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --array=0
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_simuleval_rag_sweep_baseline.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_simuleval_rag_sweep_baseline.err

set -euo pipefail

# ==================== 环境配置 ====================
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="$CONDA_PREFIX/bin:/mnt/taurus/home/jiaxuanluo/miniconda3/condabin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

# vLLM 配置
export VLLM_USE_V1=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=0

# ==================== 扫参配置 ====================
MODELS=(
  "/mnt/gemini/data2/jiaxuanluo/models/owaski/gigaspeech-zh"
  # "/mnt/gemini/data2/jiaxuanluo/models/owaski/ja"
  # "/mnt/gemini/data2/jiaxuanluo/models/owaski/de"
)
LANG_CODES=("zh")
TARGET_LANGS=("Chinese")
SEGMENT_SECS=(3.84)

NUM_MODELS=${#MODELS[@]}
NUM_SEGS=${#SEGMENT_SECS[@]}

# 解码 Task ID (12 tasks total)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SEG_IDX=$((TASK_ID % NUM_SEGS))
MODEL_IDX=$((TASK_ID / NUM_SEGS))

CUR_MODEL="${MODELS[$MODEL_IDX]}"
CUR_LANG="${LANG_CODES[$MODEL_IDX]}"
CUR_TARGET_LANG="${TARGET_LANGS[$MODEL_IDX]}"
CUR_SEG="${SEGMENT_SECS[$SEG_IDX]}"

# 根据语言设置分词器和延迟单位
if [ "${CUR_LANG}" == "zh" ]; then
  CUR_TOKENIZER="zh"
  CUR_LATENCY_UNIT="char"
elif [ "${CUR_LANG}" == "ja" ]; then
  CUR_TOKENIZER="ja-mecab"
  CUR_LATENCY_UNIT="char"
elif [ "${CUR_LANG}" == "de" ]; then
  CUR_TOKENIZER="13a"
  CUR_LATENCY_UNIT="word"
fi

# RAG 参数固定
CUR_INDEX="/mnt/gemini/data2/jiaxuanluo/index_cache_v4/final_main_result_model_v1__extracted_glossary_with_translations__tr16.pkl"
CUR_TOPK=5
CUR_THRESH=0.4

# 获取索引简称用于路径
INDEX_NAME="curated"

echo "[INFO] Task ID: ${TASK_ID}"
echo "[INFO] MODEL: ${CUR_MODEL}"
echo "[INFO] LANG: ${CUR_LANG}"
echo "[INFO] SEGMENT_SEC: ${CUR_SEG}"
echo "[INFO] INDEX: ${INDEX_NAME} (${CUR_INDEX})"
echo "[INFO] TOP_K: ${CUR_TOPK}"
echo "[INFO] THRESHOLD: ${CUR_THRESH}"
echo "[INFO] TOKENIZER: ${CUR_TOKENIZER}"
echo "[INFO] LATENCY_UNIT: ${CUR_LATENCY_UNIT}"

# ==================== 路径与参数 ====================
ROOT="/mnt/taurus/data/siqiouyang/datasets/acl6060"
LANG_CODE="${CUR_LANG}"
SOURCE_LANG="English"
TARGET_LANG="${CUR_TARGET_LANG}"

MODEL_NAME="${CUR_MODEL}"
RAG_MODEL_PATH="/mnt/gemini/data2/jiaxuanluo/final_main_result_model_v1.pt"

# 输出目录 (包含超参信息)
VERSION=""
OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_sweep_v4_baseline"

MODEL_SHORT=$(basename "${CUR_MODEL}")
OUTPUT_PATH="${OUTPUT_BASE}/${MODEL_SHORT}_seg${CUR_SEG}_${VERSION}"
mkdir -p "${OUTPUT_PATH}"

# ==================== 准备临时数据 ====================
TMP_DATA_DIR="/tmp/${USER}/infinisst_eval_${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}"
mkdir -p "${TMP_DATA_DIR}"
trap 'rm -rf "${TMP_DATA_DIR}"' EXIT

SOURCE_LIST="${TMP_DATA_DIR}/dev.source"
TARGET_LIST="${TMP_DATA_DIR}/dev.target.${LANG_CODE}"

cp "${ROOT}/dev.source" "${SOURCE_LIST}"
if [ -f "${ROOT}/dev.target.${LANG_CODE}" ]; then
  cp "${ROOT}/dev.target.${LANG_CODE}" "${TARGET_LIST}"
else
  echo "[INFO] ${ROOT}/dev.target.${LANG_CODE} not found, extracting from full text..."
  head -n 5 "${ROOT}/dev/text/txt/ACL.6060.dev.en-xx.${LANG_CODE}.txt" > "${TARGET_LIST}"
fi

# 修正音频路径
if grep -q "/mnt/data/siqiouyang" "${SOURCE_LIST}"; then
  sed -i 's|/mnt/data/siqiouyang|/mnt/taurus/data/siqiouyang|g' "${SOURCE_LIST}"
fi

RAG_GPU="cuda:1"

# ==================== 运行 SimulEval ====================
LATENCY_MULTIPLIER=1
SRC_SEGMENT_SIZE=$((1 * 960))
MAX_NEW_TOKENS=$((LATENCY_MULTIPLIER * 40))

python -u "$(which simuleval)" \
  --agent agents/infinisst_omni_vllm_rag_v4.py \
  --agent-class agents.infinisst_omni_vllm_rag_v4.InfiniSSTOmniVLLMRAGV4 \
  \
  --source "${SOURCE_LIST}" \
  --target "${TARGET_LIST}" \
  --output "${OUTPUT_PATH}" \
  \
  --source-segment-size "${SRC_SEGMENT_SIZE}" \
  --source-lang "${SOURCE_LANG}" \
  --target-lang "${TARGET_LANG}" \
  --min-start-sec 0 \
  \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --beam 1 \
  --no-repeat-ngram-lookback 100 \
  --no-repeat-ngram-size 5 \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  \
  --use-vllm 1 \
  --gpu-memory-utilization 0.8 \
  --model-name "${MODEL_NAME}" \
  --max-cache-chunks $(python3 -c "import math; print(math.ceil(80 / ${CUR_SEG}))") \
  --keep-cache-chunks $(python3 -c "import math; print(math.ceil(60 / ${CUR_SEG}))") \
  \
  --quality-metrics BLEU \
  --eval-latency-unit "${CUR_LATENCY_UNIT}" \
  --sacrebleu-tokenizer "${CUR_TOKENIZER}" \
  --vllm-segment-sec "${CUR_SEG}" \
  --rag-min-terms 0 \
  --log-sample 3

echo "[INFO] SimulEval Task ${TASK_ID} DONE"

# ==================== 后处理评估 (stream_laal_term.py) ====================
SUMMARY_LOG="${OUTPUT_BASE}/all_results_summary.log"

GLOSSARIES=(
  "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary_with_translations.json"
  "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060.json"
)

REF_FILE="/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/text/txt/ACL.6060.dev.en-xx.${CUR_LANG}.txt"
AUDIO_YAML="/mnt/taurus/data/siqiouyang/datasets/acl6060/dev.yaml"

# 使用 flock 确保多进程写入同一个日志文件时不冲突
exec 200>>"${SUMMARY_LOG}"
flock 200

echo "--------------------------------------------------------" >> "${SUMMARY_LOG}"
echo "TIMESTAMP: $(date +'%Y-%m-%d %H:%M:%S')" >> "${SUMMARY_LOG}"
echo "TASK_ID: ${TASK_ID} | MODEL: ${MODEL_SHORT} | SEG: ${CUR_SEG} | LANG: ${CUR_LANG}" >> "${SUMMARY_LOG}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}" >> "${SUMMARY_LOG}"



for GLOS in "${GLOSSARIES[@]}"; do
  GLOS_NAME=$(basename "${GLOS}")
  echo ">>> Glossary: ${GLOS_NAME}" >> "${SUMMARY_LOG}"
  
  # 运行评估脚本并提取关键分数
  python /home/jiaxuanluo/FBK-fairseq/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py \
    --simuleval-instances "${OUTPUT_PATH}/instances.log" \
    --reference "${REF_FILE}" \
    --audio-yaml "${AUDIO_YAML}" \
    --sacrebleu-tokenizer "${CUR_TOKENIZER}" \
    --latency-unit "${CUR_LATENCY_UNIT}" \
    --glossary "${GLOS}" \
    --term-lang "${CUR_LANG}" \
    --term-mismatch-examples 0 >> "${SUMMARY_LOG}" 2>&1
done

echo "--------------------------------------------------------" >> "${SUMMARY_LOG}"
echo "" >> "${SUMMARY_LOG}"

flock -u 200
exec 200>&-

echo "[INFO] All evaluations for Task ${TASK_ID} DONE"
