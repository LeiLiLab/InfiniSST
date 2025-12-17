#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64GB
#SBATCH --gres=gpu:3
#SBATCH --partition=aries
#SBATCH --array=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jaxanluo@gmail.com
#SBATCH -e /mnt/taurus/home/jiaxuanluo/InfiniSST/scripts/infer/logs/infer_infinisst_omni_vllm_rag_docker_%A_%a.err
#SBATCH -o /mnt/taurus/home/jiaxuanluo/InfiniSST/scripts/infer/logs/infer_infinisst_omni_vllm_rag_docker_%A_%a.out
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/scripts/infer

set -e  # 遇到错误立即退出

echo "========================================"
echo "Starting InfiniSST Omni VLLM RAG in Docker"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "========================================"

# ===================== 配置参数 =====================

# Data / model resources
ROOT="/mnt/taurus/data/siqiouyang/datasets/acl6060"
LANG_CODE=zh
LANG=Chinese
TOKENIZER=zh
LATENCY_UNIT=char

DEFAULT_MODEL_NAME="/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct-lora/v11-20251121-170027-hf"
MODEL_NAME="${MODEL_NAME_OVERRIDE:-${DEFAULT_MODEL_NAME}}"

RAG_INDEX_PATH="/mnt/gemini/data2/jiaxuanluo/indices/qwen2_audio_term_index_acl6060.pkl"
RAG_MODEL_PATH="/mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_level_modal_v2_best.pt"

OUTPUT_DIR="/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_docker_acl6060"
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
MAX_CACHE_CHUNKS=120
KEEP_CACHE_CHUNKS=60

RUN_TAG="omni_docker_seg${SRC_SEGMENT_SIZE}_lm${LATENCY_MULTIPLIER}_maxnt${MAX_NEW_TOKENS}_tp${TOP_P}_tk${TOP_K}"
OUTPUT_PATH="${OUTPUT_DIR}/${RUN_TAG}"

# Prepare data
TMP_DATA_DIR="/tmp/${USER}/infinisst_eval_${SLURM_JOB_ID:-$$}"
mkdir -p "${TMP_DATA_DIR}"
SOURCE_LIST="${TMP_DATA_DIR}/dev.source"
TARGET_LIST="${TMP_DATA_DIR}/dev.target.${LANG_CODE}"

cd "${ROOT}"
cp dev.source "${SOURCE_LIST}"
cp "dev.target.${LANG_CODE}" "${TARGET_LIST}"

# 修正数据列表里的绝对路径
if grep -q "/mnt/data/siqiouyang" "${SOURCE_LIST}"; then
  sed -i 's|/mnt/data/siqiouyang|/mnt/taurus/data/siqiouyang|g' "${SOURCE_LIST}"
fi

# ===================== Docker 配置 =====================

DOCKER_IMAGE="qwenllm/qwen3-omni:3-cu124"

echo "========================================"
echo "Docker Configuration:"
echo "  Image: ${DOCKER_IMAGE}"
echo "  GPUs: all (3 GPUs requested)"
echo "  Shared Memory: 16GB"
echo "========================================"

# 检查 Docker 镜像是否存在
echo "Checking if Docker image exists..."
if docker images "${DOCKER_IMAGE}" | grep -q "qwen3-omni"; then
  echo "✅ Docker image found locally"
else
  echo "⚠️ Docker image not found, will pull from Docker Hub (this may take 15-30 minutes)..."
  echo "Image size: ~25GB"
  echo "Pulling image..."
fi

echo ""
echo "========================================"
echo "Starting Docker container..."
echo "Time: $(date)"
echo "========================================"

# ===================== 在 Docker 中运行 =====================

# 注意：移除 -it 标志，因为 SLURM 是非交互式的
docker run --rm \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  -e VLLM_USE_V1=0 \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_IB_DISABLE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=0 \
  -e VLLM_NO_USAGE_STATS=1 \
  -e CUDA_VISIBLE_DEVICES=0,1,2 \
  -e OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}" \
  -v /mnt/gemini/data2/jiaxuanluo:/mnt/gemini/data2/jiaxuanluo \
  -v /mnt/gemini/data1/jiaxuanluo:/mnt/gemini/data1/jiaxuanluo \
  -v /mnt/taurus/data/siqiouyang:/mnt/taurus/data/siqiouyang \
  -v /mnt/taurus/home/jiaxuanluo/InfiniSST:/workspace/InfiniSST \
  -v "${TMP_DATA_DIR}:${TMP_DATA_DIR}" \
  -w /workspace/InfiniSST \
  "${DOCKER_IMAGE}" \
  bash -c "
    set -e
    set -x  # 打印执行的命令（调试用）
    
    echo '========================================'
    echo 'Installing dependencies...'
    echo '========================================'
    
    # 安装必要的 Python 包
    # 注意：先安装 numpy<2 和 faiss，然后允许 opencv 使用不同版本的 numpy
    pip install --quiet --no-cache-dir 'numpy<2' 'faiss-cpu>=1.7.4'
    pip install --quiet --no-cache-dir \
      evaluate jiwer lightning deepspeed torchtune \
      sentence-transformers tensorboardX matplotlib \
      soundfile simuleval jieba unbabel-comet simalign \
      praat-textgrids peft openai
    
    echo '✅ Dependencies installed'
    echo ''
    echo '========================================'
    echo 'Checking simuleval installation...'
    echo '========================================'
    which simuleval
    simuleval --version || echo 'simuleval version check failed, continuing...'
    
    echo ''
    echo '========================================'
    echo 'Running SimulEval...'
    echo '========================================'
    
    # 设置 PYTHONPATH
    export PYTHONPATH=/workspace/InfiniSST:\${PYTHONPATH}
    
    # 运行 SimulEval (使用 python -u 无缓冲模式，确保日志实时输出)
    # 注意：--agent-class 的格式是 module.ClassName
    # 由于 PYTHONPATH=/workspace/InfiniSST，所以是 agents.infinisst_omni_vllm_rag.InfiniSSTOmniVLLMRAG
    echo 'Starting SimulEval with unbuffered output...'
    python -u \$(which simuleval) \\
      --agent /workspace/InfiniSST/agents/infinisst_omni_vllm_rag.py \\
      --agent-class agents.infinisst_omni_vllm_rag.InfiniSSTOmniVLLMRAG \\
      --source-segment-size ${SRC_SEGMENT_SIZE} \\
      --source-lang English \\
      --target-lang ${LANG} \\
      --min-start-sec ${MIN_START_SEC} \\
      --source ${SOURCE_LIST} \\
      --target ${TARGET_LIST} \\
      --output ${OUTPUT_PATH} \\
      \\
      --max-new-tokens ${MAX_NEW_TOKENS} \\
      --beam ${BEAM} \\
      --no-repeat-ngram-lookback ${NO_REPEAT_NGRAM_LOOKBACK} \\
      --no-repeat-ngram-size ${NO_REPEAT_NGRAM_SIZE} \\
      --temperature ${TEMPERATURE} \\
      --top-p ${TOP_P} \\
      --top-k ${TOP_K} \\
      \\
      --use-vllm 1 \\
      --model-name ${MODEL_NAME} \\
      --max-cache-chunks ${MAX_CACHE_CHUNKS} \\
      --keep-cache-chunks ${KEEP_CACHE_CHUNKS} \\
      \\
      --quality-metrics BLEU \\
      --eval-latency-unit ${LATENCY_UNIT} \\
      --sacrebleu-tokenizer ${TOKENIZER} \\
      \\
      --rag-enabled \\
      --rag-index-path ${RAG_INDEX_PATH} \\
      --rag-model-path ${RAG_MODEL_PATH} \\
      --rag-base-model Qwen/Qwen2-Audio-7B-Instruct \\
      --rag-device cuda:2 \\
      --rag-top-k 5 \\
      --rag-target-lang zh \\
      --rag-lora-r 16 \\
      --rag-lora-alpha 32 \\
      --rag-lora-dropout 0.0
    
    echo ''
    echo '========================================'
    echo '✅ Job completed successfully!'
    echo '========================================'
  "

DOCKER_EXIT_CODE=$?

echo ""
echo "========================================"
echo "Docker container finished"
echo "Exit code: ${DOCKER_EXIT_CODE}"
echo "Time: $(date)"
echo "========================================"

if [ ${DOCKER_EXIT_CODE} -eq 0 ]; then
  echo "✅ Docker job completed successfully!"
else
  echo "❌ Docker job failed with exit code ${DOCKER_EXIT_CODE}"
fi

