#!/bin/bash
#############################################
# 在 Docker 容器内部执行：安装依赖 + 跑 SimulEval
# 改进版：添加完整的日志持久化
#############################################

set -e
set -x  # 调试输出

# 从环境变量里取参数（第一块 docker run 已经 -e 传进来了）
MODEL_NAME="${MODEL_NAME}"
SOURCE_LIST="${SOURCE_LIST}"
TARGET_LIST="${TARGET_LIST}"
OUTPUT_PATH="${OUTPUT_PATH}"
SRC_SEGMENT_SIZE="${SRC_SEGMENT_SIZE}"
LATENCY_MULTIPLIER="${LATENCY_MULTIPLIER}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS}"
RAG_INDEX_PATH="${RAG_INDEX_PATH}"
RAG_MODEL_PATH="${RAG_MODEL_PATH}"

# 日志文件路径（保存在 OUTPUT_PATH 同目录下）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${OUTPUT_PATH}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/simuleval_${TIMESTAMP}.log"

echo "===== Logging Configuration ====="
echo "All output will be saved to: ${LOG_FILE}"
echo "You can monitor in real-time with: tail -f ${LOG_FILE}"
echo "=================================="

# 定义主执行函数，这样可以通过管道重定向所有输出
main() {
    echo "===== Environment Variables ====="
    echo "MODEL_NAME       = ${MODEL_NAME}"
    echo "SOURCE_LIST      = ${SOURCE_LIST}"
    echo "TARGET_LIST      = ${TARGET_LIST}"
    echo "OUTPUT_PATH      = ${OUTPUT_PATH}"
    echo "SRC_SEGMENT_SIZE = ${SRC_SEGMENT_SIZE}"
    echo "LATENCY_MULTIPLIER = ${LATENCY_MULTIPLIER}"
    echo "MAX_NEW_TOKENS   = ${MAX_NEW_TOKENS}"
    echo "RAG_INDEX_PATH   = ${RAG_INDEX_PATH}"
    echo "RAG_MODEL_PATH   = ${RAG_MODEL_PATH}"
    echo "=================================="
    
    echo "===== Installing Python dependencies ====="
    
    pip install --quiet --no-cache-dir 'numpy<2' 'faiss-cpu>=1.7.4'
    pip install --quiet --no-cache-dir \
      evaluate jiwer lightning deepspeed torchtune \
      sentence-transformers tensorboardX matplotlib \
      soundfile simuleval jieba unbabel-comet simalign \
      praat-textgrids peft openai
    
    echo "===== Check simuleval ====="
    which simuleval || echo "simuleval not in PATH"
    simuleval --version || echo "simuleval version check failed (可能正常)"
    
    export PYTHONPATH=/workspace/InfiniSST:${PYTHONPATH}
    
    echo "===== Running SimulEval ====="
    echo "Start time: $(date)"
    
    python -u "$(which simuleval)" \
      --agent /workspace/InfiniSST/agents/infinisst_omni_vllm_rag.py \
      --agent-class agents.infinisst_omni_vllm_rag.InfiniSSTOmniVLLMRAG \
      \
      --source "${SOURCE_LIST}" \
      --target "${TARGET_LIST}" \
      --output "${OUTPUT_PATH}" \
      \
      --source-segment-size "${SRC_SEGMENT_SIZE}" \
      --source-lang English \
      --target-lang Chinese \
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
      --model-name "${MODEL_NAME}" \
      --max-cache-chunks 120 \
      --keep-cache-chunks 60 \
      \
      --quality-metrics BLEU \
      --eval-latency-unit char \
      --sacrebleu-tokenizer zh
    
    SIMULEVAL_EXIT_CODE=$?
    
    echo "===== SimulEval DONE ====="
    echo "End time: $(date)"
    echo "Exit code: ${SIMULEVAL_EXIT_CODE}"
    
    # Convert instances.log from Unicode escape to human-readable format
    if [ -f "${OUTPUT_PATH}/instances.log" ]; then
        echo "===== Converting instances.log to human-readable format ====="
        python3 -c "
import sys
import json

input_file = '${OUTPUT_PATH}/instances.log'
output_file = '${OUTPUT_PATH}/back.log'

try:
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                fout.write('\n')
                continue
            try:
                # Parse JSON and re-dump with ensure_ascii=False for readable Chinese
                obj = json.loads(line)
                fout.write(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
            except json.JSONDecodeError as e:
                print(f'Warning: Failed to parse line {line_num}: {e}', file=sys.stderr)
                # Write original line if parsing fails
                fout.write(line + '\n')
    print(f'✅ Human-readable log created: {output_file}')
except Exception as e:
    print(f'❌ Failed to convert instances.log: {e}', file=sys.stderr)
    sys.exit(1)
"
        if [ $? -eq 0 ]; then
            echo "✅ Conversion successful: ${OUTPUT_PATH}/back.log"
            echo "Preview (first 20 lines):"
            head -20 "${OUTPUT_PATH}/back.log"
        else
            echo "⚠️  Conversion failed, but SimulEval completed successfully"
        fi
    else
        echo "⚠️  instances.log not found at ${OUTPUT_PATH}/instances.log"
    fi
    
    return ${SIMULEVAL_EXIT_CODE}
}

# 执行主函数，并将 stdout 和 stderr 都重定向到日志文件和终端
# 使用 tee 可以同时输出到文件和终端
main 2>&1 | tee "${LOG_FILE}"

# 保存退出码
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
echo "Execution finished with exit code: ${EXIT_CODE}"
echo "Full log saved to: ${LOG_FILE}"
echo "=========================================="

exit ${EXIT_CODE}






















