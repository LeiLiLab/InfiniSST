#!/bin/bash
#SBATCH --job-name=extract_dev_terms
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_extract_dev_terms.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_extract_dev_terms.err

set -euo pipefail

# 环境变量设置
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spacy_gpu_env"
export PATH="$CONDA_PREFIX/bin:/mnt/taurus/home/jiaxuanluo/miniconda3/condabin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="/mnt/gemini/data1/jiaxuanluo/xdg_cache"
mkdir -p "$XDG_CACHE_HOME"

# 注入 cupy 所需的库路径
for d in $CONDA_PREFIX/lib/python*/site-packages/nvidia/*/lib; do
    if [ -d "$d" ]; then export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"; fi
done

# 验证 Python 位置
which python
python --version

export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/.local/lib/python3.10/site-packages:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"
export HF_HOME="/mnt/gemini/data1/jiaxuanluo/huggingface_cache"
mkdir -p "$HF_HOME"

# --- 配置区 ---
RESUME=false
FORCE=false

# 运行模式: all, run, merge
MODE=${1:-"all"}
# --------------

INPUT_TSV="/mnt/gemini/data1/jiaxuanluo/dev_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv"
# --- 实验配置: 合并窗口大小 (1, 2, 3, 4) ---
MULTIPLIER=${2:-2}
# 复用已有的 NER candidates 文件以节省时间
NER_JSONL="/mnt/gemini/data1/jiaxuanluo/ner_candidates_dev_en_core_web_trf.jsonl"

OUTPUT_DIR="/mnt/gemini/data1/jiaxuanluo/term_dev_audio_chunks_m${MULTIPLIER}"
OUTPUT_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_dataset_m${MULTIPLIER}.jsonl"

EXTRA_ARGS=""
if [ "$RESUME" = true ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --resume"
fi
if [ "$FORCE" = true ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --force"
fi

echo "[INFO] Mode: ${MODE}, Resume: ${RESUME}, Force: ${FORCE}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[ERROR] CUDA_VISIBLE_DEVICES is not set."
    exit 1
fi

IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
TOTAL_SHARDS=${#GPU_LIST[@]}

echo "[INFO] Final usable GPUs: ${GPU_LIST[*]}"

cd /mnt/taurus/home/jiaxuanluo/InfiniSST

# --- RUN: Term Extraction ---
if [[ "$MODE" == "all" || "$MODE" == "run" ]]; then
    echo "[RUN] Extracting terms and chunking audio..."
    for i in "${!GPU_LIST[@]}"; do
        CUR_PHYSICAL_GPU="${GPU_LIST[$i]}"
        echo "[INFO] Launching Shard $i on GPU ${CUR_PHYSICAL_GPU}"
        CUDA_VISIBLE_DEVICES="${CUR_PHYSICAL_GPU}" python retriever/gigaspeech/extract_all_terms_from_tsv.py \
          --input-tsv "${INPUT_TSV}" \
          --output-dir "${OUTPUT_DIR}" \
          --output-jsonl "${OUTPUT_JSONL}" \
          --multiplier-merge "${MULTIPLIER}" \
          --ner-candidates-jsonl "${NER_JSONL}" \
          --shard-id "$i" \
          --total-shards "${TOTAL_SHARDS}" \
          ${EXTRA_ARGS} &
        sleep 5
    done
    wait
    echo "[RUN] Completed."
fi

# --- Final Merge ---
if [[ "$MODE" == "all" || "$MODE" == "merge" ]]; then
    echo "[INFO] Merging final shards into ${OUTPUT_JSONL}..."
    FINAL_OUTPUT="${OUTPUT_JSONL}"
    BASE_OUT_PATH="${OUTPUT_JSONL%.jsonl}"
    
    # 清空或创建最终文件
    : > "${FINAL_OUTPUT}"
    
    for i in $(seq 0 $((TOTAL_SHARDS - 1))); do
        SHARD_FILE="${BASE_OUT_PATH}_shard${i}.jsonl"
        if [ -f "${SHARD_FILE}" ]; then
            echo "[INFO] Merging ${SHARD_FILE}..."
            cat "${SHARD_FILE}" >> "${FINAL_OUTPUT}"
            # 暂时不删除 shard 文件，以免出错
            # rm "${SHARD_FILE}"
        else
            echo "[WARN] Shard file not found: ${SHARD_FILE}"
        fi
    done
    echo "[INFO] All shards merged into ${FINAL_OUTPUT}."
fi
