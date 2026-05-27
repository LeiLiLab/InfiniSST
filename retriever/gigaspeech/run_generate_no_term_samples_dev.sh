#!/bin/bash
#SBATCH --job-name=gen_neg_dev
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_gen_neg_dev.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_gen_neg_dev.err

set -euo pipefail

# 环境设置
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="$CONDA_PREFIX/bin:/mnt/taurus/home/jiaxuanluo/miniconda3/condabin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

# 配置路径（指向 dev 集）
INPUT_TSV="/mnt/gemini/data1/jiaxuanluo/dev_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv"
INPUT_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_dataset.jsonl"
OUTPUT_DIR="/mnt/gemini/data1/jiaxuanluo/term_dev_audio_chunks"
OUTPUT_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_dataset_no_term.jsonl"

# 运行脚本
python generate_no_term_samples.py \
    --input-tsv "${INPUT_TSV}" \
    --input-jsonl "${INPUT_JSONL}" \
    --output-dir "${OUTPUT_DIR}" \
    --output-jsonl "${OUTPUT_JSONL}" \
    --ratio 0.1 \
    --num-workers 32

echo "[INFO] Negative samples generation for dev set completed!"


















