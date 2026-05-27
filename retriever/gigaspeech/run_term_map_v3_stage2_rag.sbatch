#!/bin/bash
#SBATCH --job-name=term_map_v3_rag
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_term_map_v3_rag.out
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_term_map_v3_rag.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

export PYTHONPATH="/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

IN_GT="/mnt/gemini/data1/jiaxuanluo/train_s_zh_v3_gt_terms_final.jsonl"
OUT_BASE="/mnt/gemini/data1/jiaxuanluo/train_s_zh_with_candidates_v3_data_driven"

cd /mnt/taurus/home/jiaxuanluo/InfiniSST

python /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/handle_train_dataset_for_term_map_v3_stage2_rag.py \
  --input-gt-jsonl "${IN_GT}" \
  --output-base "${OUT_BASE}" \
  --rag-device "cuda:0" \
  --rag-top-k 20 \
  --rag-batch-size 64 \
  --multiple-range 0 9 \
  --all-negative-ratio 0.1

echo "[INFO] Done."



















