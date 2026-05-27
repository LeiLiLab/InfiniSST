#!/bin/bash
#SBATCH --job-name=term_map_v3_rag
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
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

# GPU settings
TOTAL_GPUS="${TOTAL_GPUS:-2}"   # candidate GPU IDs from 0...(TOTAL_GPUS-1)
SKIP_GPUS="${SKIP_GPUS:-2}"     # space-separated list, e.g., "2 7"

cd /mnt/taurus/home/jiaxuanluo/InfiniSST

AVAILABLE_GPUS=()
for g in $(seq 0 $((TOTAL_GPUS - 1))); do
  if [[ " ${SKIP_GPUS} " == *" ${g} "* ]]; then
    echo "[INFO] Skip GPU ${g}"
    continue
  fi
  AVAILABLE_GPUS+=("${g}")
done

if [[ ${#AVAILABLE_GPUS[@]} -eq 0 ]]; then
  echo "[ERROR] No GPUs available after applying SKIP_GPUS=${SKIP_GPUS}"
  exit 1
fi

LOGICAL_TOTAL=${#AVAILABLE_GPUS[@]}
echo "[INFO] Will launch on GPUs: ${AVAILABLE_GPUS[*]} (logical world size=${LOGICAL_TOTAL}, skipped=${SKIP_GPUS})"

for IDX in "${!AVAILABLE_GPUS[@]}"; do
  GPU_DEV="${AVAILABLE_GPUS[$IDX]}"
  echo "[INFO] Launch shard ${IDX}/${LOGICAL_TOTAL} on physical GPU ${GPU_DEV}"
  (
    export CUDA_VISIBLE_DEVICES="${GPU_DEV}"
    python /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/handle_train_dataset_for_term_map_v3_stage2_rag.py \
      --input-gt-jsonl "${IN_GT}" \
      --output-base "${OUT_BASE}" \
      --rag-device "cuda:0" \
      --rag-top-k 20 \
      --rag-batch-size 64 \
      --multiple-range 0 9 \
      --all-negative-ratio 0.1 \
      --gpu-id "${IDX}" \
      --total-gpus "${LOGICAL_TOTAL}"
  ) &
done

wait

echo "[INFO] Done."


