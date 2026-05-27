#!/bin/bash
#SBATCH --job-name=sllm_maxsim
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_sllm_maxsim.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_sllm_maxsim.err

set -euo pipefail

# ======Configuration=====
DOCKER_IMAGE="modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.1"

TRAIN_SCRIPT="/home/jiaxuanluo/InfiniSST/documents/code/train/sst_omni_train/auto_train_sampling_rank32_try.sh"

export CUDA_VISIBLE_DEVICES="6,7"
# ======Configuration=====

echo "[SBATCH] Starting Speech LLM training at $(date)"
echo "[SBATCH] Node: $(hostname)"
echo "[SBATCH] GPUs: ${CUDA_VISIBLE_DEVICES}"

docker run --rm \
  --gpus all \
  --shm-size=32g \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_IB_DISABLE=1 \
  -e NPROC_PER_NODE=2 \
  -e EXPERT_MODEL_PARALLEL_SIZE=2 \
  -e LORA_RANK=16 \
  -e MAX_EPOCHS=1 \
  -e MICRO_BATCH_SIZE=1 \
  -e GLOBAL_BATCH_SIZE=2 \
  -e DATASET_PATH="/mnt/gemini/data1/jiaxuanluo/train_maxsim_enriched_for_speech_llm.jsonl" \
  -e SAVE_BASE="/mnt/aries/data4/jiaxuanluo/speech_llm_maxsim_enriched" \
  -e WANDB_EXP_PREFIX="omni-maxsim-enriched" \
  -e MASTER_PORT=29519 \
  -v /mnt/taurus/home/jiaxuanluo/InfiniSST/documents:/home/jiaxuanluo/InfiniSST/documents \
  -v /mnt/gemini/data/jiaxuanluo:/mnt/gemini/data/jiaxuanluo \
  -v /mnt/gemini/data1/jiaxuanluo:/mnt/gemini/data1/jiaxuanluo \
  -v /mnt/gemini/data2/jiaxuanluo:/workspace \
  -v /mnt/gemini/data2/jiaxuanluo:/mnt/gemini/data2/jiaxuanluo \
  -v /mnt/aries/data4/jiaxuanluo:/mnt/aries/data4/jiaxuanluo \
  "${DOCKER_IMAGE}" \
  bash "${TRAIN_SCRIPT}"

echo "[SBATCH] Done at $(date)"
