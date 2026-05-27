#!/bin/bash
#SBATCH --job-name=term_map_v3_dd
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_term_map_v3_dd.out
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_term_map_v3_dd.err

set -euo pipefail

echo "[INFO] Starting term_map v3 data-driven job"
echo "[INFO] SLURM_JOB_ID=$SLURM_JOB_ID"

# --- Conda env ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

# --- Python path ---
export PYTHONPATH="/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

# --- vLLM stability knobs ---
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=0
export VLLM_NO_USAGE_STATS=1

# Optional: reduce fragmentation (safe)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- Bind to SLURM-allocated GPU indices (IDX:...) ---
# This cluster may not enforce GPU isolation (task/none), so we self-restrict via CUDA_VISIBLE_DEVICES.
CUDA_LIST="$(python - <<'PY'
import re, subprocess
out = subprocess.check_output(["scontrol", "show", "job", str(int(__import__("os").environ["SLURM_JOB_ID"])), "-dd"], text=True)
idx_chunks = re.findall(r"IDX:([0-9,\\-]+)", out)
gpus = []
for chunk in idx_chunks:
    for part in chunk.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a,b = part.split("-", 1)
            for x in range(int(a), int(b)+1):
                if x not in gpus:
                    gpus.append(x)
        else:
            x = int(part)
            if x not in gpus:
                gpus.append(x)
print(",".join(map(str, gpus)))
PY
)"

if [ -z "${CUDA_LIST}" ]; then
  echo "[WARN] Failed to parse SLURM GRES IDX; falling back to existing CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_LIST}"
fi

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,pci.bus_id,memory.total --format=csv,noheader || true

# --- Layout recommendation ---
# We reserve the last visible GPU for RAG, and use the first N GPUs for vLLM tensor parallel.
ALIGN_TP_SIZE=3
RAG_DEVICE="cuda:3"     # last GPU within CUDA_VISIBLE_DEVICES

# --- Inputs/Outputs ---
INPUT_JSONL="/mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline.jsonl"
INPUT_TSV="/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv"
OUTPUT_BASE="/mnt/gemini/data1/jiaxuanluo/train_s_zh_with_candidates_v3_data_driven"

# --- LLM model ---
ALIGN_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

# --- spaCy ---
SPACY_MODEL="en_core_web_sm"

cd /mnt/taurus/home/jiaxuanluo/InfiniSST

echo "[INFO] Running v3 data-driven construction..."
python /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/handle_train_dataset_for_term_map_v3_data_driven.py \
  --input-jsonl "${INPUT_JSONL}" \
  --input-tsv "${INPUT_TSV}" \
  --output-base "${OUTPUT_BASE}" \
  --gpu-id 0 \
  --total-gpus 1 \
  --spacy-model "${SPACY_MODEL}" \
  --align-backend vllm \
  --align-model "${ALIGN_MODEL}" \
  --align-tensor-parallel-size "${ALIGN_TP_SIZE}" \
  --align-max-model-len 4096 \
  --align-gpu-memory-util 0.90 \
  --align-max-num-seqs 32 \
  --align-batch-size 16 \
  --rag-device "${RAG_DEVICE}" \
  --rag-top-k 20 \
  --rag-batch-size 64 \
  --multiple-range 0 9 \
  --all-negative-ratio 0.1

echo "[INFO] Done."



















