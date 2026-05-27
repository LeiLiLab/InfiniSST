#!/bin/bash
#SBATCH --job-name=term_map_v3_ner_align
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:3
#SBATCH --time=24:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_term_map_v4_ner_align.out
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_term_map_v4_ner_align.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv

export PYTHONPATH="/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=0
export VLLM_NO_USAGE_STATS=1
# NCCL knobs to avoid init hangs
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID}"

# Bind to SLURM-allocated GPUs and HARD-skip physical GPU 2 if present
CUDA_LIST="$(python - <<'PY'
import re, subprocess, os
out = subprocess.check_output(["scontrol", "show", "job", str(int(os.environ["SLURM_JOB_ID"])), "-dd"], text=True)
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

if [ -n "${CUDA_LIST}" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_LIST}"
fi

# Sort by free mem, skip physical GPU 2, keep first 2 for TP=2
CUDA_SORTED="$(python - <<'PY'
import os, subprocess
raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
idxs = [x.strip() for x in raw.split(",") if x.strip()]
if not idxs:
    print("")
    raise SystemExit
q = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
    text=True,
)
free = {}
for line in q.strip().splitlines():
    i,f,_ = [p.strip() for p in line.split(",")]
    free[i] = float(f)
vals = [(i, free.get(i, -1.0)) for i in idxs]
vals.sort(key=lambda x: x[1], reverse=True)
filtered = [i for i,_ in vals if i != "2"]
print(",".join(filtered[:2]))
PY
)"

if [ -z "${CUDA_SORTED}" ] || [ "$(echo "${CUDA_SORTED}" | tr ',' '\n' | wc -l)" -lt 2 ]; then
  echo "[FATAL] Fewer than 2 GPUs available after skipping GPU 2."
  exit 3
fi
export CUDA_VISIBLE_DEVICES="${CUDA_SORTED}"
echo "[INFO] Final CUDA_VISIBLE_DEVICES (TP=2)=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits || true

INPUT_GT="/mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline.jsonl"
INPUT_TSV="/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv"
OUTPUT_GT="/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_aligned.jsonl"

ALIGN_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

cd /mnt/taurus/home/jiaxuanluo/InfiniSST

python retriever/gigaspeech/handle_train_dataset_for_term_map_v3_stage1_ner_align.py \
  --input-gt "${INPUT_GT}" \
  --input-tsv "${INPUT_TSV}" \
  --output-gt "${OUTPUT_GT}" \
  --align-model "${ALIGN_MODEL}" \
  --gpu-memory-util 0.75 \
  --batch-size 32

echo "[INFO] Done."


















