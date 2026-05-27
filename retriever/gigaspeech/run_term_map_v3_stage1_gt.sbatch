#!/bin/bash
#SBATCH --job-name=term_map_v3_gt
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:3
#SBATCH --time=48:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_term_map_v3_gt.out
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_term_map_v3_gt.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv

export PYTHONPATH="/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=0
export VLLM_NO_USAGE_STATS=1
# NCCL knobs to avoid init hangs on some clusters
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID}"

# Bind to SLURM-allocated GPU indices (IDX:...) and HARD-skip physical GPU 2.
# Rationale: GPU 2 is frequently occupied by non-SLURM processes; skipping avoids vLLM init failures.
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

# sort allocated GPUs by free mem, then remove GPU 2, then pick 2 GPUs for TP
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
total = {}
for line in q.strip().splitlines():
    i,f,t = [p.strip() for p in line.split(",")]
    free[i] = float(f)
    total[i] = float(t)
vals = [(i, free.get(i, -1.0), total.get(i, 0.0)) for i in idxs]
vals.sort(key=lambda x: x[1], reverse=True)
# HARD skip physical GPU 2 if present
filtered = [i for i,_,__ in vals if i != "2"]
print(",".join(filtered))
PY
)"

if [ -n "${CUDA_SORTED}" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_SORTED}"
fi

echo "[INFO] SLURM IDX list (raw)=${CUDA_LIST:-<empty>}"
echo "[INFO] CUDA_VISIBLE_DEVICES after skip GPU2=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits || true

INPUT_JSONL="/mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline.jsonl"
INPUT_TSV="/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv"
OUT_GT="/mnt/gemini/data1/jiaxuanluo/train_s_zh_v3_gt_terms_open.jsonl"

ALIGN_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
SPACY_MODEL="en_core_web_sm"

cd /mnt/taurus/home/jiaxuanluo/InfiniSST

# Pick 2 GPUs for vLLM TP=2 from CUDA_VISIBLE_DEVICES (already excludes GPU 2).
# We keep only the first 2 GPUs visible to the process so vLLM cannot accidentally use others.
ALIGN_TP_SIZE=2
CUDA_TP_TWO="$(python - <<'PY'
import os
g = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES","").split(",") if x.strip()]
print(",".join(g[:2]))
PY
)"
if [ -z "${CUDA_TP_TWO}" ] || [ "$(echo "${CUDA_TP_TWO}" | tr ',' '\n' | wc -l)" -lt 2 ]; then
  echo "[FATAL] After skipping GPU 2, fewer than 2 GPUs remain for TP=2. Please resubmit later or change node/partition."
  exit 3
fi
export CUDA_VISIBLE_DEVICES="${CUDA_TP_TWO}"
echo "[INFO] Final CUDA_VISIBLE_DEVICES for Stage1 (TP=2)=${CUDA_VISIBLE_DEVICES}"

# Conservative vLLM settings to avoid init failures
ALIGN_MAX_MODEL_LEN=2048
ALIGN_MAX_NUM_SEQS=16
ALIGN_BATCH_SIZE=8
ALIGN_GPU_MEM_UTIL=0.75

python /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/handle_train_dataset_for_term_map_v3_stage1_gt.py \
  --input-jsonl "${INPUT_JSONL}" \
  --input-tsv "${INPUT_TSV}" \
  --output-jsonl "${OUT_GT}" \
  --spacy-model "${SPACY_MODEL}" \
  --align-model "${ALIGN_MODEL}" \
  --align-tensor-parallel-size "${ALIGN_TP_SIZE}" \
  --align-max-model-len "${ALIGN_MAX_MODEL_LEN}" \
  --align-gpu-memory-util "${ALIGN_GPU_MEM_UTIL}" \
  --align-max-num-seqs "${ALIGN_MAX_NUM_SEQS}" \
  --align-batch-size "${ALIGN_BATCH_SIZE}" \
  --extraction-mode closed \
  --max-messages 100 \

echo "[INFO] Done."

#--use-tfidf \


