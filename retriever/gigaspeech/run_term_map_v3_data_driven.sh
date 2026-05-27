#!/bin/bash
#SBATCH --job-name=term_map_v3_dd
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:3
#SBATCH --time=48:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_term_map_v3_dd.out
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_term_map_v3_dd.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv

echo "[INFO] Starting term_map v3 data-driven job"
echo "[INFO] SLURM_JOB_ID=$SLURM_JOB_ID"

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

echo "[INFO] SLURM-allocated CUDA list (raw)=${CUDA_VISIBLE_DEVICES:-<unset>}"

# --- Reorder allocated GPUs by free memory (within the allocated IDX set) ---
# Goal: avoid picking a heavily-used allocated GPU as TP rank0/1, which can break vLLM init.
# We DO NOT use GPUs outside the allocated set.
CUDA_SORTED="$(python - <<'PY'
import os, subprocess
raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
idxs = [x.strip() for x in raw.split(",") if x.strip() != ""]
if not idxs:
    print("")
    raise SystemExit

# Query free memory for only those GPU indices.
q = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
    text=True,
)
free_map = {}
for line in q.strip().splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 2:
        continue
    free_map[parts[0]] = int(parts[1])

filtered = [(i, free_map.get(i, -1)) for i in idxs]
filtered.sort(key=lambda x: x[1], reverse=True)
print(",".join([i for i,_ in filtered]))
PY
)"

if [ -n "${CUDA_SORTED}" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_SORTED}"
fi

echo "[INFO] CUDA_VISIBLE_DEVICES (sorted by free mem)=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader || true

# Hard guard: ensure first TP GPUs have enough free memory for vLLM.
# vLLM needs roughly gpu_memory_utilization * total_mem available at startup.
export GPU_MEM_UTIL=0.90
export TP_REQUIRED=${ALIGN_TP_SIZE:-2}
python - <<'PY'
import os, subprocess, sys
util = float(os.environ.get("GPU_MEM_UTIL", "0.90"))
tp = int(os.environ.get("TP_REQUIRED", "2"))
visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES","").split(",") if x.strip()]
if len(visible) < tp:
    print(f"[FATAL] Not enough GPUs in CUDA_VISIBLE_DEVICES for TP: have={len(visible)} need={tp}")
    sys.exit(2)

q = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
    text=True,
)
free = {}
total = {}
for line in q.strip().splitlines():
    i, f, t = [p.strip() for p in line.split(",")]
    free[i] = float(f)
    total[i] = float(t)

ok = True
for i in visible[:tp]:
    need = util * total.get(i, 0.0)
    have = free.get(i, 0.0)
    if have < need:
        print(f"[FATAL] GPU {i} free_mem={have:.1f}MiB < required={need:.1f}MiB (util={util}).")
        ok = False
if not ok:
    print("[FATAL] vLLM init would fail due to insufficient free memory on TP GPUs.")
    print("[HINT] Reduce --align-gpu-memory-util (e.g., 0.70) or wait for other processes to release GPU memory.")
    sys.exit(3)
print("[INFO] GPU free memory check passed for TP GPUs.")
PY

# --- Layout recommendation ---
# We reserve the last visible GPU for RAG, and use the first N GPUs for vLLM tensor parallel.
ALIGN_TP_SIZE=2
RAG_DEVICE="cuda:2"     # last GPU within CUDA_VISIBLE_DEVICES

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
  --all-negative-ratio 0.1 \
  --max-messages 100

echo "[INFO] Done."


