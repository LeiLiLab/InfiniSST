#!/bin/bash
#SBATCH --job-name=tcm_wiki100kgt_tm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --partition=taurus
#SBATCH --time=24:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_tcm_wiki100kgt_tm.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_tcm_wiki100kgt_tm.err

set -euo pipefail

ROOT_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST"
SCRIPT_DIR="${ROOT_DIR}/documents/code/data_pre/hard_negative_jsonl_for_speech_llm"
GENERATE_SCRIPT="${SCRIPT_DIR}/generate_termmap_maxsim.py"
REBUILD_SCRIPT="${SCRIPT_DIR}/rebuild_termmap.py"

CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="/mnt/gemini/data1/jiaxuanluo/huggingface_cache"
export TORCH_HOME="/mnt/gemini/data1/jiaxuanluo/torch_cache"
export XDG_CACHE_HOME="/mnt/gemini/data1/jiaxuanluo/xdg_cache"

CLEANED_JSONL="/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_cleaned.jsonl"
WIKI100KGT_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample100000_zh100k_train_gt.json"
TCM_RAG_CKPT="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_ntcm_final_v3_n64_p1n4_s2000_aries-8gpu_smallest_dense_smoke4650_best.pt"

OUT_DIR="/mnt/gemini/data1/jiaxuanluo/tcm_wiki100k_gt_zh_tau075_termmap"
SHARD_DIR="${OUT_DIR}/shards"
RETRIEVER_MERGED="${OUT_DIR}/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_tcmwiki100kgt_tau075_retriever_results.jsonl"
TRAIN_JSONL="${OUT_DIR}/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_tcmwiki100kgt_tau075_gtbackfill.jsonl"
SAMPLE_JSON="${OUT_DIR}/termmap_sample20.json"

NUM_SHARDS=4
RETRIEVAL_DENSITY=5
MAX_TOP_K=10
TAU=0.75

mkdir -p "${OUT_DIR}" "${SHARD_DIR}"

for p in "${CLEANED_JSONL}" "${WIKI100KGT_GLOSSARY}" "${TCM_RAG_CKPT}" "${GENERATE_SCRIPT}" "${REBUILD_SCRIPT}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

echo "[INFO] CLEANED_JSONL=${CLEANED_JSONL}"
echo "[INFO] WIKI100KGT_GLOSSARY=${WIKI100KGT_GLOSSARY}"
echo "[INFO] TCM_RAG_CKPT=${TCM_RAG_CKPT}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

echo "[STAGE 1] Splitting input into ${NUM_SHARDS} shards"
python3 - "${CLEANED_JSONL}" "${SHARD_DIR}" "${NUM_SHARDS}" <<'PY'
import sys
from pathlib import Path

input_path = Path(sys.argv[1])
shard_dir = Path(sys.argv[2])
n = int(sys.argv[3])
for old in shard_dir.glob("input_shard_*.jsonl"):
    old.unlink()
handles = [(shard_dir / f"input_shard_{i}.jsonl").open("w", encoding="utf-8") for i in range(n)]
try:
    with input_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            handles[idx % n].write(line)
finally:
    for h in handles:
        h.close()
for i in range(n):
    p = shard_dir / f"input_shard_{i}.jsonl"
    count = sum(1 for _ in p.open("r", encoding="utf-8"))
    print(f"[SPLIT] shard={i} lines={count} path={p}", flush=True)
PY

IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if (( ${#ALLOCATED_GPUS[@]} < NUM_SHARDS )); then
  echo "[ERROR] Need ${NUM_SHARDS} allocated GPUs, got ${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
  exit 2
fi

echo "[STAGE 2] Running TCM retriever term_map generation"
pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${ALLOCATED_GPUS[$i]}"
  in_shard="${SHARD_DIR}/input_shard_${i}.jsonl"
  out_shard="${SHARD_DIR}/retriever_results_shard_${i}.jsonl"
  log_shard="${SHARD_DIR}/retriever_results_shard_${i}.log"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" python3 "${GENERATE_SCRIPT}" \
      --cleaned_jsonl "${in_shard}" \
      --glossary_json "${WIKI100KGT_GLOSSARY}" \
      --model_path "${TCM_RAG_CKPT}" \
      --output_jsonl "${out_shard}" \
      --device "cuda:0" \
      --retrieval_density "${RETRIEVAL_DENSITY}" \
      --top_k_mode duration_sec_cap \
      --max_top_k "${MAX_TOP_K}" \
      --score_threshold "${TAU}" \
      --target_lang zh
  ) > "${log_shard}" 2>&1 &
  pids+=("$!")
  echo "[LAUNCH] shard=${i} gpu=${gpu} pid=${pids[-1]}"
  sleep 2
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "[STAGE 3] Merging retriever-result shards"
: > "${RETRIEVER_MERGED}"
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  shard="${SHARD_DIR}/retriever_results_shard_${i}.jsonl"
  if [[ ! -s "${shard}" ]]; then
    echo "[ERROR] Missing/empty retriever shard: ${shard}" >&2
    exit 3
  fi
  cat "${shard}" >> "${RETRIEVER_MERGED}"
done

echo "[STAGE 4] Rebuilding final training JSONL with GT backfill"
python3 "${REBUILD_SCRIPT}" \
  --input_jsonl "${RETRIEVER_MERGED}" \
  --output_jsonl "${TRAIN_JSONL}" \
  --termmap_mode tcm_filtered_with_gt_backfill \
  --seed 42

echo "[STAGE 5] Sampling generated term_map examples"
python3 - "${TRAIN_JSONL}" "${SAMPLE_JSON}" <<'PY'
import json
import sys
from pathlib import Path

inp = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
with inp.open("r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        for msg in obj.get("messages", []):
            if msg.get("role") != "user":
                continue
            content = msg.get("content") or ""
            if "term_map:" not in content:
                continue
            rows.append({
                "content": content,
                "audios": obj.get("audios", []),
            })
            break
        if len(rows) >= 20:
            break
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[SAMPLE] wrote {len(rows)} examples to {out}", flush=True)
PY

echo "[DONE] RETRIEVER_MERGED=${RETRIEVER_MERGED}"
echo "[DONE] TRAIN_JSONL=${TRAIN_JSONL}"
echo "[DONE] SAMPLE_JSON=${SAMPLE_JSON}"
