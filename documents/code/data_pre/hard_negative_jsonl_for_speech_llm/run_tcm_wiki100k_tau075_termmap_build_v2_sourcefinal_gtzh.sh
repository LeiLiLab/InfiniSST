#!/bin/bash
#SBATCH --job-name=tcm_w100kgt_d9k20_tm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --partition=taurus
#SBATCH --time=24:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_tcm_w100kgt_v2_tm.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_tcm_w100kgt_v2_tm.err

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

SOURCE_JSONL="/mnt/gemini/data/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl"
WIKI100KGT_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample100000_zh100k_train_gt.json"
TCM_RAG_CKPT="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_ntcm_final_v3_n64_p1n4_s2000_aries-8gpu_smallest_dense_smoke4650_best.pt"

# Output dir encodes retrieval knobs:
# recall_k=ceil(duration_sec * density=9), tau=0.75 filter, then max_top_k=20 final cap.
OUT_DIR="/mnt/gemini/data1/jiaxuanluo/tcm_wiki100k_gt_zh_tau075_d9_k20_postfiltercap_termmap_v2_sourcefinal_gtzh"
SHARD_DIR="${OUT_DIR}/shards"
RETRIEVER_MERGED="${OUT_DIR}/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_sourcefinal_tcmwiki100kgt_tau075_d9_k20_postfiltercap_retriever_results.jsonl"
TRAIN_JSONL="${OUT_DIR}/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_sourcefinal_tcmwiki100kgt_tau075_d9_k20_postfiltercap_gtzhoverride.jsonl"
SAMPLE_JSON="${OUT_DIR}/termmap_sample_conversations.json"
SAMPLE_CHUNKS_JSON="${OUT_DIR}/termmap_sample_chunks.json"
STATS_JSON="${OUT_DIR}/termmap_quality_report.json"

NUM_SHARDS=4
RETRIEVAL_DENSITY=9
MAX_TOP_K=20
TERM_MAP_MAX_TERMS=20
TAU=0.75
RAG_FEATURE_EXTRACTOR_MODEL_ID="openai/whisper-large-v3"

mkdir -p "${OUT_DIR}" "${SHARD_DIR}"

for p in "${SOURCE_JSONL}" "${WIKI100KGT_GLOSSARY}" "${TCM_RAG_CKPT}" "${GENERATE_SCRIPT}" "${REBUILD_SCRIPT}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

echo "[INFO] SOURCE_JSONL=${SOURCE_JSONL}"
echo "[INFO] WIKI100KGT_GLOSSARY=${WIKI100KGT_GLOSSARY}"
echo "[INFO] TCM_RAG_CKPT=${TCM_RAG_CKPT}"
echo "[INFO] RAG_FEATURE_EXTRACTOR_MODEL_ID=${RAG_FEATURE_EXTRACTOR_MODEL_ID}"
echo "[INFO] RETRIEVAL_DENSITY=${RETRIEVAL_DENSITY} MAX_TOP_K=${MAX_TOP_K} TERM_MAP_MAX_TERMS=${TERM_MAP_MAX_TERMS} TAU=${TAU} top_k_mode=duration_sec_cap cap_order=filter_then_cap"
echo "[INFO] OUT_DIR=${OUT_DIR}"
if [[ -n "${TCM_BUILD_GPU_DEVICES_OVERRIDE_CSV:-}" ]]; then
  TCM_BUILD_GPU_DEVICES_OVERRIDE="${TCM_BUILD_GPU_DEVICES_OVERRIDE_CSV//:/,}"
fi

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[INFO] TCM_BUILD_GPU_DEVICES_OVERRIDE=${TCM_BUILD_GPU_DEVICES_OVERRIDE:-<unset>}"

echo "[STAGE 1] Splitting input into ${NUM_SHARDS} shards"
python3 - "${SOURCE_JSONL}" "${SHARD_DIR}" "${NUM_SHARDS}" <<'PY'
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

ALLOCATED_GPU_CSV="${TCM_BUILD_GPU_DEVICES_OVERRIDE:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
IFS=',' read -r -a ALLOCATED_GPUS <<< "${ALLOCATED_GPU_CSV}"
if (( ${#ALLOCATED_GPUS[@]} < NUM_SHARDS )); then
  echo "[ERROR] Need ${NUM_SHARDS} allocated GPUs, got ${ALLOCATED_GPU_CSV:-<unset>}" >&2
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
      --rag_feature_extractor_model_id "${RAG_FEATURE_EXTRACTOR_MODEL_ID}" \
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

echo "[STAGE 4] Rebuilding final training JSONL with GT zh override"
python3 "${REBUILD_SCRIPT}" \
  --input_jsonl "${RETRIEVER_MERGED}" \
  --output_jsonl "${TRAIN_JSONL}" \
  --termmap_mode tcm_filtered_with_gt_backfill \
  --max_terms "${TERM_MAP_MAX_TERMS}" \
  --seed 42

echo "[STAGE 5] Quality report: retriever merged + rebuilt train JSONL (post-run QA; no training)"
python3 - "${RETRIEVER_MERGED}" "${TRAIN_JSONL}" "${SAMPLE_JSON}" "${SAMPLE_CHUNKS_JSON}" "${STATS_JSON}" <<'PY'
import json
import random
import sys
from collections import Counter
from pathlib import Path

# Baseline v4 gigaspeech_zh (historical): for manual comparison only — not WandB-backed.
BASELINE_REF = {
    "note": "legacy HF baseline; compare distributions qualitatively",
    "instances": 12500,
    "chunks": 68705,
    "chunk_with_term_map_ratio": 0.9614,
    "avg_term_map_entries_per_chunk": 8.16,
    "max_term_map_entries": 20,
    "gt_chunk_ratio": 0.7515,
    "avg_gt_terms_per_chunk": 1.82,
}

retriever_path = Path(sys.argv[1])
train_path = Path(sys.argv[2])
sample_conv_out = Path(sys.argv[3])
sample_chunks_out = Path(sys.argv[4])
stats_out = Path(sys.argv[5])
rng = random.Random(42)


def count_term_map_kv(content: str) -> int:
    """Count term=zh lines after 'term_map:' (rebuild_termmap format)."""
    if "term_map:" not in (content or ""):
        return 0
    body = (content or "").split("term_map:", 1)[1].strip()
    if not body or body.upper().startswith("NONE"):
        return 0
    n = 0
    for line in body.splitlines():
        line = line.strip()
        if not line or line.upper() == "NONE":
            continue
        if "=" in line:
            n += 1
    return n


def system_prompt_ok(messages) -> bool:
    if not messages or messages[0].get("role") != "system":
        return False
    s = messages[0].get("content") or ""
    return "‘term_map’" in s or "'term_map'" in s or "term_map" in s


# --- Retriever merged (post-tau, pre-rebuild) ---
ret_chunks = 0
ret_list_lens: list[int] = []
ret_nonempty = 0
with retriever_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        for row in obj.get("retriever_results_by_chunk") or []:
            ret_chunks += 1
            n = len(row) if isinstance(row, list) else 0
            ret_list_lens.append(n)
            if n > 0:
                ret_nonempty += 1

ret_list_lens.sort()
def pct(xs, p):
    if not xs:
        return 0.0
    k = int(round((p / 100.0) * (len(xs) - 1)))
    return float(xs[max(0, min(k, len(xs) - 1))])

retriever_stats = {
    "path": str(retriever_path),
    "chunks": ret_chunks,
    "chunks_with_any_retrieval_post_tau": ret_nonempty,
    "ratio_nonempty": ret_nonempty / ret_chunks if ret_chunks else 0.0,
    "avg_candidates_per_chunk": sum(ret_list_lens) / len(ret_list_lens) if ret_list_lens else 0.0,
    "max_candidates_per_chunk": max(ret_list_lens) if ret_list_lens else 0,
    "p50_candidates": pct(ret_list_lens, 50),
    "p90_candidates": pct(ret_list_lens, 90),
    "hist_top12": dict(Counter(ret_list_lens).most_common(12)),
}

# --- Rebuilt train JSONL (per user/audio chunk) ---
conv_rows = 0
total_user_chunks = 0
chunks_nonempty_tm = 0
tm_sizes: list[int] = []
gt_by_chunk_lists = []
chunk_records = []

sample_convs = []

with train_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        conv_rows += 1
        gt_bc = obj.get("gt_terms_by_chunk") or []
        messages = obj.get("messages") or []
        sys_ok = system_prompt_ok(messages)
        uidx = 0
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content") or ""
            if "<audio>" not in content:
                continue
            gt_terms = gt_bc[uidx] if uidx < len(gt_bc) else []
            uidx += 1
            total_user_chunks += 1
            n_tm = count_term_map_kv(content)
            tm_sizes.append(n_tm)
            if n_tm > 0:
                chunks_nonempty_tm += 1
            gt_by_chunk_lists.append(gt_terms)
            chunk_records.append({
                "conv_idx": conv_rows - 1,
                "chunk_idx": uidx - 1,
                "n_term_map": n_tm,
                "n_gt": len(gt_terms),
                "content_preview": (content[:1200] + "…") if len(content) > 1200 else content,
                "audio0": (obj.get("audios") or [None])[0],
            })
        if len(sample_convs) < 12:
            sample_convs.append({
                "conv_idx": conv_rows - 1,
                "system_ok_curly_term_map": sys_ok,
                "first_system": (messages[0].get("content") if messages else "")[:400],
                "audios": obj.get("audios"),
                "first_user": next((m.get("content") for m in messages if m.get("role") == "user"), "")[:1500],
            })

tm_sizes.sort()
chunks_with_gt = sum(1 for g in gt_by_chunk_lists if g)
gt_term_total = sum(len(g) for g in gt_by_chunk_lists)

train_stats = {
    "path": str(train_path),
    "conversations": conv_rows,
    "chunks": total_user_chunks,
    "chunk_with_nonempty_term_map_ratio": chunks_nonempty_tm / total_user_chunks if total_user_chunks else 0.0,
    "avg_term_map_entries_per_chunk": sum(tm_sizes) / len(tm_sizes) if tm_sizes else 0.0,
    "max_term_map_entries": max(tm_sizes) if tm_sizes else 0,
    "p50_term_map_entries": pct(tm_sizes, 50),
    "p90_term_map_entries": pct(tm_sizes, 90),
    "hist_term_map_size_top12": dict(Counter(tm_sizes).most_common(12)),
    "gt_chunk_ratio": chunks_with_gt / total_user_chunks if total_user_chunks else 0.0,
    "avg_gt_terms_per_chunk": gt_term_total / total_user_chunks if total_user_chunks else 0.0,
}

# Stratified chunk samples: a few random + a few high term_map count
if chunk_records:
    high = sorted(chunk_records, key=lambda r: -r["n_term_map"])[:15]
    pick = list(high)
    picked_ids = {(r["conv_idx"], r["chunk_idx"]) for r in pick}
    pool = [r for r in chunk_records if (r["conv_idx"], r["chunk_idx"]) not in picked_ids]
    rng.shuffle(pool)
    pick.extend(pool[:25])
    sample_chunks = pick[:40]
else:
    sample_chunks = []

report = {
    "baseline_reference_only": BASELINE_REF,
    "retriever_merged": retriever_stats,
    "rebuilt_train_jsonl": train_stats,
}
stats_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
sample_conv_out.write_text(json.dumps(sample_convs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
sample_chunks_out.write_text(json.dumps(sample_chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print("[STAGE5] wrote", stats_out, flush=True)
print("[STAGE5] wrote", sample_conv_out, flush=True)
print("[STAGE5] wrote", sample_chunks_out, flush=True)
print("[STAGE5] summary train:", json.dumps(train_stats, ensure_ascii=False)[:500], "...", flush=True)
print("[STAGE5] summary retriever:", json.dumps(retriever_stats, ensure_ascii=False)[:500], "...", flush=True)
PY

echo "[DONE] RETRIEVER_MERGED=${RETRIEVER_MERGED}"
echo "[DONE] TRAIN_JSONL=${TRAIN_JSONL}"
echo "[DONE] SAMPLE_JSON=${SAMPLE_JSON}"
echo "[DONE] SAMPLE_CHUNKS_JSON=${SAMPLE_CHUNKS_JSON}"
echo "[DONE] STATS_JSON=${STATS_JSON}"
