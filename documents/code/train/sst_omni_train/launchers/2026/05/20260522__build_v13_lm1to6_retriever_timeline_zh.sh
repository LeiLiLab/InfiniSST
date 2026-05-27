#!/usr/bin/env bash
set -euo pipefail

if [[ -d /mnt/taurus/home/jiaxuanluo/InfiniSST ]]; then
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
else
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
fi

SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/build_retriever_timeline_termmap_sft.py"
OUT_DIR="${OUT_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_v13_lm1to6_retriever_timeline_zh_lh1b88kw_tau073_20260522}"
SHARD_DIR="${SHARD_DIR_OVERRIDE:-${OUT_DIR}/shards}"

TRAIN_INPUT_JSONL="${TRAIN_INPUT_JSONL_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_train_srcchunk_asr_100k_future_ref_gt_zh_20260522/train_s_zh_srcchunk_asr_100k_future_ref_gt_termmap_none.jsonl}"
GLOSSARY_JSON="${GLOSSARY_JSON_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample100000_zh100k_train_gt.json}"
MODEL_PATH="${MODEL_PATH_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"

TEXT_INDEX_PATH="${TEXT_INDEX_PATH_OVERRIDE:-${OUT_DIR}/lh1b88kw_tau073_zh100k_text_index.pt}"
TRAIN_OUTPUT_JSONL="${TRAIN_OUTPUT_JSONL_OVERRIDE:-${OUT_DIR}/train_s_zh_v13_lm1to6_retriever_timeline_tau073_k10_minctx2p88.jsonl}"
TRAIN_STATS_JSON="${TRAIN_STATS_JSON_OVERRIDE:-${OUT_DIR}/train_s_zh_v13_lm1to6_retriever_timeline_tau073_k10_minctx2p88_stats.json}"
TRAIN_SAMPLE_JSON="${TRAIN_SAMPLE_JSON_OVERRIDE:-${OUT_DIR}/train_s_zh_v13_lm1to6_retriever_timeline_tau073_k10_minctx2p88_samples.json}"
TRAIN_SAMPLE_CHUNKS_JSON="${TRAIN_SAMPLE_CHUNKS_JSON_OVERRIDE:-${OUT_DIR}/train_s_zh_v13_lm1to6_retriever_timeline_tau073_k10_minctx2p88_sample_chunks.json}"
SUMMARY_JSON="${SUMMARY_JSON_OVERRIDE:-${OUT_DIR}/v13_lm1to6_retriever_timeline_summary.json}"

LANG_CODE="${LANG_CODE_OVERRIDE:-zh}"
TOP_K="${TOP_K_OVERRIDE:-10}"
SCORE_THRESHOLD="${SCORE_THRESHOLD_OVERRIDE:-0.73}"
LOOKBACK_SEC="${LOOKBACK_SEC_OVERRIDE:-1.92}"
MIN_CONTEXT_SEC="${MIN_CONTEXT_SEC_OVERRIDE:-2.88}"
MAX_CONTEXT_SEC="${MAX_CONTEXT_SEC_OVERRIDE:-5.76}"
MERGE_MULTIPLIER_MIN="${MERGE_MULTIPLIER_MIN_OVERRIDE:-1}"
MERGE_MULTIPLIER_MAX="${MERGE_MULTIPLIER_MAX_OVERRIDE:-6}"
AUDIO_BATCH_SIZE="${AUDIO_BATCH_SIZE_OVERRIDE:-4}"
TEXT_ENCODE_BATCH="${TEXT_ENCODE_BATCH_OVERRIDE:-256}"
NUM_SHARDS="${NUM_SHARDS_OVERRIDE:-2}"
GPU_DEVICES_CSV="${GPU_DEVICES_CSV_OVERRIDE:-6,7}"
MAX_CONVERSATIONS="${MAX_CONVERSATIONS_OVERRIDE:-0}"
SUMMARY_EVENT="${SUMMARY_EVENT_OVERRIDE:-v13_lm1to6_retriever_timeline_zh}"
export SUMMARY_EVENT

cd "${ROOT_DIR}"
mkdir -p "${OUT_DIR}" "${SHARD_DIR}"

CONDA_PREFIX="${CONDA_PREFIX_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv}"
PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-${CONDA_PREFIX}/bin/python3}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/mnt/taurus/home/jiaxuanluo/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export TORCH_HOME="${TORCH_HOME:-/mnt/gemini/data1/jiaxuanluo/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/gemini/data1/jiaxuanluo/cache}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

for p in "${SCRIPT}" "${TRAIN_INPUT_JSONL}" "${GLOSSARY_JSON}" "${MODEL_PATH}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Missing executable PYTHON_BIN: ${PYTHON_BIN}" >&2
  exit 3
fi

IFS=',' read -r -a GPU_DEVICES <<< "${GPU_DEVICES_CSV}"
if (( ${#GPU_DEVICES[@]} < NUM_SHARDS )); then
  echo "[ERROR] GPU_DEVICES_CSV must provide at least NUM_SHARDS devices: ${GPU_DEVICES_CSV}" >&2
  exit 2
fi

echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] TRAIN_INPUT_JSONL=${TRAIN_INPUT_JSONL}"
echo "[INFO] GLOSSARY_JSON=${GLOSSARY_JSON}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] TEXT_INDEX_PATH=${TEXT_INDEX_PATH}"
echo "[INFO] TOP_K=${TOP_K} SCORE_THRESHOLD=${SCORE_THRESHOLD} LOOKBACK_SEC=${LOOKBACK_SEC} MIN_CONTEXT_SEC=${MIN_CONTEXT_SEC} MAX_CONTEXT_SEC=${MAX_CONTEXT_SEC}"
echo "[INFO] MERGE_MULTIPLIER_MIN=${MERGE_MULTIPLIER_MIN} MERGE_MULTIPLIER_MAX=${MERGE_MULTIPLIER_MAX}"
echo "[INFO] GPU_DEVICES_CSV=${GPU_DEVICES_CSV} NUM_SHARDS=${NUM_SHARDS}"

if [[ ! -f "${TEXT_INDEX_PATH}" ]]; then
  echo "[INFO] Building text index on GPU ${GPU_DEVICES[0]}"
  CUDA_VISIBLE_DEVICES="${GPU_DEVICES[0]}" "${PYTHON_BIN}" "${SCRIPT}" \
    --build-index-only \
    --glossary-json "${GLOSSARY_JSON}" \
    --model-path "${MODEL_PATH}" \
    --text-index-path "${TEXT_INDEX_PATH}" \
    --lang-code "${LANG_CODE}" \
    --device cuda:0 \
    --text-encode-batch "${TEXT_ENCODE_BATCH}"
else
  echo "[INFO] Reusing text index: ${TEXT_INDEX_PATH}"
fi

echo "[INFO] Building V13 train shards"
pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPU_DEVICES[$shard]}"
  shard_out="${SHARD_DIR}/train_v13_shard${shard}_of${NUM_SHARDS}.jsonl"
  shard_stats="${SHARD_DIR}/train_v13_shard${shard}_of${NUM_SHARDS}.stats.json"
  shard_samples="${SHARD_DIR}/train_v13_shard${shard}_of${NUM_SHARDS}.samples.json"
  shard_sample_chunks="${SHARD_DIR}/train_v13_shard${shard}_of${NUM_SHARDS}.sample_chunks.json"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${SCRIPT}" \
      --input-jsonl "${TRAIN_INPUT_JSONL}" \
      --output-jsonl "${shard_out}" \
      --stats-json "${shard_stats}" \
      --sample-json "${shard_samples}" \
      --sample-chunks-json "${shard_sample_chunks}" \
      --glossary-json "${GLOSSARY_JSON}" \
      --model-path "${MODEL_PATH}" \
      --text-index-path "${TEXT_INDEX_PATH}" \
      --lang-code "${LANG_CODE}" \
      --top-k "${TOP_K}" \
      --score-threshold "${SCORE_THRESHOLD}" \
      --lookback-sec "${LOOKBACK_SEC}" \
      --min-context-sec "${MIN_CONTEXT_SEC}" \
      --max-context-sec "${MAX_CONTEXT_SEC}" \
      --merge-multiplier-min "${MERGE_MULTIPLIER_MIN}" \
      --merge-multiplier-max "${MERGE_MULTIPLIER_MAX}" \
      --audio-batch-size "${AUDIO_BATCH_SIZE}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard}" \
      --max-conversations "${MAX_CONVERSATIONS}"
  ) &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "[INFO] Merging train shards"
: > "${TRAIN_OUTPUT_JSONL}"
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  cat "${SHARD_DIR}/train_v13_shard${shard}_of${NUM_SHARDS}.jsonl" >> "${TRAIN_OUTPUT_JSONL}"
done

"${PYTHON_BIN}" - "${SUMMARY_JSON}" "${TRAIN_OUTPUT_JSONL}" "${SHARD_DIR}" "${NUM_SHARDS}" "${TRAIN_STATS_JSON}" <<'PY'
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

summary_path = Path(sys.argv[1])
train_output = Path(sys.argv[2])
shard_dir = Path(sys.argv[3])
num_shards = int(sys.argv[4])
train_stats_path = Path(sys.argv[5])

numeric_sum_keys = {
    "input_rows_seen", "rows_selected_by_shard", "rows_filtered_by_merge_multiplier",
    "rows_written", "dropped_rows", "audio_user_chunks", "gt_chunks", "no_gt_chunks",
    "gt_terms_total", "gt_terms_hit", "gt_chunks_any_hit", "gt_chunks_all_hit",
    "term_map_entries_total", "term_map_gt_entries", "term_map_non_gt_entries",
    "nonempty_term_map_chunks", "no_gt_nonempty_term_map_chunks",
    "rows_missing_gt_terms_by_chunk", "rows_mismatched_audio_gt_counts",
    "rows_mismatched_audio_message_counts",
}

stats = {}
dropped = Counter()
mult_hist = Counter()
termmap_hist = Counter()
duration_buckets = defaultdict(lambda: defaultdict(float))
bucket_sum_keys = {
    "chunks", "gt_chunks", "gt_chunks_any_hit", "gt_chunks_all_hit", "gt_terms",
    "gt_hits", "term_map_entries", "nonempty_term_maps", "no_gt_chunks",
    "no_gt_nonempty_term_maps",
}
elapsed = 0.0
for shard in range(num_shards):
    p = shard_dir / f"train_v13_shard{shard}_of{num_shards}.stats.json"
    cur = json.loads(p.read_text(encoding="utf-8"))
    for k in numeric_sum_keys:
        stats[k] = stats.get(k, 0) + cur.get(k, 0)
    dropped.update(cur.get("dropped_reasons", {}))
    mult_hist.update({str(k): int(v) for k, v in cur.get("multiplier_hist", {}).items()})
    termmap_hist.update({str(k): int(v) for k, v in cur.get("term_map_size_hist", {}).items()})
    elapsed += float(cur.get("elapsed_sec", 0.0))
    for bucket, bstats in cur.get("duration_buckets", {}).items():
        for k, v in bstats.items():
            if k in bucket_sum_keys and isinstance(v, (int, float)):
                duration_buckets[bucket][k] += v

stats["dropped_reasons"] = dict(dropped)
stats["multiplier_hist"] = dict(sorted(mult_hist.items(), key=lambda kv: int(kv[0])))
stats["term_map_size_hist"] = dict(termmap_hist.most_common(50))
stats["duration_buckets"] = {}
for bucket, raw in sorted(duration_buckets.items()):
    cur = dict(raw)
    chunks = float(cur.get("chunks", 0.0))
    gt_terms = float(cur.get("gt_terms", 0.0))
    gt_chunks = float(cur.get("gt_chunks", 0.0))
    no_gt_chunks = float(cur.get("no_gt_chunks", 0.0))
    cur["avg_term_map_entries"] = cur.get("term_map_entries", 0.0) / chunks if chunks else 0.0
    cur["gt_term_recall"] = cur.get("gt_hits", 0.0) / gt_terms if gt_terms else 0.0
    cur["gt_chunk_rate"] = gt_chunks / chunks if chunks else 0.0
    cur["gt_chunk_any_hit_rate"] = cur.get("gt_chunks_any_hit", 0.0) / gt_chunks if gt_chunks else 0.0
    cur["gt_chunk_all_hit_rate"] = cur.get("gt_chunks_all_hit", 0.0) / gt_chunks if gt_chunks else 0.0
    cur["no_gt_nonempty_term_map_rate"] = (
        cur.get("no_gt_nonempty_term_maps", 0.0) / no_gt_chunks if no_gt_chunks else 0.0
    )
    cur["nonempty_term_map_rate"] = cur.get("nonempty_term_maps", 0.0) / chunks if chunks else 0.0
    stats["duration_buckets"][bucket] = cur

stats["gt_term_recall"] = stats["gt_terms_hit"] / stats["gt_terms_total"] if stats.get("gt_terms_total") else 0.0
stats["gt_chunk_any_hit_rate"] = stats["gt_chunks_any_hit"] / stats["gt_chunks"] if stats.get("gt_chunks") else 0.0
stats["gt_chunk_all_hit_rate"] = stats["gt_chunks_all_hit"] / stats["gt_chunks"] if stats.get("gt_chunks") else 0.0
stats["nonempty_term_map_rate"] = stats["nonempty_term_map_chunks"] / stats["audio_user_chunks"] if stats.get("audio_user_chunks") else 0.0
stats["no_gt_nonempty_term_map_rate"] = stats["no_gt_nonempty_term_map_chunks"] / stats["no_gt_chunks"] if stats.get("no_gt_chunks") else 0.0
stats["avg_term_map_entries_per_chunk"] = stats["term_map_entries_total"] / stats["audio_user_chunks"] if stats.get("audio_user_chunks") else 0.0
stats["elapsed_sec_sum_shards"] = round(elapsed, 3)
stats["output_jsonl"] = str(train_output)
train_stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

summary = {
    "event": os.environ.get("SUMMARY_EVENT", "v13_lm1to6_retriever_timeline_zh"),
    "policy": {
        "merge_multiplier": "1..6 only",
        "lookback_sec": 1.92,
        "min_context_sec": 2.88,
        "max_context_sec": 5.76,
        "top_k": 10,
        "score_threshold": 0.73,
        "gt_backfill": False,
        "early_low_lm_chunks": "term_map:NONE until retriever context reaches min_context_sec",
        "high_lm_chunks": "current chunk only when chunk duration is already >= min_context_sec",
    },
    "train_output_jsonl": str(train_output),
    "train_rows": stats["rows_written"],
    "rows_filtered_by_merge_multiplier": stats["rows_filtered_by_merge_multiplier"],
    "audio_user_chunks": stats["audio_user_chunks"],
    "gt_terms_total": stats["gt_terms_total"],
    "gt_terms_hit": stats["gt_terms_hit"],
    "gt_term_recall": stats["gt_term_recall"],
    "gt_chunk_any_hit_rate": stats["gt_chunk_any_hit_rate"],
    "gt_chunk_all_hit_rate": stats["gt_chunk_all_hit_rate"],
    "nonempty_term_map_rate": stats["nonempty_term_map_rate"],
    "no_gt_nonempty_term_map_rate": stats["no_gt_nonempty_term_map_rate"],
    "avg_term_map_entries_per_chunk": stats["avg_term_map_entries_per_chunk"],
    "multiplier_hist": stats["multiplier_hist"],
    "duration_buckets": stats["duration_buckets"],
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
PY

echo "[INFO] V13 train data ready: ${OUT_DIR}"
