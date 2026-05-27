#!/usr/bin/env bash
set -euo pipefail

if [[ -d /mnt/taurus/home/jiaxuanluo/InfiniSST ]]; then
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
else
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
fi

SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/build_retriever_timeline_termmap_sft.py"
EVENT_STAMP="${EVENT_STAMP_OVERRIDE:-20260519}"
OUT_DIR="${OUT_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_retriever_timeline_termmap_zh_lh1b88kw_tau073_20260519}"
SHARD_DIR="${SHARD_DIR_OVERRIDE:-${OUT_DIR}/shards}"

TRAIN_INPUT_JSONL="${TRAIN_INPUT_JSONL_OVERRIDE:-/mnt/gemini/data/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl}"
VAL_INPUT_JSONL="${VAL_INPUT_JSONL_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/dev_s_zh_v4_ner_baseline_aligned_freq_k20.jsonl}"
GLOSSARY_JSON="${GLOSSARY_JSON_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample100000_zh100k_train_gt.json}"
MODEL_PATH="${MODEL_PATH_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"

TEXT_INDEX_PATH="${TEXT_INDEX_PATH_OVERRIDE:-${OUT_DIR}/lh1b88kw_tau073_zh100k_text_index.pt}"
TRAIN_OUTPUT_JSONL="${TRAIN_OUTPUT_JSONL_OVERRIDE:-${OUT_DIR}/train_s_zh_retriever_timeline_lh1b88kw_tau073_k10_lb1p92.jsonl}"
VAL_OUTPUT_JSONL="${VAL_OUTPUT_JSONL_OVERRIDE:-${OUT_DIR}/dev_s_zh_retriever_timeline_lh1b88kw_tau073_k10_lb1p92.jsonl}"
TRAIN_STATS_JSON="${TRAIN_STATS_JSON_OVERRIDE:-${OUT_DIR}/train_retriever_timeline_stats.json}"
VAL_STATS_JSON="${VAL_STATS_JSON_OVERRIDE:-${OUT_DIR}/dev_retriever_timeline_stats.json}"
TRAIN_SAMPLE_JSON="${TRAIN_SAMPLE_JSON_OVERRIDE:-${OUT_DIR}/train_retriever_timeline_samples.json}"
VAL_SAMPLE_JSON="${VAL_SAMPLE_JSON_OVERRIDE:-${OUT_DIR}/dev_retriever_timeline_samples.json}"
TRAIN_SAMPLE_CHUNKS_JSON="${TRAIN_SAMPLE_CHUNKS_JSON_OVERRIDE:-${OUT_DIR}/train_retriever_timeline_sample_chunks.json}"
VAL_SAMPLE_CHUNKS_JSON="${VAL_SAMPLE_CHUNKS_JSON_OVERRIDE:-${OUT_DIR}/dev_retriever_timeline_sample_chunks.json}"
SUMMARY_JSON="${SUMMARY_JSON_OVERRIDE:-${OUT_DIR}/retriever_timeline_termmap_manifest.json}"

LANG_CODE="${LANG_CODE_OVERRIDE:-zh}"
TOP_K="${TOP_K_OVERRIDE:-10}"
SCORE_THRESHOLD="${SCORE_THRESHOLD_OVERRIDE:-0.73}"
LOOKBACK_SEC="${LOOKBACK_SEC_OVERRIDE:-1.92}"
AUDIO_BATCH_SIZE="${AUDIO_BATCH_SIZE_OVERRIDE:-4}"
TEXT_ENCODE_BATCH="${TEXT_ENCODE_BATCH_OVERRIDE:-256}"
NUM_SHARDS="${NUM_SHARDS_OVERRIDE:-4}"
GPU_DEVICES_CSV="${GPU_DEVICES_CSV_OVERRIDE:-0,1,2,3}"
MAX_CONVERSATIONS="${MAX_CONVERSATIONS_OVERRIDE:-0}"

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

for p in "${SCRIPT}" "${TRAIN_INPUT_JSONL}" "${VAL_INPUT_JSONL}" "${GLOSSARY_JSON}" "${MODEL_PATH}"; do
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
echo "[INFO] VAL_INPUT_JSONL=${VAL_INPUT_JSONL}"
echo "[INFO] GLOSSARY_JSON=${GLOSSARY_JSON}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] TEXT_INDEX_PATH=${TEXT_INDEX_PATH}"
echo "[INFO] PYTHON_BIN=${PYTHON_BIN}"
echo "[INFO] GPU_DEVICES_CSV=${GPU_DEVICES_CSV}"
echo "[INFO] TOP_K=${TOP_K} SCORE_THRESHOLD=${SCORE_THRESHOLD} LOOKBACK_SEC=${LOOKBACK_SEC}"
echo "[INFO] AUDIO_BATCH_SIZE=${AUDIO_BATCH_SIZE} NUM_SHARDS=${NUM_SHARDS}"

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
  echo "[INFO] Reusing existing text index: ${TEXT_INDEX_PATH}"
fi

echo "[INFO] Building train shards"
pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPU_DEVICES[$shard]}"
  shard_out="${SHARD_DIR}/train_shard${shard}_of${NUM_SHARDS}.jsonl"
  shard_stats="${SHARD_DIR}/train_shard${shard}_of${NUM_SHARDS}.stats.json"
  shard_samples="${SHARD_DIR}/train_shard${shard}_of${NUM_SHARDS}.samples.json"
  shard_sample_chunks="${SHARD_DIR}/train_shard${shard}_of${NUM_SHARDS}.sample_chunks.json"
  echo "[INFO] launch shard=${shard}/${NUM_SHARDS} gpu=${gpu}"
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
  cat "${SHARD_DIR}/train_shard${shard}_of${NUM_SHARDS}.jsonl" >> "${TRAIN_OUTPUT_JSONL}"
done

echo "[INFO] Building validation split on GPU ${GPU_DEVICES[0]}"
CUDA_VISIBLE_DEVICES="${GPU_DEVICES[0]}" "${PYTHON_BIN}" "${SCRIPT}" \
  --input-jsonl "${VAL_INPUT_JSONL}" \
  --output-jsonl "${VAL_OUTPUT_JSONL}" \
  --stats-json "${VAL_STATS_JSON}" \
  --sample-json "${VAL_SAMPLE_JSON}" \
  --sample-chunks-json "${VAL_SAMPLE_CHUNKS_JSON}" \
  --glossary-json "${GLOSSARY_JSON}" \
  --model-path "${MODEL_PATH}" \
  --text-index-path "${TEXT_INDEX_PATH}" \
  --lang-code "${LANG_CODE}" \
  --top-k "${TOP_K}" \
  --score-threshold "${SCORE_THRESHOLD}" \
  --lookback-sec "${LOOKBACK_SEC}" \
  --audio-batch-size "${AUDIO_BATCH_SIZE}" \
  --max-conversations "${MAX_CONVERSATIONS}"

"${PYTHON_BIN}" - "${SUMMARY_JSON}" "${TRAIN_OUTPUT_JSONL}" "${VAL_OUTPUT_JSONL}" \
  "${VAL_STATS_JSON}" "${SHARD_DIR}" "${NUM_SHARDS}" "${TRAIN_STATS_JSON}" \
  "${GLOSSARY_JSON}" "${MODEL_PATH}" "${TEXT_INDEX_PATH}" <<'PY'
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

summary_path = Path(sys.argv[1])
train_output = Path(sys.argv[2])
val_output = Path(sys.argv[3])
val_stats_path = Path(sys.argv[4])
shard_dir = Path(sys.argv[5])
num_shards = int(sys.argv[6])
train_stats_path = Path(sys.argv[7])
glossary_json = Path(sys.argv[8])
model_path = Path(sys.argv[9])
text_index = Path(sys.argv[10])

numeric_sum_keys = {
    "input_rows_seen", "rows_selected_by_shard", "rows_filtered_by_merge_multiplier",
    "rows_written", "dropped_rows",
    "audio_user_chunks", "gt_chunks", "no_gt_chunks", "gt_terms_total",
    "gt_terms_hit", "gt_chunks_any_hit", "gt_chunks_all_hit",
    "term_map_entries_total", "term_map_gt_entries", "term_map_non_gt_entries",
    "nonempty_term_map_chunks", "no_gt_nonempty_term_map_chunks",
    "rows_missing_gt_terms_by_chunk", "rows_mismatched_audio_gt_counts",
    "rows_mismatched_audio_message_counts",
}

train_stats = {}
dropped = Counter()
mult_hist = Counter()
termmap_hist = Counter()
duration_buckets = defaultdict(lambda: defaultdict(float))
bucket_sum_keys = {
    "chunks", "gt_chunks", "gt_terms", "gt_hits", "term_map_entries",
    "nonempty_term_maps", "no_gt_chunks", "no_gt_nonempty_term_maps",
}
elapsed = 0.0
for shard in range(num_shards):
    p = shard_dir / f"train_shard{shard}_of{num_shards}.stats.json"
    cur = json.loads(p.read_text(encoding="utf-8"))
    for k in numeric_sum_keys:
        train_stats[k] = train_stats.get(k, 0) + cur.get(k, 0)
    dropped.update(cur.get("dropped_reasons", {}))
    mult_hist.update({str(k): int(v) for k, v in cur.get("multiplier_hist", {}).items()})
    termmap_hist.update({str(k): int(v) for k, v in cur.get("term_map_size_hist", {}).items()})
    elapsed += float(cur.get("elapsed_sec", 0.0))
    for bucket, bstats in cur.get("duration_buckets", {}).items():
        for k, v in bstats.items():
            if k in bucket_sum_keys and isinstance(v, (int, float)):
                duration_buckets[bucket][k] += v

train_stats["dropped_reasons"] = dict(dropped)
train_stats["multiplier_hist"] = dict(sorted(mult_hist.items(), key=lambda kv: int(kv[0])))
train_stats["term_map_size_hist"] = dict(termmap_hist.most_common(50))
train_stats["duration_buckets"] = {}
for bucket, raw in sorted(duration_buckets.items()):
    cur = dict(raw)
    chunks = max(1.0, float(cur.get("chunks", 0.0)))
    gt_terms = max(1.0, float(cur.get("gt_terms", 0.0)))
    no_gt_chunks = max(1.0, float(cur.get("no_gt_chunks", 0.0)))
    cur["avg_term_map_entries"] = float(cur.get("term_map_entries", 0.0)) / chunks
    cur["gt_term_recall"] = (
        float(cur.get("gt_hits", 0.0)) / gt_terms if cur.get("gt_terms", 0.0) else 0.0
    )
    cur["gt_chunk_rate"] = float(cur.get("gt_chunks", 0.0)) / chunks
    cur["no_gt_nonempty_term_map_rate"] = (
        float(cur.get("no_gt_nonempty_term_maps", 0.0)) / no_gt_chunks
        if cur.get("no_gt_chunks", 0.0) else 0.0
    )
    cur["nonempty_term_map_rate"] = float(cur.get("nonempty_term_maps", 0.0)) / chunks
    train_stats["duration_buckets"][bucket] = cur
train_stats["gt_term_recall"] = train_stats["gt_terms_hit"] / train_stats["gt_terms_total"] if train_stats.get("gt_terms_total") else 0.0
train_stats["gt_chunk_any_hit_rate"] = train_stats["gt_chunks_any_hit"] / train_stats["gt_chunks"] if train_stats.get("gt_chunks") else 0.0
train_stats["gt_chunk_all_hit_rate"] = train_stats["gt_chunks_all_hit"] / train_stats["gt_chunks"] if train_stats.get("gt_chunks") else 0.0
train_stats["nonempty_term_map_rate"] = train_stats["nonempty_term_map_chunks"] / train_stats["audio_user_chunks"] if train_stats.get("audio_user_chunks") else 0.0
train_stats["no_gt_nonempty_term_map_rate"] = train_stats["no_gt_nonempty_term_map_chunks"] / train_stats["no_gt_chunks"] if train_stats.get("no_gt_chunks") else 0.0
train_stats["avg_term_map_entries_per_chunk"] = train_stats["term_map_entries_total"] / train_stats["audio_user_chunks"] if train_stats.get("audio_user_chunks") else 0.0
train_stats["elapsed_sec_sum_shards"] = round(elapsed, 3)
train_stats["output_jsonl"] = str(train_output)
train_stats["note"] = "Merged from shard stats; score quantiles are available in per-shard stats."
train_stats_path.write_text(json.dumps(train_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

val_stats = json.loads(val_stats_path.read_text(encoding="utf-8"))
summary = {
    "event": "retriever_timeline_termmap_zh_lh1b88kw_tau073_sft_data",
    "policy": {
        "retrieval": "timeline MaxSim over [chunk_start - 1.92s, chunk_end]",
        "window_filter": "keep evidence windows overlapping the current streaming speech chunk",
        "top_k": 10,
        "score_threshold": 0.73,
        "gt_backfill": False,
        "gt_translation_override": "only when retrieved source term matches chunk GT term",
        "empty_term_map": "term_map:NONE",
    },
    "train_output_jsonl": str(train_output),
    "val_output_jsonl": str(val_output),
    "train_rows": train_stats["rows_written"],
    "train_audio_user_chunks": train_stats["audio_user_chunks"],
    "train_gt_term_recall": train_stats["gt_term_recall"],
    "train_nonempty_term_map_rate": train_stats["nonempty_term_map_rate"],
    "train_no_gt_nonempty_term_map_rate": train_stats["no_gt_nonempty_term_map_rate"],
    "train_avg_term_map_entries_per_chunk": train_stats["avg_term_map_entries_per_chunk"],
    "val_rows": val_stats["rows_written"],
    "val_audio_user_chunks": val_stats["audio_user_chunks"],
    "val_gt_term_recall": val_stats["gt_term_recall"],
    "val_nonempty_term_map_rate": val_stats["nonempty_term_map_rate"],
    "val_no_gt_nonempty_term_map_rate": val_stats["no_gt_nonempty_term_map_rate"],
    "val_avg_term_map_entries_per_chunk": val_stats["avg_term_map_entries_per_chunk"],
    "glossary_json": str(glossary_json),
    "model_path": str(model_path),
    "text_index_path": str(text_index),
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "[DONE] TRAIN_OUTPUT_JSONL=${TRAIN_OUTPUT_JSONL}"
echo "[DONE] VAL_OUTPUT_JSONL=${VAL_OUTPUT_JSONL}"
echo "[DONE] TRAIN_STATS_JSON=${TRAIN_STATS_JSON}"
echo "[DONE] VAL_STATS_JSON=${VAL_STATS_JSON}"
echo "[DONE] SUMMARY_JSON=${SUMMARY_JSON}"
