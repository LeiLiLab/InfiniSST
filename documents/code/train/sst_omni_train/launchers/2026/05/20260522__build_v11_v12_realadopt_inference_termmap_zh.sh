#!/usr/bin/env bash
set -euo pipefail

if [[ -d /mnt/taurus/home/jiaxuanluo/InfiniSST ]]; then
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
else
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
fi

BASE_TIMELINE_LAUNCHER="${ROOT_DIR}/documents/code/train/sst_omni_train/launchers/2026/05/20260519__build_retriever_timeline_termmap_zh_lh1b88kw_tau073.sh"
RESHAPE_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/reshape_streaming_chunks_for_realadopt_sft.py"
REALADOPT_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/build_realadopt_termmap_sft.py"

SRCMATCH_DIR="${SRCMATCH_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_retriever_timeline_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260519}"
OUT_DIR="${OUT_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_v11_v12_realadopt_inference_zh_lh1b88kw_tau073_20260522}"
RESHAPED_DIR="${RESHAPED_DIR_OVERRIDE:-${OUT_DIR}/reshaped_lm3to6}"
RETRIEVER_DIR="${RETRIEVER_DIR_OVERRIDE:-${OUT_DIR}/retriever_tau073_lm3to6}"

SRCMATCH_TRAIN_JSONL="${SRCMATCH_TRAIN_JSONL_OVERRIDE:-${SRCMATCH_DIR}/train_s_zh_srcmatch100k_gt.jsonl}"
SRCMATCH_DEV_JSONL="${SRCMATCH_DEV_JSONL_OVERRIDE:-${SRCMATCH_DIR}/dev_s_zh_srcmatch100k_gt.jsonl}"
GLOSSARY_JSON="${GLOSSARY_JSON_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample100000_zh100k_train_gt.json}"
MODEL_PATH="${MODEL_PATH_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"

RESHAPED_TRAIN_JSONL="${RESHAPED_DIR}/train_s_zh_srcmatch100k_lm3to6_buffered.jsonl"
RESHAPED_DEV_JSONL="${RESHAPED_DIR}/dev_s_zh_srcmatch100k_lm3to6_buffered.jsonl"
RESHAPED_TRAIN_STATS="${RESHAPED_DIR}/train_s_zh_srcmatch100k_lm3to6_buffered_stats.json"
RESHAPED_DEV_STATS="${RESHAPED_DIR}/dev_s_zh_srcmatch100k_lm3to6_buffered_stats.json"

RETRIEVER_TRAIN_JSONL="${RETRIEVER_DIR}/train_s_zh_retriever_tau073_lm3to6_k10_lb1p92.jsonl"
RETRIEVER_DEV_JSONL="${RETRIEVER_DIR}/dev_s_zh_retriever_tau073_lm3to6_k10_lb1p92.jsonl"
RETRIEVER_TRAIN_STATS="${RETRIEVER_DIR}/train_retriever_tau073_lm3to6_stats.json"
RETRIEVER_DEV_STATS="${RETRIEVER_DIR}/dev_retriever_tau073_lm3to6_stats.json"

V11_TRAIN_JSONL="${OUT_DIR}/train_s_zh_v11_realadopt_realistic_lm3to6_tau073.jsonl"
V11_DEV_JSONL="${OUT_DIR}/dev_s_zh_v11_realadopt_realistic_lm3to6_tau073.jsonl"
V12_TRAIN_JSONL="${OUT_DIR}/train_s_zh_v12_realadopt_marker_lm3to6_tau073.jsonl"
V12_DEV_JSONL="${OUT_DIR}/dev_s_zh_v12_realadopt_marker_lm3to6_tau073.jsonl"

CONDA_PREFIX="${CONDA_PREFIX_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv}"
PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-${CONDA_PREFIX}/bin/python3}"
GPU_DEVICES_CSV="${GPU_DEVICES_CSV_OVERRIDE:-4,5,6,7}"
NUM_SHARDS="${NUM_SHARDS_OVERRIDE:-4}"
MAX_CONVERSATIONS="${MAX_CONVERSATIONS_OVERRIDE:-0}"

for p in "${BASE_TIMELINE_LAUNCHER}" "${RESHAPE_SCRIPT}" "${REALADOPT_SCRIPT}" \
  "${SRCMATCH_TRAIN_JSONL}" "${SRCMATCH_DEV_JSONL}" "${GLOSSARY_JSON}" "${MODEL_PATH}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Missing executable PYTHON_BIN: ${PYTHON_BIN}" >&2
  exit 3
fi

mkdir -p "${OUT_DIR}" "${RESHAPED_DIR}" "${RETRIEVER_DIR}"
cd "${ROOT_DIR}"

echo "[INFO] Reshaping train chunks to inference-like lm=3..6"
"${PYTHON_BIN}" "${RESHAPE_SCRIPT}" \
  --input-jsonl "${SRCMATCH_TRAIN_JSONL}" \
  --output-jsonl "${RESHAPED_TRAIN_JSONL}" \
  --stats-json "${RESHAPED_TRAIN_STATS}" \
  --sample-json "${RESHAPED_DIR}/train_lm3to6_samples.json" \
  --output-audio-dir "${RESHAPED_DIR}/audio" \
  --lang-code zh \
  --min-lm 3 \
  --max-lm 6 \
  --assistant-join "" \
  --drop-bad-rows

echo "[INFO] Reshaping dev chunks to inference-like lm=3..6"
"${PYTHON_BIN}" "${RESHAPE_SCRIPT}" \
  --input-jsonl "${SRCMATCH_DEV_JSONL}" \
  --output-jsonl "${RESHAPED_DEV_JSONL}" \
  --stats-json "${RESHAPED_DEV_STATS}" \
  --sample-json "${RESHAPED_DIR}/dev_lm3to6_samples.json" \
  --output-audio-dir "${RESHAPED_DIR}/audio_dev" \
  --lang-code zh \
  --min-lm 3 \
  --max-lm 6 \
  --assistant-join "" \
  --drop-bad-rows

"${PYTHON_BIN}" - "${RESHAPED_TRAIN_STATS}" "${RESHAPED_DEV_STATS}" <<'PY'
import json, sys
for p in sys.argv[1:]:
    d = json.load(open(p, encoding="utf-8"))
    bad = {k: v for k, v in d["output_lm_hist"].items() if int(k) < 3 or int(k) > 6}
    if bad:
        raise SystemExit(f"[ERROR] invalid output lm hist in {p}: {bad}")
    if d["rows_written"] <= 0 or d["output_chunks"] <= 0:
        raise SystemExit(f"[ERROR] empty reshaped output: {p}")
    print("[CHECK reshaped]", p, {
        "rows_seen": d["rows_seen"],
        "rows_written": d["rows_written"],
        "rows_dropped": d["rows_dropped"],
        "row_keep_rate": d["row_keep_rate"],
        "output_lm_hist": d["output_lm_hist"],
        "avg_output_chunks_per_row": d["avg_output_chunks_per_row"],
        "avg_gt_terms_per_output_chunk": d["avg_gt_terms_per_output_chunk"],
    }, flush=True)
PY

echo "[INFO] Running deployed timeline retriever on reshaped chunks"
OUT_DIR_OVERRIDE="${RETRIEVER_DIR}" \
SHARD_DIR_OVERRIDE="${RETRIEVER_DIR}/shards" \
TRAIN_INPUT_JSONL_OVERRIDE="${RESHAPED_TRAIN_JSONL}" \
VAL_INPUT_JSONL_OVERRIDE="${RESHAPED_DEV_JSONL}" \
GLOSSARY_JSON_OVERRIDE="${GLOSSARY_JSON}" \
MODEL_PATH_OVERRIDE="${MODEL_PATH}" \
TEXT_INDEX_PATH_OVERRIDE="${RETRIEVER_DIR}/lh1b88kw_tau073_zh100k_text_index.pt" \
TRAIN_OUTPUT_JSONL_OVERRIDE="${RETRIEVER_TRAIN_JSONL}" \
VAL_OUTPUT_JSONL_OVERRIDE="${RETRIEVER_DEV_JSONL}" \
TRAIN_STATS_JSON_OVERRIDE="${RETRIEVER_TRAIN_STATS}" \
VAL_STATS_JSON_OVERRIDE="${RETRIEVER_DEV_STATS}" \
TRAIN_SAMPLE_JSON_OVERRIDE="${RETRIEVER_DIR}/train_retriever_tau073_lm3to6_samples.json" \
VAL_SAMPLE_JSON_OVERRIDE="${RETRIEVER_DIR}/dev_retriever_tau073_lm3to6_samples.json" \
TRAIN_SAMPLE_CHUNKS_JSON_OVERRIDE="${RETRIEVER_DIR}/train_retriever_tau073_lm3to6_sample_chunks.json" \
VAL_SAMPLE_CHUNKS_JSON_OVERRIDE="${RETRIEVER_DIR}/dev_retriever_tau073_lm3to6_sample_chunks.json" \
SUMMARY_JSON_OVERRIDE="${RETRIEVER_DIR}/retriever_tau073_lm3to6_manifest.json" \
TOP_K_OVERRIDE=10 \
SCORE_THRESHOLD_OVERRIDE=0.73 \
LOOKBACK_SEC_OVERRIDE=1.92 \
GPU_DEVICES_CSV_OVERRIDE="${GPU_DEVICES_CSV}" \
NUM_SHARDS_OVERRIDE="${NUM_SHARDS}" \
MAX_CONVERSATIONS_OVERRIDE="${MAX_CONVERSATIONS}" \
PYTHON_BIN_OVERRIDE="${PYTHON_BIN}" \
CONDA_PREFIX_OVERRIDE="${CONDA_PREFIX}" \
bash "${BASE_TIMELINE_LAUNCHER}"

build_realadopt_variant() {
  local split="$1"
  local variant="$2"
  local input="$3"
  local output="$4"
  local stats="$5"
  local sample="$6"
  echo "[INFO] Building ${split}/${variant}: ${output}"
  "${PYTHON_BIN}" "${REALADOPT_SCRIPT}" \
    --input-jsonl "${input}" \
    --output-jsonl "${output}" \
    --stats-json "${stats}" \
    --sample-json "${sample}" \
    --variant "${variant}" \
    --lang-code zh \
    --max-terms 10 \
    --max-noise-with-gt 2 \
    --max-noise-without-gt 3 \
    --no-gt-keep-prob 0.35 \
    --seed 20260522
}

build_realadopt_variant train realistic "${RETRIEVER_TRAIN_JSONL}" "${V11_TRAIN_JSONL}" "${OUT_DIR}/train_v11_realistic_stats.json" "${OUT_DIR}/train_v11_realistic_samples.json"
build_realadopt_variant dev realistic "${RETRIEVER_DEV_JSONL}" "${V11_DEV_JSONL}" "${OUT_DIR}/dev_v11_realistic_stats.json" "${OUT_DIR}/dev_v11_realistic_samples.json"
build_realadopt_variant train marker "${RETRIEVER_TRAIN_JSONL}" "${V12_TRAIN_JSONL}" "${OUT_DIR}/train_v12_marker_stats.json" "${OUT_DIR}/train_v12_marker_samples.json"
build_realadopt_variant dev marker "${RETRIEVER_DEV_JSONL}" "${V12_DEV_JSONL}" "${OUT_DIR}/dev_v12_marker_stats.json" "${OUT_DIR}/dev_v12_marker_samples.json"

"${PYTHON_BIN}" - "${OUT_DIR}" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
summary = {}
for p in sorted(out.glob("*_stats.json")):
    d = json.load(open(p, encoding="utf-8"))
    if d.get("dropped_rows", 0):
        raise SystemExit(f"[ERROR] dropped rows in {p}: {d.get('dropped_rows')} {d.get('dropped_reasons')}")
    summary[p.name.replace("_stats.json", "")] = {
        "rows_written": d.get("rows_written"),
        "chunks": d.get("chunks"),
        "avg_retrieved_entries_per_chunk": d.get("avg_retrieved_entries_per_chunk"),
        "avg_term_map_entries_per_chunk": d.get("avg_term_map_entries_per_chunk"),
        "retrieved_exact_local_gt_rate_vs_raw_gt": d.get("retrieved_exact_local_gt_rate_vs_raw_gt"),
        "gt_term_in_term_map_rate": d.get("gt_term_in_term_map_rate"),
        "chunks_with_exact_local_gt_rate": d.get("chunks_with_exact_local_gt_rate"),
        "no_gt_nonempty_term_map_rate": d.get("no_gt_nonempty_term_map_rate"),
        "marker_augmented_gt_entries": d.get("marker_augmented_gt_entries"),
        "assistant_marker_replacements": d.get("assistant_marker_replacements"),
        "term_map_size_hist": d.get("term_map_size_hist"),
    }
summary_path = out / "v11_v12_realadopt_inference_summary.json"
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"[DONE] summary={summary_path}")
PY

echo "[DONE] V11 train=${V11_TRAIN_JSONL}"
echo "[DONE] V11 dev=${V11_DEV_JSONL}"
echo "[DONE] V12 train=${V12_TRAIN_JSONL}"
echo "[DONE] V12 dev=${V12_DEV_JSONL}"
