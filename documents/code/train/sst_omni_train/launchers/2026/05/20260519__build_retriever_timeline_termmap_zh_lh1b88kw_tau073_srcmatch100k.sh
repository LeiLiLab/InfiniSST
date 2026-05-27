#!/usr/bin/env bash
set -euo pipefail

if [[ -d /mnt/taurus/home/jiaxuanluo/InfiniSST ]]; then
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
else
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
fi

BASE_LAUNCHER="${ROOT_DIR}/documents/code/train/sst_omni_train/launchers/2026/05/20260519__build_retriever_timeline_termmap_zh_lh1b88kw_tau073.sh"
SRCMATCH_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/build_source_glossary_exact_gt_terms.py"

OUT_DIR="${OUT_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_retriever_timeline_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260519}"
TRAIN_INPUT_JSONL="${TRAIN_INPUT_JSONL_OVERRIDE:-/mnt/gemini/data/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl}"
VAL_INPUT_JSONL="${VAL_INPUT_JSONL_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/dev_s_zh_v4_ner_baseline_aligned_freq_k20.jsonl}"
TRAIN_TSV="${TRAIN_TSV_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv}"
VAL_TSV="${VAL_TSV_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/dev_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv}"
GLOSSARY_JSON="${GLOSSARY_JSON_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample100000_zh100k_train_gt.json}"

SRCMATCH_TRAIN_JSONL="${SRCMATCH_TRAIN_JSONL_OVERRIDE:-${OUT_DIR}/train_s_zh_srcmatch100k_gt.jsonl}"
SRCMATCH_VAL_JSONL="${SRCMATCH_VAL_JSONL_OVERRIDE:-${OUT_DIR}/dev_s_zh_srcmatch100k_gt.jsonl}"
SRCMATCH_TRAIN_STATS="${SRCMATCH_TRAIN_STATS_OVERRIDE:-${OUT_DIR}/train_srcmatch100k_gt_stats.json}"
SRCMATCH_VAL_STATS="${SRCMATCH_VAL_STATS_OVERRIDE:-${OUT_DIR}/dev_srcmatch100k_gt_stats.json}"
SRCMATCH_TRAIN_SAMPLES="${SRCMATCH_TRAIN_SAMPLES_OVERRIDE:-${OUT_DIR}/train_srcmatch100k_gt_samples.json}"
SRCMATCH_VAL_SAMPLES="${SRCMATCH_VAL_SAMPLES_OVERRIDE:-${OUT_DIR}/dev_srcmatch100k_gt_samples.json}"

CONDA_PREFIX="${CONDA_PREFIX_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv}"
PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-${CONDA_PREFIX}/bin/python3}"

for p in "${BASE_LAUNCHER}" "${SRCMATCH_SCRIPT}" "${TRAIN_INPUT_JSONL}" "${VAL_INPUT_JSONL}" "${TRAIN_TSV}" "${VAL_TSV}" "${GLOSSARY_JSON}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Missing executable PYTHON_BIN: ${PYTHON_BIN}" >&2
  exit 3
fi

mkdir -p "${OUT_DIR}"
cd "${ROOT_DIR}"

echo "[INFO] Building source-glossary exact-match GT for train"
"${PYTHON_BIN}" "${SRCMATCH_SCRIPT}" \
  --input-jsonl "${TRAIN_INPUT_JSONL}" \
  --input-tsv "${TRAIN_TSV}" \
  --glossary-json "${GLOSSARY_JSON}" \
  --output-jsonl "${SRCMATCH_TRAIN_JSONL}" \
  --stats-json "${SRCMATCH_TRAIN_STATS}" \
  --sample-json "${SRCMATCH_TRAIN_SAMPLES}" \
  --lang-code zh \
  --max-words 6 \
  --min-norm-chars 2

echo "[INFO] Building source-glossary exact-match GT for dev"
"${PYTHON_BIN}" "${SRCMATCH_SCRIPT}" \
  --input-jsonl "${VAL_INPUT_JSONL}" \
  --input-tsv "${VAL_TSV}" \
  --glossary-json "${GLOSSARY_JSON}" \
  --output-jsonl "${SRCMATCH_VAL_JSONL}" \
  --stats-json "${SRCMATCH_VAL_STATS}" \
  --sample-json "${SRCMATCH_VAL_SAMPLES}" \
  --lang-code zh \
  --max-words 6 \
  --min-norm-chars 2

"${PYTHON_BIN}" - "${SRCMATCH_TRAIN_STATS}" "${SRCMATCH_VAL_STATS}" <<'PY'
import json
import sys
for p in sys.argv[1:]:
    d = json.load(open(p, encoding="utf-8"))
    if d["dropped_rows"] != 0:
        raise SystemExit(f"[ERROR] dropped_rows in {p}: {d['dropped_rows']}")
    if d["rows_written"] <= 0 or d["chunks_total"] <= 0:
        raise SystemExit(f"[ERROR] empty source-match output: {p}")
    print("[CHECK]", p, {
        "rows_written": d["rows_written"],
        "chunks_total": d["chunks_total"],
        "exact_gt_terms_total": d["exact_gt_terms_total"],
        "chunks_with_exact_gt_rate": d["chunks_with_exact_gt_rate"],
        "avg_exact_gt_terms_per_chunk": d["avg_exact_gt_terms_per_chunk"],
        "translation_in_assistant_chunk_rate": d["translation_in_assistant_chunk_rate"],
        "translation_in_assistant_conversation_rate": d["translation_in_assistant_conversation_rate"],
    }, flush=True)
PY

export OUT_DIR_OVERRIDE="${OUT_DIR}"
export SHARD_DIR_OVERRIDE="${OUT_DIR}/shards"
export TRAIN_INPUT_JSONL_OVERRIDE="${SRCMATCH_TRAIN_JSONL}"
export VAL_INPUT_JSONL_OVERRIDE="${SRCMATCH_VAL_JSONL}"
export GLOSSARY_JSON_OVERRIDE="${GLOSSARY_JSON}"
export TEXT_INDEX_PATH_OVERRIDE="${OUT_DIR}/lh1b88kw_tau073_zh100k_text_index.pt"
export TRAIN_OUTPUT_JSONL_OVERRIDE="${OUT_DIR}/train_s_zh_retriever_timeline_lh1b88kw_tau073_srcmatch100k_k10_lb1p92.jsonl"
export VAL_OUTPUT_JSONL_OVERRIDE="${OUT_DIR}/dev_s_zh_retriever_timeline_lh1b88kw_tau073_srcmatch100k_k10_lb1p92.jsonl"
export TRAIN_STATS_JSON_OVERRIDE="${OUT_DIR}/train_retriever_timeline_srcmatch100k_stats.json"
export VAL_STATS_JSON_OVERRIDE="${OUT_DIR}/dev_retriever_timeline_srcmatch100k_stats.json"
export TRAIN_SAMPLE_JSON_OVERRIDE="${OUT_DIR}/train_retriever_timeline_srcmatch100k_samples.json"
export VAL_SAMPLE_JSON_OVERRIDE="${OUT_DIR}/dev_retriever_timeline_srcmatch100k_samples.json"
export TRAIN_SAMPLE_CHUNKS_JSON_OVERRIDE="${OUT_DIR}/train_retriever_timeline_srcmatch100k_sample_chunks.json"
export VAL_SAMPLE_CHUNKS_JSON_OVERRIDE="${OUT_DIR}/dev_retriever_timeline_srcmatch100k_sample_chunks.json"
export SUMMARY_JSON_OVERRIDE="${OUT_DIR}/retriever_timeline_termmap_srcmatch100k_manifest.json"
export GPU_DEVICES_CSV_OVERRIDE="${GPU_DEVICES_CSV_OVERRIDE:-4,5,6,7}"
export NUM_SHARDS_OVERRIDE="${NUM_SHARDS_OVERRIDE:-4}"

bash "${BASE_LAUNCHER}"
