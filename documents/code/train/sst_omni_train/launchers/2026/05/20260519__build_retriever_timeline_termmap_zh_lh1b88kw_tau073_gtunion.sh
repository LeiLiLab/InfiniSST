#!/usr/bin/env bash
set -euo pipefail

if [[ -d /mnt/taurus/home/jiaxuanluo/InfiniSST ]]; then
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
else
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
fi

BASE_LAUNCHER="${ROOT_DIR}/documents/code/train/sst_omni_train/launchers/2026/05/20260519__build_retriever_timeline_termmap_zh_lh1b88kw_tau073.sh"
UNION_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/build_gt_union_glossary.py"

OUT_DIR="${OUT_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_retriever_timeline_termmap_zh_lh1b88kw_tau073_gtunion_20260519}"
TRAIN_INPUT_JSONL="${TRAIN_INPUT_JSONL_OVERRIDE:-/mnt/gemini/data/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl}"
VAL_INPUT_JSONL="${VAL_INPUT_JSONL_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/dev_s_zh_v4_ner_baseline_aligned_freq_k20.jsonl}"
FILLER_GLOSSARY_JSON="${FILLER_GLOSSARY_JSON_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample100000_zh100k_train_gt.json}"

UNION_GLOSSARY_JSON="${UNION_GLOSSARY_JSON_OVERRIDE:-${OUT_DIR}/gt_union_plus_zh100k_glossary.json}"
UNION_AUDIT_JSON="${UNION_AUDIT_JSON_OVERRIDE:-${OUT_DIR}/gt_union_plus_zh100k_glossary_audit.json}"

CONDA_PREFIX="${CONDA_PREFIX_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv}"
PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-${CONDA_PREFIX}/bin/python3}"

for p in "${BASE_LAUNCHER}" "${UNION_SCRIPT}" "${TRAIN_INPUT_JSONL}" "${VAL_INPUT_JSONL}" "${FILLER_GLOSSARY_JSON}"; do
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

echo "[INFO] Building GT-union glossary"
"${PYTHON_BIN}" "${UNION_SCRIPT}" \
  --input-jsonl train "${TRAIN_INPUT_JSONL}" \
  --input-jsonl dev "${VAL_INPUT_JSONL}" \
  --filler-glossary "${FILLER_GLOSSARY_JSON}" \
  --output-json "${UNION_GLOSSARY_JSON}" \
  --audit-json "${UNION_AUDIT_JSON}" \
  --lang-code zh

export OUT_DIR_OVERRIDE="${OUT_DIR}"
export SHARD_DIR_OVERRIDE="${OUT_DIR}/shards"
export TRAIN_INPUT_JSONL_OVERRIDE="${TRAIN_INPUT_JSONL}"
export VAL_INPUT_JSONL_OVERRIDE="${VAL_INPUT_JSONL}"
export GLOSSARY_JSON_OVERRIDE="${UNION_GLOSSARY_JSON}"
export TEXT_INDEX_PATH_OVERRIDE="${OUT_DIR}/lh1b88kw_tau073_gtunion_text_index.pt"
export TRAIN_OUTPUT_JSONL_OVERRIDE="${OUT_DIR}/train_s_zh_retriever_timeline_lh1b88kw_tau073_gtunion_k10_lb1p92.jsonl"
export VAL_OUTPUT_JSONL_OVERRIDE="${OUT_DIR}/dev_s_zh_retriever_timeline_lh1b88kw_tau073_gtunion_k10_lb1p92.jsonl"
export TRAIN_STATS_JSON_OVERRIDE="${OUT_DIR}/train_retriever_timeline_gtunion_stats.json"
export VAL_STATS_JSON_OVERRIDE="${OUT_DIR}/dev_retriever_timeline_gtunion_stats.json"
export TRAIN_SAMPLE_JSON_OVERRIDE="${OUT_DIR}/train_retriever_timeline_gtunion_samples.json"
export VAL_SAMPLE_JSON_OVERRIDE="${OUT_DIR}/dev_retriever_timeline_gtunion_samples.json"
export TRAIN_SAMPLE_CHUNKS_JSON_OVERRIDE="${OUT_DIR}/train_retriever_timeline_gtunion_sample_chunks.json"
export VAL_SAMPLE_CHUNKS_JSON_OVERRIDE="${OUT_DIR}/dev_retriever_timeline_gtunion_sample_chunks.json"
export SUMMARY_JSON_OVERRIDE="${OUT_DIR}/retriever_timeline_termmap_gtunion_manifest.json"

bash "${BASE_LAUNCHER}"
