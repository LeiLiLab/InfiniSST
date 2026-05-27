#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-python3}"
SCRIPT="${ROOT_DIR}/documents/code/data_pre/paper_extracted/build_acl_paper_extracted_union_glossaries.py"
EXTRACTED_DIR="${EXTRACTED_DIR_OVERRIDE:-${ROOT_DIR}/documents/data/data_pre/extracted_glossaries_by_paper}"
FILLER_GLOSSARY="${FILLER_GLOSSARY_OVERRIDE:-${ROOT_DIR}/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs_enriched.json}"
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-${ROOT_DIR}/retriever/gigaspeech/data_pre}"
STATS_JSON="${STATS_JSON_OVERRIDE:-${OUTPUT_DIR}/acl6060_paper_extracted_union_stats_zh.json}"

PAPERS=(
  2022.acl-long.268
  2022.acl-long.367
  2022.acl-long.590
  2022.acl-long.110
  2022.acl-long.117
)

for p in "${SCRIPT}" "${EXTRACTED_DIR}" "${FILLER_GLOSSARY}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

"${PYTHON_BIN}" "${SCRIPT}" \
  --extracted-glossary-dir "${EXTRACTED_DIR}" \
  --filler-glossary "${FILLER_GLOSSARY}" \
  --output-dir "${OUTPUT_DIR}" \
  --target-lang zh \
  --papers "${PAPERS[@]}" \
  --target-sizes 1000 10000 \
  --expected-raw-count 253 \
  --stats-json "${STATS_JSON}"

for p in \
  "${OUTPUT_DIR}/acl6060_paper_extracted_union_raw_zh.json" \
  "${OUTPUT_DIR}/acl6060_paper_extracted_union_gs1000_zh.json" \
  "${OUTPUT_DIR}/acl6060_paper_extracted_union_gs10000_zh.json" \
  "${STATS_JSON}"; do
  if [[ ! -s "${p}" ]]; then
    echo "[ERROR] Missing generated output: ${p}" >&2
    exit 4
  fi
done

echo "[DONE] ACL paper-extracted union glossaries built"
echo "[DONE] stats=${STATS_JSON}"
