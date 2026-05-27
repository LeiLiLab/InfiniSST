#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd)}"

OUT_DIR="${OUT_DIR:-/mnt/gemini/home/jiaxuanluo/eval_glossaries}"
MIN_NORM_CHARS="${MIN_NORM_CHARS:-2}"

RAW_INPUT="${RAW_INPUT:-${OUT_DIR}/acl6060_tagged_gt_raw_min_norm2.json}"
RAW_OUTPUT="${RAW_OUTPUT:-${OUT_DIR}/acl6060_tagged_gt_raw_min_norm2_sentence_ids.json}"
UNION_INPUT="${UNION_INPUT:-${OUT_DIR}/acl6060_tagged_gt_union_gs10000_min_norm2_backfill.json}"
UNION_OUTPUT="${UNION_OUTPUT:-${OUT_DIR}/acl6060_tagged_gt_union_gs10000_min_norm2_backfill_sentence_ids.json}"
SOURCE_DICT_INPUT="${SOURCE_DICT_INPUT:-${REPO_ROOT}/documents/data/data_pre/glossary_acl6060.json}"
SOURCE_DICT_OUTPUT="${SOURCE_DICT_OUTPUT:-${OUT_DIR}/acl6060_tagged_source_glossary_min_norm2_sentence_ids.json}"

SENTENCE_TERM_MAP_PREFIX="${SENTENCE_TERM_MAP_PREFIX:-${OUT_DIR}/acl6060_tagged_sentence_term_map_min_norm2}"
TERM_OCCURRENCES_JSONL="${TERM_OCCURRENCES_JSONL:-${OUT_DIR}/acl6060_tagged_term_occurrences_min_norm2.jsonl}"
STATS_JSON="${STATS_JSON:-${OUT_DIR}/acl6060_tagged_sentence_ids_min_norm2_stats.json}"

XML_PATH="${XML_PATH:-/mnt/data/siqiouyang/datasets/acl6060/dev/text/xml/ACL.6060.dev.en-xx.en.xml}"
TAGGED_TEXT="${TAGGED_TEXT:-/mnt/data/siqiouyang/datasets/acl6060/dev/text/tagged_terminology/ACL.6060.dev.tagged.en-xx.en.txt}"
SOURCE_TEXT="${SOURCE_TEXT:-/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt}"

echo "[ACL-TAGGED-SENTIDS] out_dir=${OUT_DIR}"
echo "[ACL-TAGGED-SENTIDS] raw_output=${RAW_OUTPUT}"
echo "[ACL-TAGGED-SENTIDS] union_output=${UNION_OUTPUT}"
echo "[ACL-TAGGED-SENTIDS] source_dict_output=${SOURCE_DICT_OUTPUT}"
echo "[ACL-TAGGED-SENTIDS] sentence_term_map_prefix=${SENTENCE_TERM_MAP_PREFIX}"
echo "[ACL-TAGGED-SENTIDS] stats_json=${STATS_JSON}"

python "${REPO_ROOT}/documents/code/data_pre/acl/src/add_sentence_ids_to_tagged_acl_glossary.py" \
  --xml "${XML_PATH}" \
  --tagged-text "${TAGGED_TEXT}" \
  --source-text "${SOURCE_TEXT}" \
  --base-glossary "${SOURCE_DICT_INPUT}" \
  --min-norm-chars "${MIN_NORM_CHARS}" \
  --glossary "${RAW_INPUT}=${RAW_OUTPUT}" \
  --glossary "${UNION_INPUT}=${UNION_OUTPUT}" \
  --glossary "${SOURCE_DICT_INPUT}=${SOURCE_DICT_OUTPUT}" \
  --sentence-term-map-prefix "${SENTENCE_TERM_MAP_PREFIX}" \
  --term-occurrences-jsonl "${TERM_OCCURRENCES_JSONL}" \
  --stats-json "${STATS_JSON}" \
  --langs zh ja de

echo "[ACL-TAGGED-SENTIDS] DONE"
