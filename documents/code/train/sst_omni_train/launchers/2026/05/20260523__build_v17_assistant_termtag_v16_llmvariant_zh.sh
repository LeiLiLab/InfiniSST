#!/usr/bin/env bash
# Build V17 assistant-side term-tag SFT data from V16 LLM-variant data.
set -euo pipefail

if [[ -d /mnt/taurus/home/jiaxuanluo/InfiniSST ]]; then
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
else
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
fi

SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/wrap_assistant_term_targets.py"
IN_DIR="${IN_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_v16_llm_variant_aug_retriever_timeline_zh_lh1b88kw_tau073_20260522}"
OUT_DIR="${OUT_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_v17_assistant_termtag_v16_llmvariant_zh_20260523}"

TRAIN_INPUT_JSONL="${TRAIN_INPUT_JSONL_OVERRIDE:-${IN_DIR}/train_s_zh_v16_llm_variant_aug_tau073_k10_minctx2p88.jsonl}"
DEV_INPUT_JSONL="${DEV_INPUT_JSONL_OVERRIDE:-${IN_DIR}/dev_s_zh_v16_llm_variant_aug_tau073_k10_minctx2p88_first200.jsonl}"

TRAIN_OUTPUT_JSONL="${TRAIN_OUTPUT_JSONL_OVERRIDE:-${OUT_DIR}/train_s_zh_v17_assistant_termtag_v16_llmvariant_tau073_k10_minctx2p88.jsonl}"
DEV_OUTPUT_JSONL="${DEV_OUTPUT_JSONL_OVERRIDE:-${OUT_DIR}/dev_s_zh_v17_assistant_termtag_v16_llmvariant_tau073_k10_minctx2p88_first200.jsonl}"
TRAIN_STATS_JSON="${TRAIN_STATS_JSON_OVERRIDE:-${OUT_DIR}/train_s_zh_v17_assistant_termtag_v16_llmvariant_tau073_k10_minctx2p88_stats.json}"
DEV_STATS_JSON="${DEV_STATS_JSON_OVERRIDE:-${OUT_DIR}/dev_s_zh_v17_assistant_termtag_v16_llmvariant_tau073_k10_minctx2p88_first200_stats.json}"
TRAIN_SAMPLE_JSON="${TRAIN_SAMPLE_JSON_OVERRIDE:-${OUT_DIR}/train_s_zh_v17_assistant_termtag_v16_llmvariant_samples.json}"
DEV_SAMPLE_JSON="${DEV_SAMPLE_JSON_OVERRIDE:-${OUT_DIR}/dev_s_zh_v17_assistant_termtag_v16_llmvariant_samples.json}"
SUMMARY_JSON="${SUMMARY_JSON_OVERRIDE:-${OUT_DIR}/v17_assistant_termtag_v16_llmvariant_summary.json}"

LANG_CODE="${LANG_CODE_OVERRIDE:-zh}"
TAG_TEMPLATE="${TAG_TEMPLATE_OVERRIDE:-<term>{translation}</term>}"
MIN_TARGET_CHARS="${MIN_TARGET_CHARS_OVERRIDE:-2}"
MAX_TAGS_PER_ROW="${MAX_TAGS_PER_ROW_OVERRIDE:-16}"

cd "${ROOT_DIR}"
mkdir -p "${OUT_DIR}"

for p in "${SCRIPT}" "${TRAIN_INPUT_JSONL}" "${DEV_INPUT_JSONL}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 4
  fi
done

echo "[INFO] SCRIPT=${SCRIPT}"
echo "[INFO] TRAIN_INPUT_JSONL=${TRAIN_INPUT_JSONL}"
echo "[INFO] DEV_INPUT_JSONL=${DEV_INPUT_JSONL}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] TAG_TEMPLATE=${TAG_TEMPLATE} MIN_TARGET_CHARS=${MIN_TARGET_CHARS} MAX_TAGS_PER_ROW=${MAX_TAGS_PER_ROW}"

python3 "${SCRIPT}" \
  --input-jsonl "${TRAIN_INPUT_JSONL}" \
  --output-jsonl "${TRAIN_OUTPUT_JSONL}" \
  --stats-json "${TRAIN_STATS_JSON}" \
  --sample-json "${TRAIN_SAMPLE_JSON}" \
  --lang-code "${LANG_CODE}" \
  --tag-template "${TAG_TEMPLATE}" \
  --min-target-chars "${MIN_TARGET_CHARS}" \
  --max-tags-per-row "${MAX_TAGS_PER_ROW}"

python3 "${SCRIPT}" \
  --input-jsonl "${DEV_INPUT_JSONL}" \
  --output-jsonl "${DEV_OUTPUT_JSONL}" \
  --stats-json "${DEV_STATS_JSON}" \
  --sample-json "${DEV_SAMPLE_JSON}" \
  --lang-code "${LANG_CODE}" \
  --tag-template "${TAG_TEMPLATE}" \
  --min-target-chars "${MIN_TARGET_CHARS}" \
  --max-tags-per-row "${MAX_TAGS_PER_ROW}"

chmod 664 "${TRAIN_OUTPUT_JSONL}" "${DEV_OUTPUT_JSONL}" || true

python3 - "${SUMMARY_JSON}" "${TRAIN_STATS_JSON}" "${DEV_STATS_JSON}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
train_stats_path = Path(sys.argv[2])
dev_stats_path = Path(sys.argv[3])
summary = {
    "event": "v17_assistant_termtag_v16_llmvariant_zh",
    "train": json.loads(train_stats_path.read_text(encoding="utf-8")),
    "dev": json.loads(dev_stats_path.read_text(encoding="utf-8")),
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "summary_json": str(summary_path),
    "train_output_jsonl": summary["train"].get("output_jsonl"),
    "dev_output_jsonl": summary["dev"].get("output_jsonl"),
    "train_assistant_tag_replacements": summary["train"].get("assistant_tag_replacements"),
    "train_assistant_tag_rate_after_min_len": summary["train"].get("assistant_tag_rate_after_min_len"),
    "dev_assistant_tag_replacements": summary["dev"].get("assistant_tag_replacements"),
}, ensure_ascii=False, indent=2), flush=True)
PY

echo "[INFO] V17 assistant term-tag data ready: ${OUT_DIR}"
