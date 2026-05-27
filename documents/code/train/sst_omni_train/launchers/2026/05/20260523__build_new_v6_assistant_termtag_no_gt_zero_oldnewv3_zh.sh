#!/usr/bin/env bash
# Build New V6 assistant-side term-tag data from New V5 no-GT-zero oldnewv3 data.
set -euo pipefail

if [[ -d /mnt/taurus/home/jiaxuanluo/InfiniSST ]]; then
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
else
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
fi

SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/wrap_assistant_term_targets.py"
IN_DIR="${IN_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v5_no_gt_zero_llm_variant_aug_oldnewv3_zh_20260522}"
OUT_DIR="${OUT_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v6_assistant_termtag_no_gt_zero_oldnewv3_zh_20260523}"

TRAIN_INPUT_JSONL="${TRAIN_INPUT_JSONL_OVERRIDE:-${IN_DIR}/train_s_zh_new_v5_no_gt_zero_llm_variant_aug_oldnewv3_cacheonly.jsonl}"
DEV_INPUT_JSONL="${DEV_INPUT_JSONL_OVERRIDE:-${IN_DIR}/dev_s_zh_new_v5_no_gt_zero_llm_variant_aug_oldnewv3_cacheonly.jsonl}"

TRAIN_OUTPUT_JSONL="${TRAIN_OUTPUT_JSONL_OVERRIDE:-${OUT_DIR}/train_s_zh_new_v6_assistant_termtag_no_gt_zero_oldnewv3.jsonl}"
DEV_OUTPUT_JSONL="${DEV_OUTPUT_JSONL_OVERRIDE:-${OUT_DIR}/dev_s_zh_new_v6_assistant_termtag_no_gt_zero_oldnewv3.jsonl}"
TRAIN_STATS_JSON="${TRAIN_STATS_JSON_OVERRIDE:-${OUT_DIR}/train_s_zh_new_v6_assistant_termtag_no_gt_zero_oldnewv3_stats.json}"
DEV_STATS_JSON="${DEV_STATS_JSON_OVERRIDE:-${OUT_DIR}/dev_s_zh_new_v6_assistant_termtag_no_gt_zero_oldnewv3_stats.json}"
TRAIN_SAMPLE_JSON="${TRAIN_SAMPLE_JSON_OVERRIDE:-${OUT_DIR}/train_s_zh_new_v6_assistant_termtag_no_gt_zero_oldnewv3_samples.json}"
DEV_SAMPLE_JSON="${DEV_SAMPLE_JSON_OVERRIDE:-${OUT_DIR}/dev_s_zh_new_v6_assistant_termtag_no_gt_zero_oldnewv3_samples.json}"
SUMMARY_JSON="${SUMMARY_JSON_OVERRIDE:-${OUT_DIR}/new_v6_assistant_termtag_no_gt_zero_oldnewv3_summary.json}"

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

python3 "${SCRIPT}" \
  --input-jsonl "${TRAIN_INPUT_JSONL}" \
  --output-jsonl "${TRAIN_OUTPUT_JSONL}" \
  --stats-json "${TRAIN_STATS_JSON}" \
  --sample-json "${TRAIN_SAMPLE_JSON}" \
  --lang-code "${LANG_CODE}" \
  --tag-template "${TAG_TEMPLATE}" \
  --min-target-chars "${MIN_TARGET_CHARS}" \
  --max-tags-per-row "${MAX_TAGS_PER_ROW}" \
  --missing-gt-policy keep_unchanged

python3 "${SCRIPT}" \
  --input-jsonl "${DEV_INPUT_JSONL}" \
  --output-jsonl "${DEV_OUTPUT_JSONL}" \
  --stats-json "${DEV_STATS_JSON}" \
  --sample-json "${DEV_SAMPLE_JSON}" \
  --lang-code "${LANG_CODE}" \
  --tag-template "${TAG_TEMPLATE}" \
  --min-target-chars "${MIN_TARGET_CHARS}" \
  --max-tags-per-row "${MAX_TAGS_PER_ROW}" \
  --missing-gt-policy keep_unchanged

chmod 664 "${TRAIN_OUTPUT_JSONL}" "${DEV_OUTPUT_JSONL}" || true

python3 - "${SUMMARY_JSON}" "${TRAIN_STATS_JSON}" "${DEV_STATS_JSON}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
train_stats_path = Path(sys.argv[2])
dev_stats_path = Path(sys.argv[3])
summary = {
    "event": "new_v6_assistant_termtag_no_gt_zero_oldnewv3_zh",
    "base_data": "new_v5_no_gt_zero_llm_variant_aug_oldnewv3_zh",
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
}, ensure_ascii=False, indent=2), flush=True)
PY

echo "[INFO] New V6 assistant term-tag data ready: ${OUT_DIR}"
