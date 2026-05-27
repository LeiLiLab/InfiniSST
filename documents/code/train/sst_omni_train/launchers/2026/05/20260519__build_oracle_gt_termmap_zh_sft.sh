#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/jiaxuanluo/InfiniSST"
SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/build_oracle_gt_termmap_sft.py"

OUT_DIR="${OUT_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_oracle_gt_termmap_zh_20260519}"
TRAIN_INPUT_JSONL="${TRAIN_INPUT_JSONL_OVERRIDE:-/mnt/gemini/data/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl}"
VAL_INPUT_JSONL="${VAL_INPUT_JSONL_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/dev_s_zh_v4_ner_baseline_aligned_freq_k20.jsonl}"

TRAIN_OUTPUT_JSONL="${TRAIN_OUTPUT_JSONL_OVERRIDE:-${OUT_DIR}/train_s_zh_v4_ner_baseline_aligned_rate1p0_k20_oracle_gt_termmap_none.jsonl}"
VAL_OUTPUT_JSONL="${VAL_OUTPUT_JSONL_OVERRIDE:-${OUT_DIR}/dev_s_zh_v4_ner_baseline_aligned_freq_k20_oracle_gt_termmap_none.jsonl}"
TRAIN_STATS_JSON="${TRAIN_STATS_JSON_OVERRIDE:-${OUT_DIR}/train_oracle_gt_termmap_stats.json}"
VAL_STATS_JSON="${VAL_STATS_JSON_OVERRIDE:-${OUT_DIR}/dev_oracle_gt_termmap_stats.json}"
TRAIN_SAMPLE_JSON="${TRAIN_SAMPLE_JSON_OVERRIDE:-${OUT_DIR}/train_oracle_gt_termmap_samples.json}"
VAL_SAMPLE_JSON="${VAL_SAMPLE_JSON_OVERRIDE:-${OUT_DIR}/dev_oracle_gt_termmap_samples.json}"
SUMMARY_JSON="${SUMMARY_JSON_OVERRIDE:-${OUT_DIR}/oracle_gt_termmap_manifest.json}"

LANG_CODE="${LANG_CODE_OVERRIDE:-zh}"
NO_GT_MODE="${NO_GT_MODE_OVERRIDE:-term_map_none}"
MAX_CONVERSATIONS="${MAX_CONVERSATIONS_OVERRIDE:-0}"
DROP_MISSING_GT_ROWS="${DROP_MISSING_GT_ROWS_OVERRIDE:-1}"
DROP_MISMATCHED_GT_ROWS="${DROP_MISMATCHED_GT_ROWS_OVERRIDE:-0}"

FILTER_ARGS=()
if [[ "${DROP_MISSING_GT_ROWS}" == "1" ]]; then
  FILTER_ARGS+=(--drop-missing-gt-rows)
fi
if [[ "${DROP_MISMATCHED_GT_ROWS}" == "1" ]]; then
  FILTER_ARGS+=(--drop-mismatched-gt-rows)
fi

cd "${ROOT_DIR}"
mkdir -p "${OUT_DIR}"

for p in "${SCRIPT}" "${TRAIN_INPUT_JSONL}" "${VAL_INPUT_JSONL}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

echo "[INFO] Building oracle-GT train JSONL"
python3 "${SCRIPT}" \
  --input-jsonl "${TRAIN_INPUT_JSONL}" \
  --output-jsonl "${TRAIN_OUTPUT_JSONL}" \
  --stats-json "${TRAIN_STATS_JSON}" \
  --sample-json "${TRAIN_SAMPLE_JSON}" \
  --lang-code "${LANG_CODE}" \
  --no-gt-mode "${NO_GT_MODE}" \
  --max-conversations "${MAX_CONVERSATIONS}" \
  "${FILTER_ARGS[@]}"

echo "[INFO] Building oracle-GT validation JSONL"
python3 "${SCRIPT}" \
  --input-jsonl "${VAL_INPUT_JSONL}" \
  --output-jsonl "${VAL_OUTPUT_JSONL}" \
  --stats-json "${VAL_STATS_JSON}" \
  --sample-json "${VAL_SAMPLE_JSON}" \
  --lang-code "${LANG_CODE}" \
  --no-gt-mode "${NO_GT_MODE}" \
  --max-conversations "${MAX_CONVERSATIONS}" \
  "${FILTER_ARGS[@]}"

python3 - "${SUMMARY_JSON}" "${TRAIN_INPUT_JSONL}" "${VAL_INPUT_JSONL}" \
  "${TRAIN_OUTPUT_JSONL}" "${VAL_OUTPUT_JSONL}" "${TRAIN_STATS_JSON}" "${VAL_STATS_JSON}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
train_input = Path(sys.argv[2])
val_input = Path(sys.argv[3])
train_output = Path(sys.argv[4])
val_output = Path(sys.argv[5])
train_stats = json.loads(Path(sys.argv[6]).read_text(encoding="utf-8"))
val_stats = json.loads(Path(sys.argv[7]).read_text(encoding="utf-8"))

summary = {
    "event": "oracle_gt_termmap_zh_sft_data",
    "train_input_jsonl": str(train_input),
    "val_input_jsonl": str(val_input),
    "train_output_jsonl": str(train_output),
    "val_output_jsonl": str(val_output),
    "train_rows": train_stats["rows"],
    "train_audio_user_chunks": train_stats["audio_user_chunks"],
    "train_gt_chunk_ratio": train_stats["gt_chunk_ratio"],
    "train_gt_terms_total": train_stats["gt_terms_total"],
    "val_rows": val_stats["rows"],
    "val_audio_user_chunks": val_stats["audio_user_chunks"],
    "val_gt_chunk_ratio": val_stats["gt_chunk_ratio"],
    "val_gt_terms_total": val_stats["gt_terms_total"],
    "policy": train_stats["policy"],
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "[DONE] TRAIN_OUTPUT_JSONL=${TRAIN_OUTPUT_JSONL}"
echo "[DONE] VAL_OUTPUT_JSONL=${VAL_OUTPUT_JSONL}"
echo "[DONE] SUMMARY_JSON=${SUMMARY_JSON}"
