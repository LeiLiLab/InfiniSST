#!/usr/bin/env bash
# Build V15 marker-augmented speech-LLM SFT data from V13 retriever timeline data.
set -euo pipefail

if [[ -d /mnt/taurus/home/jiaxuanluo/InfiniSST ]]; then
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
else
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
fi

SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/augment_term_translation_markers.py"
IN_DIR="${IN_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_v13_lm1to6_retriever_timeline_zh_lh1b88kw_tau073_20260522}"
OUT_DIR="${OUT_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_v15_marker_aug_retriever_timeline_zh_lh1b88kw_tau073_20260522}"

TRAIN_INPUT_JSONL="${TRAIN_INPUT_JSONL_OVERRIDE:-${IN_DIR}/train_s_zh_v13_lm1to6_retriever_timeline_tau073_k10_minctx2p88.jsonl}"
DEV_INPUT_JSONL="${DEV_INPUT_JSONL_OVERRIDE:-${IN_DIR}/dev_s_zh_v13_lm1to6_retriever_timeline_tau073_k10_minctx2p88_first200.jsonl}"

TRAIN_OUTPUT_JSONL="${TRAIN_OUTPUT_JSONL_OVERRIDE:-${OUT_DIR}/train_s_zh_v15_marker_aug_tau073_k10_minctx2p88.jsonl}"
DEV_OUTPUT_JSONL="${DEV_OUTPUT_JSONL_OVERRIDE:-${OUT_DIR}/dev_s_zh_v15_marker_aug_tau073_k10_minctx2p88_first200.jsonl}"
TRAIN_STATS_JSON="${TRAIN_STATS_JSON_OVERRIDE:-${OUT_DIR}/train_s_zh_v15_marker_aug_tau073_k10_minctx2p88_stats.json}"
DEV_STATS_JSON="${DEV_STATS_JSON_OVERRIDE:-${OUT_DIR}/dev_s_zh_v15_marker_aug_tau073_k10_minctx2p88_first200_stats.json}"
TRAIN_SAMPLE_JSON="${TRAIN_SAMPLE_JSON_OVERRIDE:-${OUT_DIR}/train_s_zh_v15_marker_aug_samples.json}"
DEV_SAMPLE_JSON="${DEV_SAMPLE_JSON_OVERRIDE:-${OUT_DIR}/dev_s_zh_v15_marker_aug_samples.json}"
SUMMARY_JSON="${SUMMARY_JSON_OVERRIDE:-${OUT_DIR}/v15_marker_aug_summary.json}"

LANG_CODE="${LANG_CODE_OVERRIDE:-zh}"
AUGMENT_PROB="${AUGMENT_PROB_OVERRIDE:-0.50}"
SEED="${SEED_OVERRIDE:-1505}"
MARKER_TEMPLATE="${MARKER_TEMPLATE_OVERRIDE:-}"
if [[ -z "${MARKER_TEMPLATE}" ]]; then
  MARKER_TEMPLATE='{translation}__tm{code}'
fi
MARKER_LEN="${MARKER_LEN_OVERRIDE:-4}"
MAX_AUGMENTED_TERMS_PER_ROW="${MAX_AUGMENTED_TERMS_PER_ROW_OVERRIDE:-8}"

cd "${ROOT_DIR}"
mkdir -p "${OUT_DIR}"

for p in "${SCRIPT}" "${TRAIN_INPUT_JSONL}" "${DEV_INPUT_JSONL}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

echo "[INFO] SCRIPT=${SCRIPT}"
echo "[INFO] TRAIN_INPUT_JSONL=${TRAIN_INPUT_JSONL}"
echo "[INFO] DEV_INPUT_JSONL=${DEV_INPUT_JSONL}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] AUGMENT_PROB=${AUGMENT_PROB} MARKER_TEMPLATE=${MARKER_TEMPLATE}"

train_tmp="${TRAIN_OUTPUT_JSONL}.tmp.$$"
dev_tmp="${DEV_OUTPUT_JSONL}.tmp.$$"
train_stats_tmp="${TRAIN_STATS_JSON}.tmp.$$"
dev_stats_tmp="${DEV_STATS_JSON}.tmp.$$"

cleanup() {
  rm -f "${train_tmp}" "${dev_tmp}" "${train_stats_tmp}" "${dev_stats_tmp}"
}
trap cleanup EXIT

python "${SCRIPT}" \
  --input-jsonl "${TRAIN_INPUT_JSONL}" \
  --output-jsonl "${train_tmp}" \
  --stats-json "${train_stats_tmp}" \
  --sample-json "${TRAIN_SAMPLE_JSON}" \
  --lang-code "${LANG_CODE}" \
  --augment-prob "${AUGMENT_PROB}" \
  --seed "${SEED}" \
  --marker-template "${MARKER_TEMPLATE}" \
  --marker-len "${MARKER_LEN}" \
  --max-augmented-terms-per-row "${MAX_AUGMENTED_TERMS_PER_ROW}"

python "${SCRIPT}" \
  --input-jsonl "${DEV_INPUT_JSONL}" \
  --output-jsonl "${dev_tmp}" \
  --stats-json "${dev_stats_tmp}" \
  --sample-json "${DEV_SAMPLE_JSON}" \
  --lang-code "${LANG_CODE}" \
  --augment-prob "${AUGMENT_PROB}" \
  --seed "$((SEED + 1))" \
  --marker-template "${MARKER_TEMPLATE}" \
  --marker-len "${MARKER_LEN}" \
  --max-augmented-terms-per-row "${MAX_AUGMENTED_TERMS_PER_ROW}"

mv "${train_tmp}" "${TRAIN_OUTPUT_JSONL}"
mv "${dev_tmp}" "${DEV_OUTPUT_JSONL}"
mv "${train_stats_tmp}" "${TRAIN_STATS_JSON}"
mv "${dev_stats_tmp}" "${DEV_STATS_JSON}"
trap - EXIT

python - "${SUMMARY_JSON}" "${TRAIN_STATS_JSON}" "${DEV_STATS_JSON}" "${TRAIN_OUTPUT_JSONL}" "${DEV_OUTPUT_JSONL}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
train_stats_path = Path(sys.argv[2])
dev_stats_path = Path(sys.argv[3])
train_output_path = Path(sys.argv[4])
dev_output_path = Path(sys.argv[5])
train_stats = json.loads(train_stats_path.read_text(encoding="utf-8"))
dev_stats = json.loads(dev_stats_path.read_text(encoding="utf-8"))
train_stats["output_jsonl"] = str(train_output_path)
dev_stats["output_jsonl"] = str(dev_output_path)
train_stats_path.write_text(json.dumps(train_stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
dev_stats_path.write_text(json.dumps(dev_stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
summary = {
    "event": "v15_marker_aug_retriever_timeline_zh",
    "train": train_stats,
    "dev": dev_stats,
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "summary_json": str(summary_path),
    "train_output_jsonl": train_stats.get("output_jsonl"),
    "dev_output_jsonl": dev_stats.get("output_jsonl"),
    "train_augmented_terms": train_stats.get("augmented_terms"),
    "train_augmented_over_gt_terms_rate": train_stats.get("augmented_over_gt_terms_rate"),
    "train_augmented_over_gt_in_map_rate": train_stats.get("augmented_over_gt_in_map_rate"),
}, ensure_ascii=False, indent=2), flush=True)
PY

echo "[INFO] V15 marker augmentation data ready: ${OUT_DIR}"
