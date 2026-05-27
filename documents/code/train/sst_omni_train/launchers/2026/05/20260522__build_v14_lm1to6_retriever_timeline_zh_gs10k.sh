#!/usr/bin/env bash
set -euo pipefail

if [[ -d /mnt/taurus/home/jiaxuanluo/InfiniSST ]]; then
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
else
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
fi

SOURCE_GT_LAUNCHER="${ROOT_DIR}/documents/code/train/sst_omni_train/launchers/2026/05/20260522__build_train_srcchunk_asr_100k_future_ref_gt_zh.sh"
RETRIEVER_TM_LAUNCHER="${ROOT_DIR}/documents/code/train/sst_omni_train/launchers/2026/05/20260522__build_v13_lm1to6_retriever_timeline_zh.sh"

OUT_DIR="${OUT_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_v14_lm1to6_retriever_timeline_zh_lh1b88kw_tau073_gs10k_20260522}"
SHARD_DIR="${SHARD_DIR_OVERRIDE:-${OUT_DIR}/shards}"
GLOSSARY_JSON="${GLOSSARY_JSON_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample10000_zh.json}"

SOURCE_GT_JSONL="${SOURCE_GT_JSONL_OVERRIDE:-${OUT_DIR}/train_s_zh_srcchunk_asr_10k_future_ref_gt_termmap_none.jsonl}"
SOURCE_GT_STATS="${SOURCE_GT_STATS_OVERRIDE:-${OUT_DIR}/train_s_zh_srcchunk_asr_10k_future_ref_gt_stats.json}"
SOURCE_GT_SAMPLES="${SOURCE_GT_SAMPLES_OVERRIDE:-${OUT_DIR}/train_s_zh_srcchunk_asr_10k_future_ref_gt_samples.json}"
SOURCE_GT_SUMMARY="${SOURCE_GT_SUMMARY_OVERRIDE:-${OUT_DIR}/train_srcchunk_asr_10k_future_ref_gt_summary.json}"

TEXT_INDEX_PATH="${TEXT_INDEX_PATH_OVERRIDE:-${OUT_DIR}/lh1b88kw_tau073_zh10k_text_index.pt}"
TRAIN_OUTPUT_JSONL="${TRAIN_OUTPUT_JSONL_OVERRIDE:-${OUT_DIR}/train_s_zh_v14_lm1to6_retriever_timeline_tau073_k10_minctx2p88_gs10k.jsonl}"
TRAIN_STATS_JSON="${TRAIN_STATS_JSON_OVERRIDE:-${OUT_DIR}/train_s_zh_v14_lm1to6_retriever_timeline_tau073_k10_minctx2p88_gs10k_stats.json}"
TRAIN_SAMPLE_JSON="${TRAIN_SAMPLE_JSON_OVERRIDE:-${OUT_DIR}/train_s_zh_v14_lm1to6_retriever_timeline_tau073_k10_minctx2p88_gs10k_samples.json}"
TRAIN_SAMPLE_CHUNKS_JSON="${TRAIN_SAMPLE_CHUNKS_JSON_OVERRIDE:-${OUT_DIR}/train_s_zh_v14_lm1to6_retriever_timeline_tau073_k10_minctx2p88_gs10k_sample_chunks.json}"
SUMMARY_JSON="${SUMMARY_JSON_OVERRIDE:-${OUT_DIR}/v14_lm1to6_retriever_timeline_gs10k_summary.json}"
VAL_JSONL="${VAL_JSONL_OVERRIDE:-${OUT_DIR}/dev_s_zh_v14_lm1to6_retriever_timeline_tau073_k10_minctx2p88_gs10k_first200.jsonl}"

cd "${ROOT_DIR}"
mkdir -p "${OUT_DIR}" "${SHARD_DIR}"

for p in "${SOURCE_GT_LAUNCHER}" "${RETRIEVER_TM_LAUNCHER}" "${GLOSSARY_JSON}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

echo "[INFO] Building V14 10k source GT: ${SOURCE_GT_JSONL}"
OUT_DIR_OVERRIDE="${OUT_DIR}" \
GLOSSARY_JSON_OVERRIDE="${GLOSSARY_JSON}" \
OUTPUT_JSONL_OVERRIDE="${SOURCE_GT_JSONL}" \
STATS_JSON_OVERRIDE="${SOURCE_GT_STATS}" \
SAMPLE_JSON_OVERRIDE="${SOURCE_GT_SAMPLES}" \
SUMMARY_JSON_OVERRIDE="${SOURCE_GT_SUMMARY}" \
bash "${SOURCE_GT_LAUNCHER}"

echo "[INFO] Building V14 10k retriever timeline term maps: ${TRAIN_OUTPUT_JSONL}"
OUT_DIR_OVERRIDE="${OUT_DIR}" \
SHARD_DIR_OVERRIDE="${SHARD_DIR}" \
TRAIN_INPUT_JSONL_OVERRIDE="${SOURCE_GT_JSONL}" \
GLOSSARY_JSON_OVERRIDE="${GLOSSARY_JSON}" \
TEXT_INDEX_PATH_OVERRIDE="${TEXT_INDEX_PATH}" \
TRAIN_OUTPUT_JSONL_OVERRIDE="${TRAIN_OUTPUT_JSONL}" \
TRAIN_STATS_JSON_OVERRIDE="${TRAIN_STATS_JSON}" \
TRAIN_SAMPLE_JSON_OVERRIDE="${TRAIN_SAMPLE_JSON}" \
TRAIN_SAMPLE_CHUNKS_JSON_OVERRIDE="${TRAIN_SAMPLE_CHUNKS_JSON}" \
SUMMARY_JSON_OVERRIDE="${SUMMARY_JSON}" \
SUMMARY_EVENT_OVERRIDE="v14_lm1to6_retriever_timeline_zh_gs10k" \
GPU_DEVICES_CSV_OVERRIDE="${GPU_DEVICES_CSV_OVERRIDE:-6,7}" \
NUM_SHARDS_OVERRIDE="${NUM_SHARDS_OVERRIDE:-2}" \
bash "${RETRIEVER_TM_LAUNCHER}"

python3 - "${TRAIN_OUTPUT_JSONL}" "${VAL_JSONL}" <<'PY'
import sys
from pathlib import Path
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
with src.open("r", encoding="utf-8") as f, dst.open("w", encoding="utf-8") as g:
    for i, line in enumerate(f):
        if i >= 200:
            break
        g.write(line)
print(dst)
PY

echo "[INFO] V14 train data ready: ${OUT_DIR}"
