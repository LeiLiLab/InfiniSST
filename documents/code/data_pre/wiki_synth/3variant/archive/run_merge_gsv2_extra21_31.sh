#!/bin/bash
set -euo pipefail

# Merge only the missing suffix for the full GSV2 dataset:
# local shard 21 + teammate merged shards 22-31.

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"
MERGE_SCRIPT="${REPO_ROOT}/documents/code/data_pre/wiki_synth/3variant/merge_gsv2_full0_31.py"

OUTPUT_DIR="/mnt/gemini/home/jiaxuanluo/wiki_synth_tts_3variant_gigaspeech_extra21_31"
MERGED_JSONL="${OUTPUT_DIR}/wiki_synth_3variant_gs_v2_clean_extra21_31.jsonl"

mkdir -p "${OUTPUT_DIR}"

python "${MERGE_SCRIPT}" \
    --local-shards "21" \
    --output "${MERGED_JSONL}" \
    --expected-total 1030799 \
    --verify-sample 1000

echo "[MERGE][GSV2-EXTRA21_31] Output: ${MERGED_JSONL}"
