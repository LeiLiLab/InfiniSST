#!/bin/bash
set -euo pipefail

# Merge only local completed TTS shards 0-20 for an early speaker-rich scout.
# Keep outputs isolated from the later full 32-shard GSV2 pipeline.

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"
MERGE_SCRIPT="${REPO_ROOT}/documents/code/data_pre/wiki_synth/3variant/merge_tts_3variant.py"

TTS_JSONL_DIR="/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/3variant"
TTS_SHARD_PREFIX="wiki_synth_3variant_gs_v2_clean_with_tts"
TOTAL_TTS_SHARDS=21

OUTPUT_DIR="/mnt/gemini/home/jiaxuanluo/wiki_synth_tts_3variant_gigaspeech_partial0_20"
MERGED_JSONL="${OUTPUT_DIR}/wiki_synth_3variant_gs_v2_clean_partial0_20.jsonl"

echo "[MERGE][GSV2-PARTIAL0_20] Verifying shards 0-$((TOTAL_TTS_SHARDS - 1))"
for i in $(seq 0 $((TOTAL_TTS_SHARDS - 1))); do
    shard_path="${TTS_JSONL_DIR}/${TTS_SHARD_PREFIX}_shard${i}.jsonl"
    if [ ! -f "${shard_path}" ]; then
        echo "[MERGE][ERROR] Missing shard ${i}: ${shard_path}" >&2
        exit 1
    fi
    wc -l "${shard_path}"
done

mkdir -p "${OUTPUT_DIR}"

python "${MERGE_SCRIPT}" \
    --shard_dir "${TTS_JSONL_DIR}" \
    --shard_prefix "${TTS_SHARD_PREFIX}" \
    --total_shards "${TOTAL_TTS_SHARDS}" \
    --output "${MERGED_JSONL}"

echo "[MERGE][GSV2-PARTIAL0_20] Output: ${MERGED_JSONL}"
