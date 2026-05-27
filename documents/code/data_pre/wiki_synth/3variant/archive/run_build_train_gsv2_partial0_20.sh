#!/bin/bash
set -euo pipefail

# Build retriever training data from partial GSV2 MFA shards 0-19 plus the
# original GigaSpeech training set. This is the clean-only speaker-diversity
# scout using TTS shards 0-20 before the full 32-shard pool is ready.

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"
BUILD_SCRIPT="${REPO_ROOT}/documents/code/data_pre/wiki_synth/3variant/build_train_3variant.py"

MFA_DIR="/mnt/gemini/home/jiaxuanluo/MFA/3variant_gsv2_partial0_20/output"
GIGASPEECH_MFA_TRAIN="/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_mfa.jsonl"
NUM_SHARDS=20
SHARD_PATTERN="wiki_synth_train_shard_{:02d}.jsonl"
OUTPUT_TRAIN="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_partial0_20_clean_mfa.jsonl"

echo "[BUILD][GSV2-PARTIAL0_20] Verifying MFA shards in ${MFA_DIR}"
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    shard_path="${MFA_DIR}/wiki_synth_train_shard_$(printf '%02d' "${i}").jsonl"
    if [ ! -f "${shard_path}" ]; then
        echo "[BUILD][ERROR] Missing MFA shard ${i}: ${shard_path}" >&2
        exit 1
    fi
    wc -l "${shard_path}"
done

python "${BUILD_SCRIPT}" \
    --mfa-dir "${MFA_DIR}" \
    --gigaspeech-train "${GIGASPEECH_MFA_TRAIN}" \
    --gigaspeech-skip-wiki-synth \
    --num-shards "${NUM_SHARDS}" \
    --shard-pattern "${SHARD_PATTERN}" \
    --output-train "${OUTPUT_TRAIN}"

echo "[BUILD][GSV2-PARTIAL0_20] Output: ${OUTPUT_TRAIN}"
echo "[BUILD][GSV2-PARTIAL0_20] Stats:  ${OUTPUT_TRAIN%.jsonl}_stats.json"
