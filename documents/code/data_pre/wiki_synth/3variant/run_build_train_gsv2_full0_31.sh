#!/bin/bash
set -euo pipefail

# Build retriever training data from full GSV2 MFA rows plus the original
# MFA-enriched GigaSpeech rows. Reuse existing partial0_20 MFA output and append
# the extra21_31 MFA output with offset-preserved utter_ids.

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"
BUILD_SCRIPT="${REPO_ROOT}/documents/code/data_pre/wiki_synth/3variant/build_train_3variant.py"

PARTIAL_MFA_DIR="/mnt/gemini/home/jiaxuanluo/MFA/3variant_gsv2_partial0_20/output"
EXTRA_MFA_DIR="/mnt/gemini/home/jiaxuanluo/MFA/3variant_gsv2_extra21_31/output"
COMBINED_MFA_DIR="/mnt/gemini/home/jiaxuanluo/MFA/3variant_gsv2_full0_31_from_parts/output"
GIGASPEECH_MFA_TRAIN="/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_mfa.jsonl"
PARTIAL_SHARDS=20
EXTRA_SHARDS=11
NUM_SHARDS=$((PARTIAL_SHARDS + EXTRA_SHARDS))
SHARD_PATTERN="wiki_synth_train_shard_{:02d}.jsonl"
OUTPUT_TRAIN="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa.jsonl"

echo "[BUILD][GSV2-FULL0_31] Verifying partial MFA shards in ${PARTIAL_MFA_DIR}"
for i in $(seq 0 $((PARTIAL_SHARDS - 1))); do
    shard_path="${PARTIAL_MFA_DIR}/wiki_synth_train_shard_$(printf '%02d' "${i}").jsonl"
    if [ ! -f "${shard_path}" ]; then
        echo "[BUILD][ERROR] Missing partial MFA shard ${i}: ${shard_path}" >&2
        exit 1
    fi
    wc -l "${shard_path}"
done

echo "[BUILD][GSV2-FULL0_31] Verifying extra MFA shards in ${EXTRA_MFA_DIR}"
for i in $(seq 0 $((EXTRA_SHARDS - 1))); do
    shard_path="${EXTRA_MFA_DIR}/wiki_synth_train_shard_$(printf '%02d' "${i}").jsonl"
    if [ ! -f "${shard_path}" ]; then
        echo "[BUILD][ERROR] Missing extra MFA shard ${i}: ${shard_path}" >&2
        exit 1
    fi
    wc -l "${shard_path}"
done

echo "[BUILD][GSV2-FULL0_31] Assembling combined MFA dir: ${COMBINED_MFA_DIR}"
rm -rf "${COMBINED_MFA_DIR}"
mkdir -p "${COMBINED_MFA_DIR}"

for i in $(seq 0 $((PARTIAL_SHARDS - 1))); do
    src="${PARTIAL_MFA_DIR}/wiki_synth_train_shard_$(printf '%02d' "${i}").jsonl"
    dst="${COMBINED_MFA_DIR}/wiki_synth_train_shard_$(printf '%02d' "${i}").jsonl"
    ln -s "${src}" "${dst}"
done

for i in $(seq 0 $((EXTRA_SHARDS - 1))); do
    out_idx=$((PARTIAL_SHARDS + i))
    src="${EXTRA_MFA_DIR}/wiki_synth_train_shard_$(printf '%02d' "${i}").jsonl"
    dst="${COMBINED_MFA_DIR}/wiki_synth_train_shard_$(printf '%02d' "${out_idx}").jsonl"
    ln -s "${src}" "${dst}"
done

python "${BUILD_SCRIPT}" \
    --mfa-dir "${COMBINED_MFA_DIR}" \
    --gigaspeech-train "${GIGASPEECH_MFA_TRAIN}" \
    --gigaspeech-skip-wiki-synth \
    --num-shards "${NUM_SHARDS}" \
    --shard-pattern "${SHARD_PATTERN}" \
    --output-train "${OUTPUT_TRAIN}"

echo "[BUILD][GSV2-FULL0_31] Output: ${OUTPUT_TRAIN}"
echo "[BUILD][GSV2-FULL0_31] Stats:  ${OUTPUT_TRAIN%.jsonl}_stats.json"
