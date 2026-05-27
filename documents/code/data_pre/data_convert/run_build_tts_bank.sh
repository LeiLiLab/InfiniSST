#!/bin/bash
set -euo pipefail

# ============================== Configuration ==============================
TERM_TRAIN_JSONL="/mnt/gemini/data/siqiouyang/term_train_dataset_final_with_tts.jsonl"
GLOSSARY_JSON="/mnt/gemini/data1/jiaxuanluo/glossary_for_zh_rate1.0_k20.json"

OUTPUT_DIR="/mnt/gemini/data/jiaxuanluo/tts_bank_from_term_train_v2"
OUTPUT_NPY="${OUTPUT_DIR}/terms.npy"
OUTPUT_WAV_DIR="${OUTPUT_DIR}/wav"

MAX_PROTOTYPES_PER_TERM=8
SEED=42
# ===========================================================================

mkdir -p "${OUTPUT_DIR}"

echo "[INFO] Building TTS bank from term_train_dataset_final.jsonl"
echo "[INFO]   JSONL:     ${TERM_TRAIN_JSONL}"
echo "[INFO]   Glossary:  ${GLOSSARY_JSON}"
echo "[INFO]   Output:    ${OUTPUT_NPY}  +  ${OUTPUT_WAV_DIR}/"
echo "[INFO]   Max prototypes/term: ${MAX_PROTOTYPES_PER_TERM}"

python /home/jiaxuanluo/InfiniSST/documents/code/data_pre/data_convert/build_tts_bank_from_term_train.py \
    --term-train-jsonl "${TERM_TRAIN_JSONL}" \
    --glossary-json "${GLOSSARY_JSON}" \
    --output-npy "${OUTPUT_NPY}" \
    --output-wav-dir "${OUTPUT_WAV_DIR}" \
    --max-prototypes-per-term "${MAX_PROTOTYPES_PER_TERM}" \
    --seed "${SEED}" \
    --skip-exist-check

echo "[INFO] TTS bank build complete."
echo "[INFO] To use in pipeline, set:"
echo "  RAG_TTS_TERMS_NPY_PATH=\"${OUTPUT_NPY}\""
echo "  RAG_TTS_WAV_DIR=\"${OUTPUT_WAV_DIR}\""
