#!/usr/bin/env bash
set -euo pipefail

# Build RAG indices for all per-paper extracted glossaries.
# Reads the manifest JSON from prepare_extracted_glossary_by_paper_inputs.py
# and builds an index for each paper.
#
# All user-facing strings are in English.

# ======Configuration=====
# Exit codes
EXIT_CONFIG_ERROR="2"
EXIT_DATA_ERROR="3"

# Repo
ROOT_DIR="/home/jiaxuanluo/InfiniSST"

# RAG model
RAG_MODEL_PATH="/mnt/gemini/data2/jiaxuanluo/final_main_result_model_v1.pt"
INDEX_CACHE_DIR="/mnt/gemini/data2/jiaxuanluo/index_cache_v4"
TARGET_LANG_CODE="zh"

# GPU selection (for building index, use a single GPU)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

# Per-paper glossaries (read directly from extract script output, not from paper_inputs_map.json)
EXTRACTED_GLOSSARIES_DIR="/home/jiaxuanluo/InfiniSST/documents/data/data_pre/extracted_glossaries_by_paper"
EXTRACTED_GLOSSARY_MANIFEST="/home/jiaxuanluo/InfiniSST/documents/data/data_pre/extracted_glossary_by_paper_manifest.json"

# Optional: specify which papers to build (space-separated paper_ids). Empty means all.
PAPER_IDS_OVERRIDE="${PAPER_IDS_OVERRIDE:-}"

# Build script
BUILD_INDEX_SCRIPT="${ROOT_DIR}/retriever/gigaspeech/run_build_index_v4.sh"
# ======Configuration=====

if [[ ! -f "${RAG_MODEL_PATH}" ]]; then
  echo "[ERROR] RAG_MODEL_PATH not found: ${RAG_MODEL_PATH}" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi
if [[ ! -d "${INDEX_CACHE_DIR}" ]]; then
  echo "[INFO] Creating INDEX_CACHE_DIR: ${INDEX_CACHE_DIR}"
  mkdir -p "${INDEX_CACHE_DIR}"
fi
if [[ ! -f "${EXTRACTED_GLOSSARY_MANIFEST}" ]]; then
  echo "[ERROR] Manifest JSON not found: ${EXTRACTED_GLOSSARY_MANIFEST}" >&2
  echo "[ERROR] Run extract_acl_terms_from_paper_v2.py first." >&2
  exit "${EXIT_DATA_ERROR}"
fi
if [[ ! -f "${BUILD_INDEX_SCRIPT}" ]]; then
  echo "[ERROR] Build index script not found: ${BUILD_INDEX_SCRIPT}" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi

PAPERS="$(
python3 - <<PY
import json
with open("${EXTRACTED_GLOSSARY_MANIFEST}",'r',encoding='utf-8') as f:
    obj=json.load(f)
papers=obj.get('papers',{})
print(" ".join(sorted(papers.keys())))
PY
)"

if [[ -z "${PAPERS}" ]]; then
  echo "[ERROR] No papers in manifest: ${EXTRACTED_GLOSSARY_MANIFEST}" >&2
  exit "${EXIT_DATA_ERROR}"
fi

if [[ -n "${PAPER_IDS_OVERRIDE}" ]]; then
  PAPERS="${PAPER_IDS_OVERRIDE}"
fi

echo "[INFO] Papers to build indices for: ${PAPERS}"
echo "[INFO] RAG_MODEL_PATH: ${RAG_MODEL_PATH}"
echo "[INFO] INDEX_CACHE_DIR: ${INDEX_CACHE_DIR}"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

export CUDA_VISIBLE_DEVICES

MODEL_TAG="$(basename "${RAG_MODEL_PATH}" .pt)"

for PAPER_ID in ${PAPERS}; do
  GLOSSARY_PATH="$(
  python3 - <<PY
import json
pid="${PAPER_ID}"
with open("${EXTRACTED_GLOSSARY_MANIFEST}",'r',encoding='utf-8') as f:
    obj=json.load(f)
papers=obj.get('papers',{})
item=papers.get(pid,{})
print(item.get('glossary_path',''))
PY
  )"

  if [[ -z "${GLOSSARY_PATH}" ]] || [[ ! -f "${GLOSSARY_PATH}" ]]; then
    echo "[WARN] Skip paper_id=${PAPER_ID}: glossary not found: ${GLOSSARY_PATH}" >&2
    continue
  fi

  GLOSSARY_TAG="$(basename "${GLOSSARY_PATH}" .json)"
  INDEX_OUTPUT_PATH="${INDEX_CACHE_DIR}/${MODEL_TAG}__${GLOSSARY_TAG}__tr16.pkl"

  if [[ -f "${INDEX_OUTPUT_PATH}" ]]; then
    echo "[INFO] Index already exists (skip): ${INDEX_OUTPUT_PATH}"
    continue
  fi

  echo "[INFO] ============================================================"
  echo "[INFO] Building index for paper_id=${PAPER_ID}"
  echo "[INFO] GLOSSARY_PATH=${GLOSSARY_PATH}"
  echo "[INFO] INDEX_OUTPUT_PATH=${INDEX_OUTPUT_PATH}"

  MODEL_PATH="${RAG_MODEL_PATH}" \
  GLOSSARY_PATH="${GLOSSARY_PATH}" \
  OUTPUT_PATH="${INDEX_OUTPUT_PATH}" \
  TARGET_LANG_CODE="${TARGET_LANG_CODE}" \
  bash "${BUILD_INDEX_SCRIPT}"

  echo "[INFO] ✓ Index built: ${INDEX_OUTPUT_PATH}"
done

echo "[INFO] Done. All per-paper indices built."

