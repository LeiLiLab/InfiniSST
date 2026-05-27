#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
QUICK="${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260522__tagged_acl_speech_llm_quick_zh_lm2_raw_wait_hf.sh"
RUN_STAMP="${RUN_STAMP:-20260523T0042}"

MODEL_ROOT="${MODEL_ROOT:-/mnt/aries/data6/jiaxuanluo/slm/speech_llm_v16_llm_variant_aug_retriever_timeline_zh_lh1b88kw_tau073_r32a64_tp2_m4096_aries2/keep1.0_r32}"
MODEL_LABEL="${MODEL_LABEL:-v16_llmvariant_r32a64}"
GPU_PAIR="${GPU_PAIR:-2,3}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260523__tagged_acl_v16_llmvariant_r32_quick_zh_lm2_raw.md}"
RAW_DENOM="${RAW_DENOM:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"

# data7 is full on aries; keep this eval's generated outputs and maxsim cache on data6.
OUT_ROOT="${OUT_ROOT:-/mnt/aries/data6/jiaxuanluo/slm/tagged_acl_${MODEL_LABEL}_quick_zh_lm2_raw_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_${MODEL_LABEL}_quick_${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/aries/data6/jiaxuanluo/slm/maxsim_index_cache}"

for p in "${QUICK}" "${MODEL_ROOT}" "${NOTES_FILE}" "${RAW_DENOM}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

ROOT_DIR="${ROOT_DIR}" \
MODEL_ROOT="${MODEL_ROOT}" \
MODEL_LABEL="${MODEL_LABEL}" \
GPU_PAIR="${GPU_PAIR}" \
RUN_STAMP="${RUN_STAMP}" \
OUT_ROOT="${OUT_ROOT}" \
LOG_ROOT="${LOG_ROOT}" \
INDEX_CACHE_DIR="${INDEX_CACHE_DIR}" \
NOTES_FILE="${NOTES_FILE}" \
GLOSSARY_KINDS=raw \
EVAL_GLOSSARY_PATH_GLOBAL="${RAW_DENOM}" \
EVAL_GLOSSARY_FOLLOWS_KIND=0 \
WAIT_FOR_HF_SECS=0 \
bash "${QUICK}"
