#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
QUICK="${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260522__tagged_acl_speech_llm_quick_zh_lm2_raw_wait_hf.sh"
RUN_STAMP="${RUN_STAMP:-20260523T0214}"

MODEL_ROOT="${MODEL_ROOT:-/mnt/aries/data6/jiaxuanluo/slm/speech_llm_new_v5_no_gt_zero_llm_variant_aug_oldnewv3_zh_r32a64_tp2_aries2/keep1.0_r32}"
MODEL_LABEL="${MODEL_LABEL:-new_v5_no_gt_zero_oldnewv3_r32a64}"
GPU_PAIR="${GPU_PAIR:-6,7}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260523__tagged_acl_new_v5_no_gt_zero_oldnewv3_r32_gs1k_gs10k_zh_lm2.md}"
RAW_DENOM="${RAW_DENOM:-${RAW_GLOSSARY_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}}"
GS1K_GLOSSARY="${GS1K_GLOSSARY:-${GS1K_GLOSSARY_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_union_gs1000_min_norm2_backfill.json}}"
GS10K_GLOSSARY="${GS10K_GLOSSARY:-${GS10K_GLOSSARY_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_union_gs10000_min_norm2_backfill.json}}"
GLOSSARY_KINDS="${GLOSSARY_KINDS:-gs1k gs10k}"

OUT_ROOT="${OUT_ROOT:-/mnt/aries/data6/jiaxuanluo/slm/tagged_acl_${MODEL_LABEL}_quick_zh_lm2_gs1k_gs10k_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_${MODEL_LABEL}_quick_gs1k_gs10k_${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/aries/data6/jiaxuanluo/slm/maxsim_index_cache}"

for p in "${QUICK}" "${NOTES_FILE}" "${RAW_DENOM}" "${GS1K_GLOSSARY}" "${GS10K_GLOSSARY}"; do
  if [[ ! -f "${p}" ]]; then
    echo "[ERROR] Missing required file: ${p}" >&2
    exit 3
  fi
done

if [[ ! -d "${MODEL_ROOT}" ]]; then
  echo "[ERROR] Missing model root: ${MODEL_ROOT}" >&2
  echo "[HINT] Set MODEL_ROOT to a directory containing a complete '*-hf' export." >&2
  exit 3
fi

ROOT_DIR="${ROOT_DIR}" \
MODEL_ROOT="${MODEL_ROOT}" \
MODEL_LABEL="${MODEL_LABEL}" \
GPU_PAIR="${GPU_PAIR}" \
RUN_STAMP="${RUN_STAMP}" \
OUT_ROOT="${OUT_ROOT}" \
LOG_ROOT="${LOG_ROOT}" \
INDEX_CACHE_DIR="${INDEX_CACHE_DIR}" \
NOTES_FILE="${NOTES_FILE}" \
GLOSSARY_KINDS="${GLOSSARY_KINDS}" \
EVAL_GLOSSARY_PATH_GLOBAL="${RAW_DENOM}" \
RAW_GLOSSARY_OVERRIDE="${RAW_DENOM}" \
GS1K_GLOSSARY_OVERRIDE="${GS1K_GLOSSARY}" \
GS10K_GLOSSARY_OVERRIDE="${GS10K_GLOSSARY}" \
EVAL_GLOSSARY_FOLLOWS_KIND=0 \
EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR_OVERRIDE:-${TMPDIR:-}}" \
WAIT_FOR_HF_SECS="${WAIT_FOR_HF_SECS:-60}" \
bash "${QUICK}"
