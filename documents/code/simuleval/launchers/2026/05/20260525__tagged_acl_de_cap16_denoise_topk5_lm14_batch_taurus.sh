#!/usr/bin/env bash
set -euo pipefail

# En-De tagged ACL raw batch readout for latest cap16-denoise Speech LLM.
# This wrapper changes only retriever top-k from the default 10 to 5.

ROOT_DIR="${ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

RUN_STAMP="${RUN_STAMP:-20260525T1525_tagged_acl_de_cap16_denoise_topk5_lm14_batch_taurus}"
OUT_ROOT="${OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_de_cap16_denoise_topk5_lm14_batch_taurus_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_de_cap16_denoise_topk5_lm14_batch_taurus_${RUN_STAMP}}"
EVAL_TMPDIR_ROOT="${EVAL_TMPDIR_ROOT:-/tmp/jx_den5b30}"

RUN_STAMP="${RUN_STAMP}" \
ROOT_DIR="${ROOT_DIR}" \
LMS="1 4" \
GPU_PAIRS_CSV="${GPU_PAIRS_CSV:-4,5;6,7}" \
MODEL_NAME="/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_de_cap16_denoise_budget_ttag_r32a32_ep1_taurus6/keep1.0_r32/v0-20260525-203735-hf" \
MODEL_LABEL="de_cap16_denoise_ttag_hn1024_tau078_topk5_omit_batch_chunks30" \
DENSITY_TAG="tagacl_bv1_dedenoise_ttag_hn1024_tau078_topk5_omit_chunks30_batch" \
TRAIN_EVENT_ID="20260525T1236__speech_llm_train__de_cap16_denoise_budget_ttag_r32a32_ep1_taurus6" \
STRIP_OUTPUT_TAGS="term_t" \
EMPTY_TERM_MAP_POLICY="omit" \
RAG_PROMPT_POLICY="given_chunks" \
RAG_TOP_K="5" \
MAX_CACHE_CHUNKS="30" \
KEEP_CACHE_CHUNKS="30" \
OUT_ROOT="${OUT_ROOT}" \
LOG_ROOT="${LOG_ROOT}" \
EVAL_TMPDIR_ROOT="${EVAL_TMPDIR_ROOT}" \
bash documents/code/simuleval/launchers/2026/05/20260525__tagged_acl_de_lm14_batch_chunks30.sh
