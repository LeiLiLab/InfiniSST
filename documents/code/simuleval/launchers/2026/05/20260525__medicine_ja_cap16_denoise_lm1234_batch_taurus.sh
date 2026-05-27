#!/usr/bin/env bash
set -euo pipefail

# Taurus wrapper for JA cap16-denoise same-LM medicine hardraw batch-vLLM readout.

ROOT_DIR="${ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260525__medicine_ja_cap16_denoise_lm1234_batch_aries.sh}"

RUN_STAMP="${RUN_STAMP:-20260525T184043_ja_med_cap16den_lm1234_taurus}"
RUN_STAMP_SHORT="${RUN_STAMP%%_*}"

MODEL_NAME="${MODEL_NAME:-/mnt/data1/jiaxuanluo/slm_local_cache/ja_tagged_acl_20260525/cap16_denoise_ttag/v2-20260525-235251-hf}"
MODEL_LABEL="${MODEL_LABEL:-ja_cap16_denoise_ttag_hn1024_tau078_omit_chunks30_maxnew40lm}"
TRAIN_EVENT_ID="${TRAIN_EVENT_ID:-20260525T1550__speech_llm_train__ja_cap16_denoise_budget_ttag_r32a32_ep1_taurus4_retry1}"

OUTPUT_BASE="${OUTPUT_BASE:-/mnt/gemini/data1/jiaxuanluo/medicine_hardraw_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/medicine_hardraw_${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/medicine_hardraw_${RUN_STAMP_SHORT}}"
EVAL_TMPDIR_ROOT="${EVAL_TMPDIR_ROOT:-/tmp/jx_jmed_${RUN_STAMP_SHORT}}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260525__medicine_ja_cap16_denoise_lm1234_batch_taurus.md}"

GPU_PAIRS_CSV="${GPU_PAIRS_CSV:-2,3;4,5}"
LMS="${LMS:-1 2 3 4}"
MAX_CACHE_CHUNKS="${MAX_CACHE_CHUNKS:-30}"
KEEP_CACHE_CHUNKS="${KEEP_CACHE_CHUNKS:-30}"
MAX_NEW_TOKENS_PER_LM="${MAX_NEW_TOKENS_PER_LM:-40}"
RAG_PROMPT_POLICY="${RAG_PROMPT_POLICY:-given_chunks}"

export ROOT_DIR
export RUN_STAMP
export MODEL_NAME
export MODEL_LABEL
export TRAIN_EVENT_ID
export OUTPUT_BASE
export LOG_ROOT
export INDEX_CACHE_DIR
export EVAL_TMPDIR_ROOT
export NOTES_FILE
export GPU_PAIRS_CSV
export LMS
export MAX_CACHE_CHUNKS
export KEEP_CACHE_CHUNKS
export MAX_NEW_TOKENS_PER_LM
export RAG_PROMPT_POLICY

exec bash "${BASE_LAUNCHER}"
