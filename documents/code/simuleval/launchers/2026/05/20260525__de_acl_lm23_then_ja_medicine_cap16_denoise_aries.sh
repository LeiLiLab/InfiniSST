#!/usr/bin/env bash
set -euo pipefail

# Aries orchestration:
# 1) DE cap16-denoise tagged ACL lm=2,3 same-LM batch.
# 2) After DE succeeds and JA local cache is ready, JA cap16-denoise medicine lm=1..4 same-LM batch.

ROOT_DIR="${ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

RUN_STAMP="${RUN_STAMP:-20260525T172158_deacl23_jamed1234_cap16denoise_aries}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/de_acl_then_ja_medicine_cap16_denoise_aries_${RUN_STAMP}}"
DE_LAUNCHER="${DE_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260525__tagged_acl_de_lm14_batch_chunks30.sh}"
JA_MED_LAUNCHER="${JA_MED_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260525__medicine_ja_cap16_denoise_lm1234_batch_aries.sh}"
JA_LOCAL_MODEL="${JA_LOCAL_MODEL:-/mnt/data3/jiaxuanluo/slm_local_cache/ja_tagged_acl_20260525/cap16_denoise_ttag/v2-20260525-235251-hf}"
JA_LOCAL_COMPLETE="${JA_LOCAL_MODEL}/.stage_complete"
WAIT_JA_MODEL_SECS="${WAIT_JA_MODEL_SECS:-21600}"
POLL_SECS="${POLL_SECS:-60}"

DE_MODEL_NAME="${DE_MODEL_NAME:-/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_de_cap16_denoise_budget_ttag_r32a32_ep1_taurus6/keep1.0_r32/v0-20260525-203735-hf}"
DE_OUT_ROOT="${DE_OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_de_cap16_denoise_lm23_batch_aries_${RUN_STAMP}}"
DE_LOG_ROOT="${DE_LOG_ROOT:-${LOG_ROOT}/de_acl_lm23}"
DE_EVAL_TMPDIR_ROOT="${DE_EVAL_TMPDIR_ROOT:-/tmp/jx_de23_${RUN_STAMP%%_*}}"

JA_OUT_ROOT="${JA_OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/medicine_hardraw_ja_cap16_denoise_lm1234_batch_aries_${RUN_STAMP}}"
JA_LOG_ROOT="${JA_LOG_ROOT:-${LOG_ROOT}/ja_medicine_lm1234}"
JA_EVAL_TMPDIR_ROOT="${JA_EVAL_TMPDIR_ROOT:-/tmp/jx_jamed_${RUN_STAMP%%_*}}"

mkdir -p "${LOG_ROOT}" "${DE_LOG_ROOT}" "${JA_LOG_ROOT}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

require_file() {
  local path="$1"
  [[ -s "${path}" ]] || {
    log "[ERROR] Missing/empty required file: ${path}" >&2
    exit 3
  }
}

wait_for_ja_model() {
  local waited=0
  while true; do
    if [[ -f "${JA_LOCAL_COMPLETE}" && -s "${JA_LOCAL_MODEL}/model.safetensors.index.json" ]]; then
      log "[READY] JA local model cache: ${JA_LOCAL_MODEL}"
      return 0
    fi
    if (( waited >= WAIT_JA_MODEL_SECS )); then
      log "[ERROR] Timed out waiting for JA local cache: ${JA_LOCAL_MODEL}" >&2
      exit 4
    fi
    log "[WAIT] JA local model cache not ready yet: ${JA_LOCAL_MODEL}"
    sleep "${POLL_SECS}"
    waited=$((waited + POLL_SECS))
  done
}

require_file "${DE_LAUNCHER}"
require_file "${JA_MED_LAUNCHER}"
require_file "${DE_MODEL_NAME}/config.json"
require_file "${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260525__tagged_acl_de_cap16_denoise_lm23_batch_aries.md"
require_file "${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260525__medicine_ja_cap16_denoise_lm1234_batch_aries.md"

{
  echo "run_stamp=${RUN_STAMP}"
  echo "host=$(hostname -s)"
  echo "de_launcher=${DE_LAUNCHER}"
  echo "ja_medicine_launcher=${JA_MED_LAUNCHER}"
  echo "de_model=${DE_MODEL_NAME}"
  echo "ja_local_model=${JA_LOCAL_MODEL}"
  echo "de_out_root=${DE_OUT_ROOT}"
  echo "ja_out_root=${JA_OUT_ROOT}"
} | tee "${LOG_ROOT}/orchestrator_meta.txt"

log "[STEP] DE tagged ACL lm=2,3 batch"
RUN_STAMP="${RUN_STAMP}_de_acl_lm23" \
LMS="2 3" \
GPU_PAIRS_CSV="4,5;6,7" \
MODEL_NAME="${DE_MODEL_NAME}" \
MODEL_LABEL="de_cap16_denoise_ttag_hn1024_tau078_omit_batch_chunks30_lm23_aries" \
DENSITY_TAG="tagacl_bv1_decap16den_ttag_hn1024_tau078_omit_chunks30_lm23_aries" \
TRAIN_EVENT_ID="20260525T1236__speech_llm_train__de_cap16_denoise_budget_ttag_r32a32_ep1_taurus6" \
STRIP_OUTPUT_TAGS="term_t" \
EMPTY_TERM_MAP_POLICY="omit" \
RAG_PROMPT_POLICY="given_chunks" \
RAG_TOP_K="10" \
MAX_CACHE_CHUNKS="30" \
KEEP_CACHE_CHUNKS="30" \
OUT_ROOT="${DE_OUT_ROOT}" \
LOG_ROOT="${DE_LOG_ROOT}" \
EVAL_TMPDIR_ROOT="${DE_EVAL_TMPDIR_ROOT}" \
NOTES_FILE="${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260525__tagged_acl_de_cap16_denoise_lm23_batch_aries.md" \
bash "${DE_LAUNCHER}" \
  > "${DE_LOG_ROOT}/top.out" \
  2> "${DE_LOG_ROOT}/top.err"

log "[STEP] Waiting for JA local cache"
wait_for_ja_model

log "[STEP] JA medicine hardraw lm=1,2,3,4 batch"
RUN_STAMP="${RUN_STAMP}_ja_medicine_lm1234" \
LANG_CODE="ja" \
LMS="1 2 3 4" \
GPU_PAIRS_CSV="4,5;6,7" \
MODEL_NAME="${JA_LOCAL_MODEL}" \
OUTPUT_BASE="${JA_OUT_ROOT}" \
LOG_ROOT="${JA_LOG_ROOT}" \
EVAL_TMPDIR_ROOT="${JA_EVAL_TMPDIR_ROOT}" \
MODEL_LABEL="ja_cap16_denoise_ttag_hn1024_tau078_omit_batch_chunks30" \
TRAIN_EVENT_ID="20260525T1550__speech_llm_train__ja_cap16_denoise_budget_ttag_r32a32_ep1_taurus4_retry1" \
bash "${JA_MED_LAUNCHER}" \
  > "${JA_LOG_ROOT}/top.out" \
  2> "${JA_LOG_ROOT}/top.err"

date -u +%Y-%m-%dT%H:%M:%SZ > "${LOG_ROOT}/.success"
log "[ALL DONE] de_summary=${DE_OUT_ROOT}/de_cap16_denoise_ttag_hn1024_tau078_omit_batch_chunks30_lm23_aries/__summary__/summary_de_lm2_lm3.tsv"
log "[ALL DONE] ja_summary=${JA_OUT_ROOT}/__summary__/summary_medicine_hardraw_ja_lm1_lm2_lm3_lm4.tsv"
