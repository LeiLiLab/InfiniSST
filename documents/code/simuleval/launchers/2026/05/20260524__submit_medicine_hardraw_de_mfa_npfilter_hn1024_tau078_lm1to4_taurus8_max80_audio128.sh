#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
BATCH_LAUNCHER="${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__medicine_hardraw_lm1to4_5samples_hn1024_tau078_new_v9_batch.sh"

RUN_STAMP="${RUN_STAMP:-20260524T182901_de_medhard_mfa_npfilter_max80_audio128}"
RUN_STAMP_SHORT="${RUN_STAMP%%_*}"
LANG_CODE="de"
TARGET_SAMPLES="${TARGET_SAMPLES_OVERRIDE:-404 545006 596001 605000 606}"
HARD_RAW_GLOSSARY="${HARD_RAW_GLOSSARY_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_hard_manual_glossary_streamlaal_20260524.json}"
MODEL_NAME="${MODEL_NAME_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_de_r32a64_tp2_taurus8/keep1.0_r32/v0-20260525-001708-hf}"
MODEL_LABEL="${MODEL_LABEL_OVERRIDE:-new_v9_mfa_npfilter_lexexact_oldnewv3_de_r32a64}"
HN1024_CKPT="${HN1024_CKPT_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"

OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_hardraw_de_mfa_npfilter_hn1024_tau078_lm1to4_5samples_max80_audio128_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs/medicine_hardraw_de_mfa_npfilter_hn1024_tau078_lm1to4_5samples_max80_audio128_${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/medicine_hardraw_de_mfa_npfilter_hn1024_tau078_${RUN_STAMP}}"
EVAL_TMPDIR="${EVAL_TMPDIR_OVERRIDE:-/tmp/jxmdde_${RUN_STAMP_SHORT}}"
DENSITY_TAG="${DENSITY_TAG_OVERRIDE:-medhard5_mfa_npfilter_de_hn1024_tau0p78_raw_max80_audio128}"
NOTES_FILE="${NOTES_FILE_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260524__medicine_hardraw_de_mfa_npfilter_hn1024_tau078_lm1to4_taurus8_max80_audio128.md}"

MAX_NEW_TOKENS_OVERRIDE="${MAX_NEW_TOKENS_OVERRIDE:-80}"
MAX_NEW_TOKENS_POLICY_OVERRIDE="${MAX_NEW_TOKENS_POLICY_OVERRIDE:-fixed}"
VLLM_LIMIT_AUDIO_OVERRIDE="${VLLM_LIMIT_AUDIO_OVERRIDE:-128}"
SCHEDULER_BATCH_SIZE_OVERRIDE="${SCHEDULER_BATCH_SIZE_OVERRIDE:-5}"
MAX_NUM_SEQS_OVERRIDE="${MAX_NUM_SEQS_OVERRIDE:-8}"

for p in "${ROOT_DIR}" "${BATCH_LAUNCHER}" "${MODEL_NAME}" "${HN1024_CKPT}" "${HARD_RAW_GLOSSARY}" "${NOTES_FILE}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

mkdir -p "${OUTPUT_BASE}" "${LOG_ROOT}" "${INDEX_CACHE_DIR}" "${EVAL_TMPDIR}"

common_env=(
  ROOT_DIR_OVERRIDE="${ROOT_DIR}"
  RUN_STAMP="${RUN_STAMP}"
  LANG_CODE_OVERRIDE="${LANG_CODE}"
  TARGET_SAMPLES_OVERRIDE="${TARGET_SAMPLES}"
  MODEL_NAME_OVERRIDE="${MODEL_NAME}"
  MODEL_LABEL="${MODEL_LABEL}"
  HARD_RAW_GLOSSARY_OVERRIDE="${HARD_RAW_GLOSSARY}"
  HN1024_CKPT_OVERRIDE="${HN1024_CKPT}"
  OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}"
  LOG_ROOT_OVERRIDE="${LOG_ROOT}"
  INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}"
  EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR}"
  DENSITY_TAG_OVERRIDE="${DENSITY_TAG}"
  NOTES_FILE_OVERRIDE="${NOTES_FILE}"
  RAG_SCORE_THRESHOLD_OVERRIDE="0.78"
  RAG_TOP_K_OVERRIDE="10"
  RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE="1.92"
  GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}"
  VLLM_TP_SIZE_OVERRIDE="${VLLM_TP_SIZE_OVERRIDE:-2}"
  MAX_NEW_TOKENS_OVERRIDE="${MAX_NEW_TOKENS_OVERRIDE}"
  MAX_NEW_TOKENS_POLICY_OVERRIDE="${MAX_NEW_TOKENS_POLICY_OVERRIDE}"
  VLLM_LIMIT_AUDIO_OVERRIDE="${VLLM_LIMIT_AUDIO_OVERRIDE}"
  SCHEDULER_BATCH_SIZE_OVERRIDE="${SCHEDULER_BATCH_SIZE_OVERRIDE}"
  MAX_NUM_SEQS_OVERRIDE="${MAX_NUM_SEQS_OVERRIDE}"
)

echo "[INFO] Preparing combined medicine inputs before concurrent LM launch"
env "${common_env[@]}" \
  TARGET_LMS_OVERRIDE="1" \
  GPU_PAIR="0,1" \
  RAG_GPU_OVERRIDE="cuda:1" \
  PREP_ONLY_OVERRIDE="1" \
  bash "${BATCH_LAUNCHER}" \
  > "${LOG_ROOT}/prepare.out" 2> "${LOG_ROOT}/prepare.err"

submit_lm() {
  local lm="$1"
  local gpu_pair="$2"
  local rag_gpu="$3"
  local out_log="${LOG_ROOT}/submit_lm${lm}.out"
  local err_log="${LOG_ROOT}/submit_lm${lm}.err"
  local pid_file="${LOG_ROOT}/submit_lm${lm}.pid"

  echo "[SUBMIT] lm=${lm} gpu_pair=${gpu_pair}"
  setsid bash -lc "
    cd '${ROOT_DIR}' &&
    env \
      ROOT_DIR_OVERRIDE='${ROOT_DIR}' \
      RUN_STAMP='${RUN_STAMP}' \
      LANG_CODE_OVERRIDE='${LANG_CODE}' \
      TARGET_SAMPLES_OVERRIDE='${TARGET_SAMPLES}' \
      TARGET_LMS_OVERRIDE='${lm}' \
      GPU_PAIR='${gpu_pair}' \
      RAG_GPU_OVERRIDE='${rag_gpu}' \
      MODEL_NAME_OVERRIDE='${MODEL_NAME}' \
      MODEL_LABEL='${MODEL_LABEL}' \
      HARD_RAW_GLOSSARY_OVERRIDE='${HARD_RAW_GLOSSARY}' \
      HN1024_CKPT_OVERRIDE='${HN1024_CKPT}' \
      OUTPUT_BASE_OVERRIDE='${OUTPUT_BASE}' \
      LOG_ROOT_OVERRIDE='${LOG_ROOT}' \
      INDEX_CACHE_DIR_OVERRIDE='${INDEX_CACHE_DIR}' \
      EVAL_TMPDIR_OVERRIDE='${EVAL_TMPDIR}' \
      DENSITY_TAG_OVERRIDE='${DENSITY_TAG}' \
      NOTES_FILE_OVERRIDE='${NOTES_FILE}' \
      RAG_SCORE_THRESHOLD_OVERRIDE='0.78' \
      RAG_TOP_K_OVERRIDE='10' \
      RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE='1.92' \
      GPU_MEMORY_UTILIZATION_OVERRIDE='${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}' \
      VLLM_TP_SIZE_OVERRIDE='${VLLM_TP_SIZE_OVERRIDE:-2}' \
      MAX_NEW_TOKENS_OVERRIDE='${MAX_NEW_TOKENS_OVERRIDE}' \
      MAX_NEW_TOKENS_POLICY_OVERRIDE='${MAX_NEW_TOKENS_POLICY_OVERRIDE}' \
      VLLM_LIMIT_AUDIO_OVERRIDE='${VLLM_LIMIT_AUDIO_OVERRIDE}' \
      SCHEDULER_BATCH_SIZE_OVERRIDE='${SCHEDULER_BATCH_SIZE_OVERRIDE}' \
      MAX_NUM_SEQS_OVERRIDE='${MAX_NUM_SEQS_OVERRIDE}' \
      SLURM_JOB_ID='mdde_mfa_lm${lm}' \
      bash '${BATCH_LAUNCHER}'
  " > "${out_log}" 2> "${err_log}" < /dev/null &
  echo $! > "${pid_file}"
  echo "[PID] lm=${lm} pid=$(cat "${pid_file}") out=${out_log} err=${err_log}"
}

submit_lm 1 "0,1" "cuda:1"
submit_lm 2 "2,3" "cuda:1"
submit_lm 3 "4,5" "cuda:1"
submit_lm 4 "6,7" "cuda:1"

echo "[ALL SUBMITTED] output=${OUTPUT_BASE} logs=${LOG_ROOT}"
