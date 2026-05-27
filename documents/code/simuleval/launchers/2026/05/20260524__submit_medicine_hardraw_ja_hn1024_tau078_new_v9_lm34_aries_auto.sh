#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
BATCH_LAUNCHER="${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__medicine_hardraw_lm1to4_5samples_hn1024_tau078_new_v9_batch.sh"

RUN_STAMP="${RUN_STAMP:-20260524T0710_ja_hardraw_newv9_hn1024_tau078}"
RUN_STAMP_SHORT="${RUN_STAMP%%_*}"
LANG_CODE="ja"
TARGET_SAMPLES="${TARGET_SAMPLES_OVERRIDE:-404 545006 596001 605000 606}"
HARD_RAW_GLOSSARY="${HARD_RAW_GLOSSARY_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_hard_manual_glossary_streamlaal_20260524.json}"
MODEL_NAME="${MODEL_NAME_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_ja_r32a64_tp2_taurus4/keep1.0_r32/v0-20260524-123624-hf}"
MODEL_LABEL="${MODEL_LABEL_OVERRIDE:-newv9_ja_r32a64}"
HN1024_CKPT="${HN1024_CKPT_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
ESO_TEST_ROOT="${ESO_TEST_ROOT_OVERRIDE:-/mnt/taurus/home/jiaxingxu/rag-sst/eso-dataset/outputs_v2/test}"

OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_hardraw_ja_hn1024_tau078_new_v9_lm12_5samples_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs/medicine_hardraw_ja_hn1024_tau078_new_v9_lm12_5samples_${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/medicine_hardraw_ja_hn1024_tau078_new_v9_${RUN_STAMP}}"
EVAL_TMPDIR="${EVAL_TMPDIR_OVERRIDE:-/tmp/jxmdja_${RUN_STAMP_SHORT}}"
DENSITY_TAG="${DENSITY_TAG_OVERRIDE:-medhard5_newv9_ja_hn1024_tau0p78_raw}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}"
VLLM_TP_SIZE="${VLLM_TP_SIZE_OVERRIDE:-2}"

for p in "${ROOT_DIR}" "${BATCH_LAUNCHER}" "${MODEL_NAME}" "${HN1024_CKPT}" "${HARD_RAW_GLOSSARY}" "${ESO_TEST_ROOT}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

mkdir -p "${OUTPUT_BASE}" "${LOG_ROOT}" "${INDEX_CACHE_DIR}" "${EVAL_TMPDIR}"

submit_lm() {
  local lm="$1"
  local gpu_pair="$2"
  local rag_gpu="${3:-cuda:1}"
  local out_log="${LOG_ROOT}/submit_lm${lm}.out"
  local err_log="${LOG_ROOT}/submit_lm${lm}.err"
  local pid_file="${LOG_ROOT}/submit_lm${lm}.pid"

  if [[ -s "${pid_file}" ]] && ps -p "$(cat "${pid_file}")" >/dev/null 2>&1; then
    echo "[SKIP] lm=${lm} already running pid=$(cat "${pid_file}")"
    return 0
  fi

  echo "[SUBMIT] lm=${lm} gpu_pair=${gpu_pair}"
  setsid bash -lc "
    cd '${ROOT_DIR}' &&
    env \
      ROOT_DIR_OVERRIDE='${ROOT_DIR}' \
      ESO_TEST_ROOT_OVERRIDE='${ESO_TEST_ROOT}' \
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
      RAG_SCORE_THRESHOLD_OVERRIDE='0.78' \
      RAG_TOP_K_OVERRIDE='10' \
      RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE='1.92' \
      GPU_MEMORY_UTILIZATION_OVERRIDE='${GPU_MEMORY_UTILIZATION}' \
      VLLM_TP_SIZE_OVERRIDE='${VLLM_TP_SIZE}' \
      SLURM_JOB_ID='mdja_lm${lm}' \
      bash '${BATCH_LAUNCHER}'
  " > "${out_log}" 2> "${err_log}" < /dev/null &
  echo $! > "${pid_file}"
  echo "[PID] lm=${lm} pid=$(cat "${pid_file}") out=${out_log} err=${err_log}"
}

find_free_pair() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, '$2 + 0 < 1000 {gsub(/ /, "", $1); print $1}' \
    | head -n 2 \
    | paste -sd, -
}

if [[ "${1:-}" == "SUBMIT_LM4_ONLY" ]]; then
  submit_lm 4 "${2:?gpu pair required}" "cuda:1"
  exit 0
fi

submit_lm 3 "${GPU_PAIR_LM3_OVERRIDE:-6,7}" "${RAG_GPU_LM3_OVERRIDE:-cuda:1}"

setsid bash -lc "
  set -euo pipefail
  mkdir -p '${LOG_ROOT}'
  sleep 300
  while true; do
    if [[ -s '${LOG_ROOT}/submit_lm4.pid' ]] && ps -p \"\$(cat '${LOG_ROOT}/submit_lm4.pid')\" >/dev/null 2>&1; then
      echo \"[\$(date -u +%Y-%m-%dT%H:%M:%SZ)] lm4 already running pid=\$(cat '${LOG_ROOT}/submit_lm4.pid')\"
      exit 0
    fi
    busy_regex='^$'
    for spec in '1:2,3' '2:4,5' '3:6,7'; do
      lm=\${spec%%:*}
      gpus=\${spec#*:}
      if [[ -s '${LOG_ROOT}/submit_lm'\${lm}'.pid' ]] && ps -p \"\$(cat '${LOG_ROOT}/submit_lm'\${lm}'.pid')\" >/dev/null 2>&1; then
        busy_regex=\"\${busy_regex}|\$(printf '%s' \"\${gpus}\" | sed 's/,/|/g')\"
      fi
    done
    pair=\$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v busy=\"\${busy_regex}\" '\$2 + 0 < 1000 {gsub(/ /, \"\", \$1); if (\$1 !~ busy) print \$1}' | head -n 2 | paste -sd, -)
    echo \"[\$(date -u +%Y-%m-%dT%H:%M:%SZ)] free_pair=\${pair:-none}\"
    if [[ \"\${pair}\" == *,* ]]; then
      '${BASH_SOURCE[0]}' SUBMIT_LM4_ONLY \"\${pair}\"
      exit 0
    fi
    sleep 300
  done
" > "${LOG_ROOT}/lm4_waiter.out" 2> "${LOG_ROOT}/lm4_waiter.err" < /dev/null &
echo $! > "${LOG_ROOT}/lm4_waiter.pid"
echo "[WAITER] lm4 waiter pid=$(cat "${LOG_ROOT}/lm4_waiter.pid")"

echo "[ALL SUBMITTED] lm3 running; lm4 waiter active. output=${OUTPUT_BASE} logs=${LOG_ROOT}"
