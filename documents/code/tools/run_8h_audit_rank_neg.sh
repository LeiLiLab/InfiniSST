#!/usr/bin/env bash
set -euo pipefail

# 8h unattended orchestrator: data audit + rank ablation (+ optional
# LLM-neg feasibility probe).  Every stage writes into a single run dir so
# you can come back and read STATE + logs.
#
# Usage:
#   nohup bash documents/code/tools/run_8h_audit_rank_neg.sh > /dev/null 2>&1 &
#
# The orchestrator is idempotent at the stage level: if a stage's completion
# marker exists, it skips that stage.  This lets you re-launch after an
# interruption without redoing expensive training runs.

# ======Configuration=====
ROOT_DIR="/home/jiaxuanluo/InfiniSST"
TOOLS_DIR="${ROOT_DIR}/documents/code/tools"
SST_EVAL_DIR="${ROOT_DIR}/documents/code/simuleval"
SST_TRAIN_DIR="${ROOT_DIR}/documents/code/train/sst_omni_train"

TS="$(date +%Y%m%d_%H%M%S)"
# Resume-friendly: accept an explicit RUN_ROOT via env (RUN_ROOT_OVERRIDE) so a crashed
# orchestrator can be restarted without losing .done / pid / job_id state files.
RUN_ROOT="${RUN_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs/run_8h_audit_rank_neg_${TS}}"

# Inputs
UPSTREAM_JSONL="/mnt/gemini/data1/jiaxuanluo/train_cleaned_with_retriever_results_varlen.jsonl"
TRAIN_D5_JSONL="/mnt/gemini/data1/jiaxuanluo/density_ablation/train_maxsim_varlen_d5.jsonl"
TRAIN_D5CAP_JSONL="/mnt/gemini/data1/jiaxuanluo/density_ablation/train_maxsim_varlen_d5_cap.jsonl"

OLD_SLM_MODEL="/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4"
D5_NOCAP_R16_HF="/mnt/aries/data4/jiaxuanluo/speech_llm_density_ablation/d5/r16/v0-20260414-010020-hf"
D5_CAP_R16_HF="/mnt/taurus/data2/jiaxuanluo/speech_llm_density_ablation/d5_cap/r16/v0-20260418-125814-hf"

# Training knobs
T1_LORA_RANK=32
T1_LORA_ALPHA=32
T2_LORA_RANK=64
T2_LORA_ALPHA=32

T1_SAVE_BASE="/mnt/taurus/data2/jiaxuanluo/8h_audit_rank_neg/train_outputs/d5_r32"
T2_SAVE_BASE="/mnt/taurus/data2/jiaxuanluo/8h_audit_rank_neg/train_outputs/d5_r64"
T1_DENSITY_ARG="5_r32"
T2_DENSITY_ARG="5_r64"

# Eval output
EVAL_OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/density_eval_rank_ablation"
EVAL_GPUS="7,5,6"
EVAL_LM="1"
EVAL_K="10"
PAPER_ID="2022.acl-long.110"
GLOSSARY_TAG="extracted_glossary__${PAPER_ID}"

# Audit knobs
AUDIT_WAV_SAMPLE_ROWS=2000
AUDIT_WHISPER_SAMPLES=0   # whisper spot-check skipped by default (GPU time is precious)

# Time limits (safety: kill stuck waits)
TIMEOUT_OLD_SLM_EVAL_SEC=2700    # 45 min
TIMEOUT_MODEL_EVAL_SEC=2700       # 45 min
TIMEOUT_TRAIN_SEC=16200           # 4.5h; includes queue + export
TIMEOUT_AUDIT_SEC=1800            # 30 min per audit

# spaCyEnv bin path (eval needs simuleval/sacrebleu)
SPACY_ENV_BIN="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin"
# ======Configuration=====


mkdir -p "${RUN_ROOT}"
STATE_DIR="${RUN_ROOT}/state"
LOG_DIR="${RUN_ROOT}/logs"
AUDIT_DIR="${RUN_ROOT}/audit"
PHASE1_DIR="${RUN_ROOT}/phase1"
REPORT_DIR="${RUN_ROOT}/report"
mkdir -p "${STATE_DIR}" "${LOG_DIR}" "${AUDIT_DIR}" "${PHASE1_DIR}" "${REPORT_DIR}"

MAIN_LOG="${LOG_DIR}/orchestrator.log"

log() {
  local msg="$*"
  local ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[${ts}] ${msg}" | tee -a "${MAIN_LOG}"
}

stage_done() { [[ -f "${STATE_DIR}/$1.done" ]]; }
stage_mark_done() { touch "${STATE_DIR}/$1.done"; log "STAGE DONE: $1"; }
stage_fail() {
  local name="$1"
  local reason="$2"
  echo "${reason}" > "${STATE_DIR}/${name}.failed"
  log "STAGE FAILED: ${name} -- ${reason}"
}

# --- helper: wait for a slurm job to finish (or hit timeout) ---
wait_for_slurm_job() {
  local job_id="$1"
  local label="$2"
  local timeout_sec="$3"
  local start=$(date +%s)
  while true; do
    local state
    state="$(squeue -j "${job_id}" -h -o '%T' 2>/dev/null || true)"
    if [[ -z "${state}" ]]; then
      # Not in squeue -> finished. Check sacct for final state.
      local final
      final="$(sacct -j "${job_id}" -n -X -o State 2>/dev/null | head -1 | awk '{print $1}')"
      log "${label}: slurm job ${job_id} left queue, final=${final:-unknown}"
      case "${final:-}" in
        COMPLETED) return 0 ;;
        FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_ME+|OUT_OF_MEMORY|PREEMPTED) return 1 ;;
        *) return 0 ;;  # treat unknown as OK (sacct may lag); downstream will verify artifacts
      esac
    fi
    local now=$(date +%s)
    local el=$((now - start))
    if (( el > timeout_sec )); then
      log "${label}: TIMEOUT waiting on job ${job_id} (${el}s > ${timeout_sec}s); cancelling"
      scancel "${job_id}" 2>/dev/null || true
      return 2
    fi
    sleep 30
  done
}

# --- helper: wait for a PID with timeout ---
wait_for_pid() {
  local pid="$1"
  local label="$2"
  local timeout_sec="$3"
  local start=$(date +%s)
  while kill -0 "${pid}" 2>/dev/null; do
    local now=$(date +%s)
    local el=$((now - start))
    if (( el > timeout_sec )); then
      log "${label}: TIMEOUT (${el}s > ${timeout_sec}s); killing pid=${pid}"
      kill -9 "${pid}" 2>/dev/null || true
      return 2
    fi
    sleep 10
  done
  wait "${pid}"
  local rc=$?
  log "${label}: pid=${pid} exited rc=${rc}"
  return "${rc}"
}

log "====================================================="
log "orchestrator started  RUN_ROOT=${RUN_ROOT}"
log "TS=${TS}  host=$(hostname)  user=$(whoami)"
log "====================================================="


# ============================================================
# STAGE: audit (CPU, parallel for two files)
# ============================================================
run_audit_one() {
  local label="$1"
  local in_jsonl="$2"
  local out_md="${AUDIT_DIR}/audit_${label}.md"
  local out_json="${AUDIT_DIR}/audit_${label}.json"
  if stage_done "audit_${label}"; then
    log "audit_${label}: skip (already done)"
    return 0
  fi
  log "audit_${label}: start"
  (
    python3 "${TOOLS_DIR}/audit_training_jsonl.py" \
      --input-jsonl "${in_jsonl}" \
      --output-md "${out_md}" \
      --summary-json "${out_json}" \
      --wav-sample-rows "${AUDIT_WAV_SAMPLE_ROWS}" \
      --whisper-samples "${AUDIT_WHISPER_SAMPLES}" \
      > "${LOG_DIR}/audit_${label}.log" 2>&1
  ) &
  local pid=$!
  echo "${pid}" > "${STATE_DIR}/audit_${label}.pid"
  echo "${pid}"
}

if ! stage_done audit_all; then
  AUDIT_NOCAP_PID="$(run_audit_one d5_nocap "${TRAIN_D5_JSONL}" | tail -1)"
  AUDIT_CAP_PID="$(run_audit_one d5_cap "${TRAIN_D5CAP_JSONL}" | tail -1)"
  log "audit pids: nocap=${AUDIT_NOCAP_PID} cap=${AUDIT_CAP_PID}"
fi


# ============================================================
# STAGE: old SLM paper-110 eval (GPU 7, bg, ~15-25min)
# ============================================================
if ! stage_done eval_old_slm && [[ -d "${OLD_SLM_MODEL}" ]]; then
  log "eval_old_slm: launching (bg)"
  (
    set -euo pipefail
    MODEL_TYPE="old_slm" \
    DENSITY_TAG="old_slm_pp110" \
    MODEL_NAME="${OLD_SLM_MODEL}" \
    OUTPUT_BASE="${EVAL_OUTPUT_BASE}" \
    GPUS="${EVAL_GPUS}" \
    LATENCY_MULTIPLIER="${EVAL_LM}" \
    bash "${SST_EVAL_DIR}/run_paper110_eval.sh"
  ) > "${LOG_DIR}/eval_old_slm.log" 2>&1 &
  echo "$!" > "${STATE_DIR}/eval_old_slm.pid"
  log "eval_old_slm: pid=$(cat "${STATE_DIR}/eval_old_slm.pid")"
fi


# ============================================================
# STAGE: d5 no-cap r=16 paper-110 eval (GPU 7, sequential AFTER old SLM)
# Produces the reference "r16" row the user explicitly asked for.
# ============================================================
# queued to run after eval_old_slm completes (sequential to avoid vLLM GPU clash)


# ============================================================
# Wait: audits + old SLM eval
# ============================================================
if ! stage_done audit_all; then
  log "waiting for audit_nocap (pid=${AUDIT_NOCAP_PID})..."
  if wait_for_pid "${AUDIT_NOCAP_PID}" "audit_nocap" "${TIMEOUT_AUDIT_SEC}"; then
    stage_mark_done audit_nocap
  else
    stage_fail audit_nocap "pid ${AUDIT_NOCAP_PID} non-zero exit"
  fi
  log "waiting for audit_cap (pid=${AUDIT_CAP_PID})..."
  if wait_for_pid "${AUDIT_CAP_PID}" "audit_cap" "${TIMEOUT_AUDIT_SEC}"; then
    stage_mark_done audit_cap
  else
    stage_fail audit_cap "pid ${AUDIT_CAP_PID} non-zero exit"
  fi
  stage_mark_done audit_all
fi

# Gate check: if either audit decided to gate, abort.
GATED_NOCAP=$(python3 -c "import json; d=json.load(open('${AUDIT_DIR}/audit_d5_nocap.json')); print(int(d.get('gated', True)))" 2>/dev/null || echo 1)
GATED_CAP=$(python3 -c "import json; d=json.load(open('${AUDIT_DIR}/audit_d5_cap.json')); print(int(d.get('gated', True)))" 2>/dev/null || echo 1)
log "audit gate: nocap=${GATED_NOCAP} cap=${GATED_CAP}"
if [[ "${GATED_NOCAP}" == "1" || "${GATED_CAP}" == "1" ]]; then
  log "AUDIT GATE TRIGGERED: one or both audits flagged blocking errors."
  log "Run continues to produce aggregate with only old-SLM + d5 r16 baselines (no new training)."
  AUDIT_GATED=1
else
  AUDIT_GATED=0
fi

if [[ -f "${STATE_DIR}/eval_old_slm.pid" ]] && ! stage_done eval_old_slm; then
  PID=$(cat "${STATE_DIR}/eval_old_slm.pid")
  log "waiting for eval_old_slm (pid=${PID})..."
  if wait_for_pid "${PID}" "eval_old_slm" "${TIMEOUT_OLD_SLM_EVAL_SEC}"; then
    stage_mark_done eval_old_slm
  else
    stage_fail eval_old_slm "non-zero exit"
  fi
fi


# ============================================================
# STAGE: d5 no-cap r=16 paper-110 eval (runs now that GPU 7 is free)
# ============================================================
if ! stage_done eval_d5_r16 && [[ -d "${D5_NOCAP_R16_HF}" ]]; then
  log "eval_d5_r16: launching (bg)"
  (
    set -euo pipefail
    MODEL_TYPE="speech_llm" \
    DENSITY_TAG="5_r16_pp110" \
    MODEL_NAME="${D5_NOCAP_R16_HF}" \
    OUTPUT_BASE="${EVAL_OUTPUT_BASE}" \
    GPUS="${EVAL_GPUS}" \
    LATENCY_MULTIPLIER="${EVAL_LM}" \
    bash "${SST_EVAL_DIR}/run_paper110_eval.sh"
  ) > "${LOG_DIR}/eval_d5_r16.log" 2>&1 &
  EVAL_R16_PID=$!
  echo "${EVAL_R16_PID}" > "${STATE_DIR}/eval_d5_r16.pid"
  log "eval_d5_r16: pid=${EVAL_R16_PID}"
fi


# ============================================================
# STAGE: submit T1 (r=32) training sbatch
# ============================================================
if [[ "${AUDIT_GATED}" == "0" ]] && ! stage_done t1_submit; then
  log "t1_submit: submitting r=${T1_LORA_RANK} alpha=${T1_LORA_ALPHA}"
  set +e
  T1_JOB_ID=$(
    RUN_TAG="rank_abl_${T1_DENSITY_ARG}" \
    DENSITY_ARG="${T1_DENSITY_ARG}" \
    DATASET_PATH="${TRAIN_D5_JSONL}" \
    SAVE_BASE="${T1_SAVE_BASE}" \
    LORA_RANK="${T1_LORA_RANK}" \
    LORA_ALPHA="${T1_LORA_ALPHA}" \
    MAX_LENGTH_OVERRIDE="${T1_MAX_LENGTH:-3072}" \
    PARTITION="taurus" \
    BASE_PORT="29561" \
    bash "${SST_TRAIN_DIR}/run_rank_ablation_sbatch.sh" 2>"${LOG_DIR}/t1_submit.err"
  )
  rc=$?
  set -e
  if [[ ${rc} -ne 0 || -z "${T1_JOB_ID}" ]]; then
    stage_fail t1_submit "submit returned rc=${rc}, job_id='${T1_JOB_ID}'"
    T1_JOB_ID=""
  else
    echo "${T1_JOB_ID}" > "${STATE_DIR}/t1_job_id.txt"
    stage_mark_done t1_submit
    log "t1_submit: job_id=${T1_JOB_ID}"
  fi
else
  T1_JOB_ID=""
  [[ -f "${STATE_DIR}/t1_job_id.txt" ]] && T1_JOB_ID="$(cat "${STATE_DIR}/t1_job_id.txt")"
fi


# Meanwhile, wait for eval_d5_r16 (GPU 7) before starting any more GPU 7 work.
if [[ -f "${STATE_DIR}/eval_d5_r16.pid" ]] && ! stage_done eval_d5_r16; then
  PID=$(cat "${STATE_DIR}/eval_d5_r16.pid")
  log "waiting for eval_d5_r16 (pid=${PID})..."
  if wait_for_pid "${PID}" "eval_d5_r16" "${TIMEOUT_MODEL_EVAL_SEC}"; then
    stage_mark_done eval_d5_r16
  else
    stage_fail eval_d5_r16 "non-zero exit"
  fi
fi


# ============================================================
# STAGE: Phase 1 feasibility (cache coverage, no probe by default)
# ============================================================
if ! stage_done phase1; then
  log "phase1: cache coverage check (no probe)"
  python3 "${TOOLS_DIR}/check_llm_neg_feasibility.py" \
    --output-md "${PHASE1_DIR}/neg_source_feasibility.md" \
    --output-json "${PHASE1_DIR}/neg_source_feasibility.json" \
    > "${LOG_DIR}/phase1.log" 2>&1 || true
  stage_mark_done phase1
fi


# ============================================================
# STAGE: wait for T1 training to finish
# ============================================================
if [[ -n "${T1_JOB_ID}" ]] && ! stage_done t1_train; then
  log "t1_train: waiting for job ${T1_JOB_ID}..."
  if wait_for_slurm_job "${T1_JOB_ID}" "t1_train" "${TIMEOUT_TRAIN_SEC}"; then
    stage_mark_done t1_train
  else
    stage_fail t1_train "job ${T1_JOB_ID} non-success"
  fi
fi


# ============================================================
# STAGE: locate T1 HF and eval on paper-110
# ============================================================
T1_HF=""
if stage_done t1_train && ! stage_done t1_eval; then
  # Find the -hf subdir produced by swift export.
  T1_HF="$(ls -1dt "${T1_SAVE_BASE}/r${T1_LORA_RANK}"/*-hf 2>/dev/null | head -1 || true)"
  if [[ -z "${T1_HF}" || ! -d "${T1_HF}" ]]; then
    stage_fail t1_eval "no -hf dir under ${T1_SAVE_BASE}/r${T1_LORA_RANK}/"
  else
    log "t1_eval: HF model at ${T1_HF}"
    echo "${T1_HF}" > "${STATE_DIR}/t1_hf.txt"
    set +e
    (
      set -euo pipefail
      MODEL_TYPE="speech_llm" \
      DENSITY_TAG="5_r${T1_LORA_RANK}_pp110" \
      MODEL_NAME="${T1_HF}" \
      OUTPUT_BASE="${EVAL_OUTPUT_BASE}" \
      GPUS="${EVAL_GPUS}" \
      LATENCY_MULTIPLIER="${EVAL_LM}" \
      bash "${SST_EVAL_DIR}/run_paper110_eval.sh"
    ) > "${LOG_DIR}/eval_t1.log" 2>&1
    rc=$?
    set -e
    if [[ ${rc} -eq 0 ]]; then
      stage_mark_done t1_eval
    else
      stage_fail t1_eval "paper110 eval rc=${rc}"
    fi
  fi
fi


# ============================================================
# STAGE: submit T2 (r=64, default) training
# ============================================================
if [[ "${AUDIT_GATED}" == "0" ]] && stage_done t1_train && ! stage_done t2_submit; then
  log "t2_submit: submitting r=${T2_LORA_RANK} alpha=${T2_LORA_ALPHA}"
  set +e
  T2_JOB_ID=$(
    RUN_TAG="rank_abl_${T2_DENSITY_ARG}" \
    DENSITY_ARG="${T2_DENSITY_ARG}" \
    DATASET_PATH="${TRAIN_D5_JSONL}" \
    SAVE_BASE="${T2_SAVE_BASE}" \
    LORA_RANK="${T2_LORA_RANK}" \
    LORA_ALPHA="${T2_LORA_ALPHA}" \
    MAX_LENGTH_OVERRIDE="${T2_MAX_LENGTH:-2560}" \
    PARTITION="taurus" \
    BASE_PORT="29562" \
    bash "${SST_TRAIN_DIR}/run_rank_ablation_sbatch.sh" 2>"${LOG_DIR}/t2_submit.err"
  )
  rc=$?
  set -e
  if [[ ${rc} -ne 0 || -z "${T2_JOB_ID}" ]]; then
    stage_fail t2_submit "submit rc=${rc} job_id='${T2_JOB_ID}'"
    T2_JOB_ID=""
  else
    echo "${T2_JOB_ID}" > "${STATE_DIR}/t2_job_id.txt"
    stage_mark_done t2_submit
    log "t2_submit: job_id=${T2_JOB_ID}"
  fi
else
  T2_JOB_ID=""
  [[ -f "${STATE_DIR}/t2_job_id.txt" ]] && T2_JOB_ID="$(cat "${STATE_DIR}/t2_job_id.txt")"
fi

if [[ -n "${T2_JOB_ID}" ]] && ! stage_done t2_train; then
  log "t2_train: waiting for job ${T2_JOB_ID}..."
  if wait_for_slurm_job "${T2_JOB_ID}" "t2_train" "${TIMEOUT_TRAIN_SEC}"; then
    stage_mark_done t2_train
  else
    stage_fail t2_train "job ${T2_JOB_ID} non-success"
  fi
fi

T2_HF=""
if stage_done t2_train && ! stage_done t2_eval; then
  T2_HF="$(ls -1dt "${T2_SAVE_BASE}/r${T2_LORA_RANK}"/*-hf 2>/dev/null | head -1 || true)"
  if [[ -z "${T2_HF}" || ! -d "${T2_HF}" ]]; then
    stage_fail t2_eval "no -hf dir under ${T2_SAVE_BASE}/r${T2_LORA_RANK}/"
  else
    log "t2_eval: HF model at ${T2_HF}"
    echo "${T2_HF}" > "${STATE_DIR}/t2_hf.txt"
    set +e
    (
      set -euo pipefail
      MODEL_TYPE="speech_llm" \
      DENSITY_TAG="5_r${T2_LORA_RANK}_pp110" \
      MODEL_NAME="${T2_HF}" \
      OUTPUT_BASE="${EVAL_OUTPUT_BASE}" \
      GPUS="${EVAL_GPUS}" \
      LATENCY_MULTIPLIER="${EVAL_LM}" \
      bash "${SST_EVAL_DIR}/run_paper110_eval.sh"
    ) > "${LOG_DIR}/eval_t2.log" 2>&1
    rc=$?
    set -e
    if [[ ${rc} -eq 0 ]]; then
      stage_mark_done t2_eval
    else
      stage_fail t2_eval "paper110 eval rc=${rc}"
    fi
  fi
fi


# ============================================================
# STAGE: aggregate report
# ============================================================
log "aggregate: building REPORT..."

# combined dir helpers
combined_dir_for_tag() {
  echo "${EVAL_OUTPUT_BASE}/zh/d${1}_lm${EVAL_LM}_k${EVAL_K}_per_paper_combined"
}

# For baselines we reuse cached 5-paper eval artifacts post-processed to
# paper-110-only metrics by compute_paper110_metrics_from_cache.py.  This
# avoids re-evaluating ~2h of GPU time for models that haven't changed.
CACHED_PP_ROOT="/mnt/taurus/data2/jiaxuanluo/8h_audit_rank_neg/cached_paper110"
OLD_SLM_PP_DIR="${CACHED_PP_ROOT}/old_slm"
D5_R16_PP_DIR="${CACHED_PP_ROOT}/d5_r16"

MODEL_ENTRIES=(
  "old_slm (cached pp110):${OLD_SLM_PP_DIR}"
  "d5_r16 (cached pp110):${D5_R16_PP_DIR}"
)
if stage_done t1_eval; then
  MODEL_ENTRIES+=("d5_r${T1_LORA_RANK}:$(combined_dir_for_tag 5_r${T1_LORA_RANK}_pp110)")
fi
if stage_done t2_eval; then
  MODEL_ENTRIES+=("d5_r${T2_LORA_RANK}:$(combined_dir_for_tag 5_r${T2_LORA_RANK}_pp110)")
fi

# Build --model-entry args safely (tags may contain spaces/parens).
AGG_ARGS=()
for e in "${MODEL_ENTRIES[@]}"; do
  AGG_ARGS+=(--model-entry "${e}")
done

python3 "${TOOLS_DIR}/aggregate_rank_ablation_report.py" \
  --audit-nocap-json "${AUDIT_DIR}/audit_d5_nocap.json" \
  --audit-cap-json "${AUDIT_DIR}/audit_d5_cap.json" \
  --phase1-json "${PHASE1_DIR}/neg_source_feasibility.json" \
  "${AGG_ARGS[@]}" \
  --output-md "${REPORT_DIR}/REPORT_audit_and_rank_ablation.md" \
  --output-json "${REPORT_DIR}/summary.json" \
  2>&1 | tee -a "${LOG_DIR}/aggregate.log" || true

# Publish the report to the canonical output path (non-timestamped) as well.
CANON_REPORT_DIR="${ROOT_DIR}/documents/data/phase456_orchestration"
mkdir -p "${CANON_REPORT_DIR}"
cp "${REPORT_DIR}/REPORT_audit_and_rank_ablation.md" \
   "${CANON_REPORT_DIR}/REPORT_audit_and_rank_ablation.md" 2>/dev/null || true

log "aggregate: DONE"
log "REPORT: ${REPORT_DIR}/REPORT_audit_and_rank_ablation.md"
log "        ${CANON_REPORT_DIR}/REPORT_audit_and_rank_ablation.md"
log "RUN_ROOT: ${RUN_ROOT}"
log "====================================================="
log "orchestrator finished"
