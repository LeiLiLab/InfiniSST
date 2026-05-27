#!/usr/bin/env bash
set -uo pipefail
# NOTE: we intentionally do NOT set -e. This script runs unattended overnight
# and must NEVER abort on a recoverable sub-step; instead we check explicit
# rc/file guards, log, and either retry or stop gracefully while still writing
# a final report.

# Overnight orchestrator for Phase 5 (eval + compare + decision gate) and the
# conditional Phase 6 (sub-problem B fix: shorten no-GT term_map + retrain).
#
# Starting state assumed when this script is launched:
#   * Control HF model exists at CONTROL_HF.
#   * Control eval may already be running (OK; we detect via the TSV output).
#   * Experiment training (slurm job EXPERIMENT_JOB_ID) is running.
#
# Outputs:
#   * Per-step logs under ORCH_DIR/logs/
#   * Final report at ORCH_DIR/REPORT.md
#   * Decision JSON at ORCH_DIR/phase5_decision.json
#
# All user-facing strings are in English.

# ======Configuration=====
ROOT_DIR="/home/jiaxuanluo/InfiniSST"
ORCH_DIR="/home/jiaxuanluo/InfiniSST/documents/data/phase456_orchestration"
LOG_DIR="${ORCH_DIR}/logs"
REPORT="${ORCH_DIR}/REPORT.md"
DECISION_JSON="${ORCH_DIR}/phase5_decision.json"

# Phase 4 model outputs (HF exports). Experiment HF is written post-train by
# run_speech_llm_4gpu_maxsim.sh; we wait for *-hf/config.json to exist.
CONTROL_HF="/mnt/taurus/data2/jiaxuanluo/speech_llm_density_ablation/d5_cap/r16/v0-20260418-125814-hf"
EXPERIMENT_SAVE_ROOT="/mnt/taurus/data2/jiaxuanluo/speech_llm_density_ablation/d5_cap_adv/r16"
EXPERIMENT_JOB_ID="43715"

# Eval output config (must match run_phase5_model_eval.sh defaults)
EVAL_OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/density_eval_maxsim_fixed"
EVAL_RAG_TOP_K="10"
EVAL_LM="1"
CONTROL_TAG="5_cap"
EXPERIMENT_TAG="5_cap_adv"
CONTROL_BP_TSV="${EVAL_OUTPUT_BASE}/zh/d${CONTROL_TAG}_lm${EVAL_LM}_k${EVAL_RAG_TOP_K}_per_paper_combined/eval_results_by_paper.tsv"
EXPERIMENT_BP_TSV="${EVAL_OUTPUT_BASE}/zh/d${EXPERIMENT_TAG}_lm${EVAL_LM}_k${EVAL_RAG_TOP_K}_per_paper_combined/eval_results_by_paper.tsv"

# Phase 6 config (conditional)
PHASE6_TAG="5_cap_adv_B"
PHASE6_SAVE_BASE="/mnt/taurus/data2/jiaxuanluo/speech_llm_density_ablation/d5_cap_adv_B"
PHASE6_DATASET="/mnt/gemini/data1/jiaxuanluo/adversarial/train_maxsim_varlen_d5_cap_adv_B.jsonl"
PHASE6_SOURCE_JSONL="/mnt/gemini/data1/jiaxuanluo/adversarial/train_cleaned_with_retriever_results_varlen_adv.jsonl"
PHASE6_NO_GT_MAX_TERMS="5"      # shorten no-GT term_map to 5 (was effectively up to 10 via density*multiplier capped at 20)
PHASE6_EMPTY_PROB_NO_GT="0.70"  # raise from 0.50 -> 0.70
PHASE6_BP_TSV="${EVAL_OUTPUT_BASE}/zh/d${PHASE6_TAG}_lm${EVAL_LM}_k${EVAL_RAG_TOP_K}_per_paper_combined/eval_results_by_paper.tsv"

# Decision gate threshold (experiment TERM_FCR)
TERM_FCR_THRESHOLD="0.05"

# Max wall times (defensive guards so the orchestrator never hangs forever)
WAIT_EXPERIMENT_HF_MAX_SEC=$((4*3600))        # experiment training + HF export: cap 4h
WAIT_EVAL_MAX_SEC=$((3*3600))                 # per-model eval: cap 3h
WAIT_PHASE6_TRAIN_MAX_SEC=$((4*3600))         # phase 6 training + export: cap 4h

POLL_INTERVAL_SEC="60"
# ======Configuration=====

mkdir -p "${LOG_DIR}"

log() {
  local msg="$1"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${msg}" | tee -a "${LOG_DIR}/orchestrator.log"
}

abort_with_report() {
  local reason="$1"
  log "ABORT: ${reason}"
  write_final_report "ABORTED: ${reason}"
  exit 0  # graceful exit so nohup cleanup is clean
}

# ---- Helpers ----------------------------------------------------------------

wait_for_file() {
  # wait_for_file <path> <max_sec> <poll_sec> <tag>
  local path="$1"
  local max_sec="$2"
  local poll="$3"
  local tag="$4"
  local waited=0
  log "wait_for_file(${tag}): ${path} (max=${max_sec}s)"
  while [[ ! -e "${path}" ]]; do
    sleep "${poll}"
    waited=$((waited + poll))
    if [[ ${waited} -ge ${max_sec} ]]; then
      log "wait_for_file(${tag}): TIMEOUT after ${waited}s"
      return 1
    fi
    if (( waited % 300 == 0 )); then
      log "wait_for_file(${tag}): still waiting (${waited}s / ${max_sec}s)"
    fi
  done
  log "wait_for_file(${tag}): found after ${waited}s -> ${path}"
  return 0
}

wait_for_slurm_job() {
  # wait_for_slurm_job <jobid> <max_sec> <tag>
  local jobid="$1"
  local max_sec="$2"
  local tag="$3"
  local waited=0
  log "wait_for_slurm_job(${tag}): jobid=${jobid} (max=${max_sec}s)"
  while squeue -j "${jobid}" -h -o "%T" 2>/dev/null | grep -qE 'RUNNING|PENDING|CONFIGURING|COMPLETING'; do
    sleep "${POLL_INTERVAL_SEC}"
    waited=$((waited + POLL_INTERVAL_SEC))
    if [[ ${waited} -ge ${max_sec} ]]; then
      log "wait_for_slurm_job(${tag}): TIMEOUT after ${waited}s"
      return 1
    fi
    if (( waited % 300 == 0 )); then
      log "wait_for_slurm_job(${tag}): still running (${waited}s)"
    fi
  done
  local state
  state="$(sacct -j "${jobid}" -X --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')"
  log "wait_for_slurm_job(${tag}): finished state=${state}"
  [[ "${state}" == "COMPLETED" ]] && return 0 || return 2
}

find_latest_hf_dir() {
  # Echo the newest *-hf subdir under a save_root, or empty if none.
  local save_root="$1"
  ls -1dt "${save_root}"/v*-hf 2>/dev/null | head -n 1 || true
}

wait_for_hf_export() {
  # wait_for_hf_export <save_root> <max_sec> <tag>
  # Polls until some <save_root>/v*-hf/config.json exists.
  local save_root="$1"
  local max_sec="$2"
  local tag="$3"
  local waited=0
  log "wait_for_hf_export(${tag}): root=${save_root} (max=${max_sec}s)"
  while true; do
    local hf
    hf="$(find_latest_hf_dir "${save_root}")"
    if [[ -n "${hf}" && -f "${hf}/config.json" ]]; then
      log "wait_for_hf_export(${tag}): found HF -> ${hf}"
      echo "${hf}"
      return 0
    fi
    sleep "${POLL_INTERVAL_SEC}"
    waited=$((waited + POLL_INTERVAL_SEC))
    if [[ ${waited} -ge ${max_sec} ]]; then
      log "wait_for_hf_export(${tag}): TIMEOUT after ${waited}s"
      return 1
    fi
    if (( waited % 300 == 0 )); then
      log "wait_for_hf_export(${tag}): still waiting (${waited}s)"
    fi
  done
}

# Run a per-model eval, blocking. Reuses run_phase5_model_eval.sh.
run_eval_blocking() {
  # run_eval_blocking <density_tag> <hf_model_path> <expected_tsv> <log_name>
  local density_tag="$1"
  local hf="$2"
  local expected_tsv="$3"
  local log_name="$4"

  if [[ -s "${expected_tsv}" ]]; then
    log "run_eval_blocking(${density_tag}): TSV already exists, skipping eval: ${expected_tsv}"
    return 0
  fi

  log "run_eval_blocking(${density_tag}): starting (log=${LOG_DIR}/${log_name})"
  (
    export DENSITY_TAG="${density_tag}"
    export MODEL_NAME="${hf}"
    bash "${ROOT_DIR}/documents/code/simuleval/run_phase5_model_eval.sh"
  ) >"${LOG_DIR}/${log_name}" 2>&1
  local rc=$?
  log "run_eval_blocking(${density_tag}): rc=${rc}"

  if [[ ! -s "${expected_tsv}" ]]; then
    log "run_eval_blocking(${density_tag}): ERROR expected TSV missing after eval: ${expected_tsv}"
    return 2
  fi
  return 0
}

wait_for_bg_eval() {
  # wait_for_bg_eval <pid_file> <expected_tsv> <max_sec> <tag>
  local pid_file="$1"
  local expected_tsv="$2"
  local max_sec="$3"
  local tag="$4"
  local waited=0
  log "wait_for_bg_eval(${tag}): pid_file=${pid_file} tsv=${expected_tsv}"
  if [[ ! -f "${pid_file}" ]]; then
    log "wait_for_bg_eval(${tag}): no pid file; assuming detached eval; polling TSV only"
  fi
  while true; do
    if [[ -s "${expected_tsv}" ]]; then
      log "wait_for_bg_eval(${tag}): TSV ready after ${waited}s"
      return 0
    fi
    if [[ -f "${pid_file}" ]]; then
      local pid
      pid="$(cat "${pid_file}")"
      if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
        # PID exited but no TSV -> failure
        if [[ ! -s "${expected_tsv}" ]]; then
          log "wait_for_bg_eval(${tag}): background eval pid ${pid} exited but TSV missing; ERROR"
          return 2
        fi
      fi
    fi
    sleep "${POLL_INTERVAL_SEC}"
    waited=$((waited + POLL_INTERVAL_SEC))
    if [[ ${waited} -ge ${max_sec} ]]; then
      log "wait_for_bg_eval(${tag}): TIMEOUT after ${waited}s"
      return 1
    fi
    if (( waited % 300 == 0 )); then
      log "wait_for_bg_eval(${tag}): still waiting (${waited}s)"
    fi
  done
}

# ---- Report writer ----------------------------------------------------------

write_final_report() {
  local status="$1"
  {
    echo "# Phase 4/5/6 overnight orchestration report"
    echo ""
    echo "**Status**: ${status}"
    echo "**Generated**: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "**Orchestration dir**: \`${ORCH_DIR}\`"
    echo ""
    echo "## Phase 4 training"
    echo ""
    echo "- Control: \`${CONTROL_HF}\`"
    local exp_hf
    exp_hf="$(find_latest_hf_dir "${EXPERIMENT_SAVE_ROOT}")"
    if [[ -n "${exp_hf}" ]]; then
      echo "- Experiment: \`${exp_hf}\`"
    else
      echo "- Experiment: **not exported yet**"
    fi
    echo "- Experiment slurm job: \`${EXPERIMENT_JOB_ID}\`"
    echo ""

    echo "## Phase 5 eval (stride=window, k=10, lm=1, 5 papers)"
    echo ""
    for pair in "Control|${CONTROL_BP_TSV}" "Experiment|${EXPERIMENT_BP_TSV}"; do
      local name="${pair%%|*}"
      local tsv="${pair##*|}"
      echo "### ${name}"
      if [[ -s "${tsv}" ]]; then
        echo '```tsv'
        cat "${tsv}"
        echo '```'
      else
        echo "TSV not produced: \`${tsv}\`"
      fi
      echo ""
    done

    echo "## Phase 5 decision gate"
    echo ""
    if [[ -s "${DECISION_JSON}" ]]; then
      echo '```json'
      cat "${DECISION_JSON}"
      echo '```'
    else
      echo "Decision JSON not produced."
    fi
    echo ""

    if [[ -s "${PHASE6_BP_TSV}" ]]; then
      echo "## Phase 6 (sub-problem B fix) eval"
      echo ""
      echo '```tsv'
      cat "${PHASE6_BP_TSV}"
      echo '```'
      echo ""
    fi

    echo "## Logs"
    echo ""
    for f in "${LOG_DIR}"/*.log; do
      [[ -f "${f}" ]] || continue
      echo "- \`${f}\`"
    done
    echo ""
  } > "${REPORT}"
  log "Report written: ${REPORT}"
}

# ---- Orchestration steps ----------------------------------------------------

log "========================================================================"
log "Phase 4/5/6 orchestrator start"
log "  control_hf=${CONTROL_HF}"
log "  experiment_job=${EXPERIMENT_JOB_ID}"
log "  experiment_save_root=${EXPERIMENT_SAVE_ROOT}"
log "========================================================================"

# --- Step 1: wait for control eval to finish --------------------------------
# Phase 5 control eval was launched separately as a background process; its
# pid is tracked at ORCH_DIR/control_eval.pid and output TSV is CONTROL_BP_TSV.
log "Step 1: waiting for control eval TSV..."
if ! wait_for_bg_eval \
     "${ORCH_DIR}/control_eval.pid" \
     "${CONTROL_BP_TSV}" \
     "${WAIT_EVAL_MAX_SEC}" "control_eval"; then
  abort_with_report "Control eval did not produce TSV in time"
fi

# --- Step 2: wait for experiment training + HF export -----------------------
log "Step 2: waiting for experiment slurm job ${EXPERIMENT_JOB_ID}..."
wait_for_slurm_job "${EXPERIMENT_JOB_ID}" "${WAIT_EXPERIMENT_HF_MAX_SEC}" "experiment_train"
job_rc=$?
if [[ ${job_rc} -ne 0 ]]; then
  log "Experiment slurm job finished with non-COMPLETED state (rc=${job_rc}); will still check for HF export..."
fi

EXPERIMENT_HF="$(wait_for_hf_export "${EXPERIMENT_SAVE_ROOT}" "${WAIT_EXPERIMENT_HF_MAX_SEC}" "experiment_hf")"
if [[ -z "${EXPERIMENT_HF:-}" || ! -f "${EXPERIMENT_HF}/config.json" ]]; then
  abort_with_report "Experiment HF export not found under ${EXPERIMENT_SAVE_ROOT}"
fi
log "Experiment HF ready: ${EXPERIMENT_HF}"

# --- Step 3: experiment eval ------------------------------------------------
log "Step 3: running experiment eval..."
if ! run_eval_blocking \
     "${EXPERIMENT_TAG}" "${EXPERIMENT_HF}" \
     "${EXPERIMENT_BP_TSV}" "eval_experiment.log"; then
  abort_with_report "Experiment eval failed; see ${LOG_DIR}/eval_experiment.log"
fi

# --- Step 4: A/B compare + decision gate ------------------------------------
log "Step 4: A/B compare + decision gate"
python3 "${ROOT_DIR}/documents/code/offline_sst_eval/phase5_compare.py" \
  --control-tsv "${CONTROL_BP_TSV}" \
  --experiment-tsv "${EXPERIMENT_BP_TSV}" \
  --decision-json "${DECISION_JSON}" \
  --term-fcr-threshold "${TERM_FCR_THRESHOLD}" \
  2>&1 | tee "${LOG_DIR}/phase5_compare.log"
compare_rc="${PIPESTATUS[0]}"
if [[ ${compare_rc} -ne 0 || ! -s "${DECISION_JSON}" ]]; then
  abort_with_report "phase5_compare.py failed (rc=${compare_rc})"
fi

TRIGGER=$(python3 -c "
import json
with open('${DECISION_JSON}') as f:
    d = json.load(f)
print('yes' if d.get('trigger_phase6') else 'no')
")
log "Phase 6 trigger: ${TRIGGER}"

# --- Step 5 (conditional): Phase 6 rebuild + retrain + eval -----------------
if [[ "${TRIGGER}" != "yes" ]]; then
  log "Phase 6 skipped (gate not triggered)."
  write_final_report "SUCCESS (Phase 6 not triggered)"
  log "========================================================================"
  log "Orchestrator done."
  log "========================================================================"
  exit 0
fi

log "Phase 6 TRIGGERED. Rebuilding no-GT-shortened dataset..."

if [[ ! -s "${PHASE6_DATASET}" ]]; then
  if [[ ! -f "${PHASE6_SOURCE_JSONL}" ]]; then
    abort_with_report "Phase 6 source JSONL missing: ${PHASE6_SOURCE_JSONL}"
  fi
  python3 "${ROOT_DIR}/documents/code/data_pre/hard_negative_jsonl_for_speech_llm/rebuild_termmap.py" \
    --input_jsonl "${PHASE6_SOURCE_JSONL}" \
    --output_jsonl "${PHASE6_DATASET}" \
    --density_coeff 5 \
    --max_terms 20 \
    --no_gt_max_terms "${PHASE6_NO_GT_MAX_TERMS}" \
    --empty_prob_no_gt "${PHASE6_EMPTY_PROB_NO_GT}" \
    2>&1 | tee "${LOG_DIR}/phase6_rebuild.log"
  rebuild_rc="${PIPESTATUS[0]}"
  if [[ ${rebuild_rc} -ne 0 || ! -s "${PHASE6_DATASET}" ]]; then
    abort_with_report "Phase 6 dataset rebuild failed (rc=${rebuild_rc})"
  fi
else
  log "Phase 6 dataset already exists: ${PHASE6_DATASET}"
fi

# Submit Phase 6 training via the same sbatch mechanism by forking a tiny
# wrapper that reuses run_adversarial_train_sbatch's VARIANTS pattern.
log "Submitting Phase 6 training..."
PHASE6_SUBMIT_LOG="${LOG_DIR}/phase6_submit.log"

SBATCH_TMP="$(mktemp /tmp/phase6_train_XXXXXX.sh)"
# Partition: use aries (GPUs idle). The docker GPU isolation + WANDB_MODE=offline
# fixes from Phase 4 are needed.
cat > "${SBATCH_TMP}" << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=train_${PHASE6_TAG}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
set -euo pipefail

echo "[TRAIN ${PHASE6_TAG}] starting on \$(hostname)"
ALLOCATED_GPUS="\${CUDA_VISIBLE_DEVICES:-0,1}"

docker run --rm \
    --gpus "\"device=\${ALLOCATED_GPUS}\"" \
    --shm-size=32g \
    --ipc=host \
    -e CUDA_VISIBLE_DEVICES="0,1" \
    -e NCCL_P2P_DISABLE=1 \
    -e NCCL_IB_DISABLE=1 \
    -e WANDB_MODE=offline \
    -e DATASET_PATH_OVERRIDE="${PHASE6_DATASET}" \
    -e SAVE_BASE_OVERRIDE="${PHASE6_SAVE_BASE}" \
    -e LORA_RANK_OVERRIDE="16" \
    -v /mnt/taurus/home/jiaxuanluo/InfiniSST:/workspace/InfiniSST \
    -v /mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct:/workspace/Qwen3-Omni-30B-A3B-Instruct:ro \
    -v /mnt/gemini/data:/mnt/gemini/data \
    -v /mnt/gemini/data1:/mnt/gemini/data1 \
    -v /mnt/gemini/data2:/mnt/gemini/data2 \
    -v /mnt/taurus/data2:/mnt/taurus/data2 \
    modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.1 \
    bash /workspace/InfiniSST/documents/code/train/sst_omni_train/run_speech_llm_4gpu_maxsim.sh "${PHASE6_TAG}" "29553"
SBATCH_EOF

PHASE6_JOB_ID="$(sbatch --parsable \
  -p aries \
  -o "${LOG_DIR}/%j_phase6_train.out" \
  -e "${LOG_DIR}/%j_phase6_train.err" \
  "${SBATCH_TMP}")"
log "Phase 6 training submitted: job=${PHASE6_JOB_ID}"
echo "${PHASE6_JOB_ID}" > "${ORCH_DIR}/phase6_job.id"

log "Waiting for Phase 6 slurm job ${PHASE6_JOB_ID}..."
wait_for_slurm_job "${PHASE6_JOB_ID}" "${WAIT_PHASE6_TRAIN_MAX_SEC}" "phase6_train" || \
  log "Phase 6 slurm job ended non-COMPLETED; will still check for HF export"

PHASE6_HF="$(wait_for_hf_export "${PHASE6_SAVE_BASE}/r16" "${WAIT_PHASE6_TRAIN_MAX_SEC}" "phase6_hf")"
if [[ -z "${PHASE6_HF:-}" || ! -f "${PHASE6_HF}/config.json" ]]; then
  abort_with_report "Phase 6 HF export not found under ${PHASE6_SAVE_BASE}/r16"
fi
log "Phase 6 HF ready: ${PHASE6_HF}"

log "Running Phase 6 eval..."
if ! run_eval_blocking \
     "${PHASE6_TAG}" "${PHASE6_HF}" \
     "${PHASE6_BP_TSV}" "eval_phase6.log"; then
  abort_with_report "Phase 6 eval failed; see ${LOG_DIR}/eval_phase6.log"
fi

log "Phase 6 vs Experiment compare..."
PHASE6_DECISION_JSON="${ORCH_DIR}/phase6_vs_experiment_compare.json"
python3 "${ROOT_DIR}/documents/code/offline_sst_eval/phase5_compare.py" \
  --control-tsv "${EXPERIMENT_BP_TSV}" \
  --experiment-tsv "${PHASE6_BP_TSV}" \
  --decision-json "${PHASE6_DECISION_JSON}" \
  --term-fcr-threshold "${TERM_FCR_THRESHOLD}" \
  2>&1 | tee "${LOG_DIR}/phase6_compare.log"

write_final_report "SUCCESS (Phase 6 executed)"
log "========================================================================"
log "Orchestrator done."
log "========================================================================"
