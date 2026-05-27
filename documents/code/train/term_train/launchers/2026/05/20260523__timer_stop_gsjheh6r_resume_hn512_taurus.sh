#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
LOCAL_REPO_ROOT="${LOCAL_REPO_ROOT:-/home/jiaxuanluo/InfiniSST}"
cd "${REPO_ROOT}"

SLEEP_SECONDS="${SLEEP_SECONDS:-10800}"
TIMER_EVENT_ID="${TIMER_EVENT_ID:-20260523T0754__maintenance__timer_stop_hn256_resume_hn512_taurus}"
HN512_EVENT_ID="${HN512_EVENT_ID:-20260523T1054__retriever_train__varctx_lmlb_v3_hn512_resume_after_hn256_taurus6}"

HN256_RUN_ID="${HN256_RUN_ID:-gsjheh6r}"
HN256_MATCH="${HN256_MATCH:-step1200_aclmetric_reset}"
HN256_PID_FILE="${HN256_PID_FILE:-/mnt/gemini/data1/jiaxuanluo/logs/hn256_step1200_aclmetric_reset_taurus6_20260523T0335Z.direct.pid}"
HN256_MANIFEST="${HN256_MANIFEST:-documents/code/train/term_train/manifests/2026/05/20260523T0335__retriever_train__varctx_lmlb_v3_hn256_step1200_aclmetric_reset_taurus6.json}"

HN512_SOURCE_RUN_ID="${HN512_SOURCE_RUN_ID:-5fwrs7rh}"
HN512_LAUNCHER="${HN512_LAUNCHER:-documents/code/train/term_train/launchers/2026/05/20260522__varctx_lmlb_v3_hn512_gc256_taurus6.sh}"
HN512_MANIFEST="${HN512_MANIFEST:-documents/code/train/term_train/manifests/2026/05/20260523T1054__retriever_train__varctx_lmlb_v3_hn512_resume_after_hn256_taurus6.json}"
HN512_NOTES_FILE="${HN512_NOTES_FILE:-${REPO_ROOT}/documents/code/train/term_train/notes/2026/05/20260523__varctx_lmlb_v3_hn512_resume_after_hn256_taurus6.md}"
HN512_RESUME_CKPT="${HN512_RESUME_CKPT:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8190_gc256_wr1000k_m0.0_maxsim_mfa_variantE_hn512_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu012345_taurus_latest.pt}"
HN512_CUDA_DEVICE_LIST="${HN512_CUDA_DEVICE_LIST:-0,1,2,3,4,5}"
HN512_WANDB_WAIT_SECONDS="${HN512_WANDB_WAIT_SECONDS:-14400}"

LOG_DIR="${LOG_DIR:-/mnt/gemini/data1/jiaxuanluo/logs}"
mkdir -p "${LOG_DIR}"

ts() {
    date -u +%Y-%m-%dT%H:%M:%SZ
}

log() {
    echo "[$(ts)] $*"
}

update_manifest() {
    local manifest_path="$1"
    shift
    python3 - "$manifest_path" "$@" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_absolute():
    path = Path("/mnt/taurus/home/jiaxuanluo/InfiniSST") / path
data = json.loads(path.read_text())
for pair in sys.argv[2:]:
    key, value = pair.split("=", 1)
    if key.startswith("metadata."):
        data.setdefault("metadata", {})[key[len("metadata."):]] = value
    else:
        data[key] = value
now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
data["updated_at"] = now
data.setdefault("metadata", {})["manifest_updated_at_utc"] = now
path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n")
PY
}

register_manifest() {
    local manifest_path="$1"
    python documents/code/general/experiment_event.py register "${manifest_path}" || \
        log "[WARN] manifest register failed for ${manifest_path}"
}

annotate_wandb_paused() {
    python documents/code/general/wandb_tool.py --project qwen3_rag annotate "${HN256_RUN_ID}" \
        --remove-tags status:running \
        --add-tags status:paused \
        --set-summary "paused_at_utc=$(ts)" \
        "pause_reason=scheduled_hn512_resume" \
        "replacement_event_id=${HN512_EVENT_ID}" \
        --yes || log "[WARN] W&B pause annotation failed for ${HN256_RUN_ID}"
}

stop_pid_group() {
    local pid="$1"
    if [ -z "${pid}" ] || ! kill -0 "${pid}" 2>/dev/null; then
        return 0
    fi
    local pgid
    pgid="$(ps -o pgid= -p "${pid}" | tr -d ' ')"
    if [ -z "${pgid}" ]; then
        log "[WARN] could not resolve PGID for pid=${pid}"
        return 0
    fi
    log "Sending SIGTERM to pid=${pid} pgid=${pgid}"
    kill -TERM "-${pgid}" 2>/dev/null || true
}

hn256_pids() {
    {
        if [ -f "${HN256_PID_FILE}" ]; then
            tr -cd '0-9\n' < "${HN256_PID_FILE}" || true
        fi
        pgrep -f "${HN256_MATCH}" || true
    } | awk 'NF && $1 != self {print $1}' self="$$" | sort -nu
}

hn256_still_running() {
    pgrep -f "${HN256_MATCH}" >/dev/null 2>&1
}

force_kill_hn256_if_needed() {
    if ! hn256_still_running; then
        return 0
    fi
    log "[WARN] HN256 still has matching processes after grace period; escalating"
    while read -r pid; do
        [ -n "${pid}" ] || continue
        local pgid
        pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
        if [ -n "${pgid}" ]; then
            log "Sending SIGKILL to pid=${pid} pgid=${pgid}"
            kill -KILL "-${pgid}" 2>/dev/null || true
        fi
    done < <(hn256_pids)
}

stop_hn256() {
    log "Stopping HN256 run ${HN256_RUN_ID}; match=${HN256_MATCH}"
    local found=0
    while read -r pid; do
        [ -n "${pid}" ] || continue
        found=1
        stop_pid_group "${pid}"
    done < <(hn256_pids)
    if [ "${found}" -eq 0 ]; then
        log "No HN256 matching processes found."
    fi

    local deadline=$((SECONDS + 600))
    while hn256_still_running && [ "${SECONDS}" -lt "${deadline}" ]; do
        sleep 10
    done
    force_kill_hn256_if_needed
    sleep 10

    annotate_wandb_paused
    update_manifest "${HN256_MANIFEST}" \
        "status=paused" \
        "metadata.scheduled_pause_at_utc=$(ts)" \
        "metadata.pause_reason=scheduled_hn512_resume" \
        "metadata.replacement_event_id=${HN512_EVENT_ID}" \
        "metadata.pause_timer_event_id=${TIMER_EVENT_ID}"
    register_manifest "${HN256_MANIFEST}"
}

wait_for_process_exit_or_startup() {
    local pid="$1"
    local log_path="$2"
    local deadline=$((SECONDS + HN512_WANDB_WAIT_SECONDS))
    local run_id=""
    while [ "${SECONDS}" -lt "${deadline}" ]; do
        if [ -f "${log_path}" ]; then
            run_id="$(grep -oE 'runs/[A-Za-z0-9]+' "${log_path}" | tail -1 | cut -d/ -f2 || true)"
            if [ -z "${run_id}" ]; then
                run_id="$(grep -oE 'setting up run [A-Za-z0-9]+' "${log_path}" | tail -1 | awk '{print $4}' || true)"
            fi
            if [ -n "${run_id}" ]; then
                echo "${run_id}"
                return 0
            fi
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            return 1
        fi
        sleep 20
    done
    return 2
}

launch_hn512() {
    if [ ! -f "${HN512_RESUME_CKPT}" ]; then
        log "[FATAL] missing HN512 resume checkpoint: ${HN512_RESUME_CKPT}"
        exit 2
    fi
    local run_stamp
    run_stamp="hn512_resume_after_${HN256_RUN_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
    local hn512_log="${LOG_DIR}/${run_stamp}.direct.log"
    local hn512_pid_file="${LOG_DIR}/${run_stamp}.direct.pid"
    local hn512_wandb_name="variantE_hn512_gsv2full_gsdedup_varctx576_v3_bs8190_gc256_resume_latest_after_${HN256_RUN_ID}_${run_stamp}"

    log "Launching HN512 resume from ${HN512_RESUME_CKPT}"
    log "HN512 log: ${hn512_log}"
    update_manifest "${HN512_MANIFEST}" \
        "status=launching" \
        "metadata.scheduled_by_event_id=${TIMER_EVENT_ID}" \
        "metadata.resume_after_run_id=${HN256_RUN_ID}" \
        "metadata.launch_log=${hn512_log}" \
        "metadata.pid_file=${hn512_pid_file}" \
        "metadata.resume_checkpoint=${HN512_RESUME_CKPT}" \
        "metadata.cuda_device_list=${HN512_CUDA_DEVICE_LIST}" \
        "metadata.launch_requested_at_utc=$(ts)"
    register_manifest "${HN512_MANIFEST}"

    setsid env \
        RUN_STAMP="${run_stamp}" \
        CUDA_DEVICE_LIST="${HN512_CUDA_DEVICE_LIST}" \
        NUM_GPUS=6 \
        TARGET_GLOBAL_BATCH=8192 \
        PER_GPU_BATCH=1365 \
        BATCH_SIZE=8190 \
        GRAD_CACHE_CHUNK_SIZE=256 \
        HARD_NEG_K=0 \
        HARD_NEG_K_PER_SAMPLE=512 \
        RESUME="${HN512_RESUME_CKPT}" \
        RESET_SCHEDULER=false \
        RESET_BEST_ON_RESUME=false \
        SELECT_CLEAN_GPUS=true \
        WAIT_FOR_CLEAN_GPUS=true \
        GPU_CLEAN_THRESHOLD_MIB=500 \
        GPU_WAIT_INTERVAL_SEC=60 \
        GPU_WAIT_TIMEOUT_SEC=172800 \
        MASTER_PORT=29996 \
        WANDB_EXP_NAME="${hn512_wandb_name}" \
        NOTES_FILE="${HN512_NOTES_FILE}" \
        MEDICINE_DEV_JSONL="/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_dev_dataset.jsonl" \
        MEDICINE_EVAL_WIKI_GLOSSARY="/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_glossary_gt_plus_medicine_wiki_gs10000.json" \
        EXTRA_WANDB_TAGS="variant:hn512_varctx576_v3 compute:taurus-6gpu ablation:hard_neg512 source:lh1b88kw resume_of:5fwrs7rh scheduled_after:${HN256_RUN_ID}" \
        BASELINE_RUN_IDS="lh1b88kw 5fwrs7rh ${HN256_RUN_ID} e981df6j 40fgbr2y bgz7akb6 ah9u1bao dxwrgbln" \
        RUN_VERDICT="Scheduled HN512 resume after pausing ${HN256_RUN_ID}; resume from latest HN512 checkpoint, preserve existing best trackers, dev-primary checkpoint metric remains eval_dev/recall@10_gs10000." \
        bash "${HN512_LAUNCHER}" > "${hn512_log}" 2>&1 < /dev/null &
    local hn512_pid=$!
    echo "${hn512_pid}" > "${hn512_pid_file}"

    update_manifest "${HN512_MANIFEST}" \
        "status=running_pending_wandb" \
        "command=setsid env RUN_STAMP=${run_stamp} CUDA_DEVICE_LIST=${HN512_CUDA_DEVICE_LIST} NUM_GPUS=6 TARGET_GLOBAL_BATCH=8192 PER_GPU_BATCH=1365 BATCH_SIZE=8190 GRAD_CACHE_CHUNK_SIZE=256 HARD_NEG_K=0 HARD_NEG_K_PER_SAMPLE=512 RESUME=${HN512_RESUME_CKPT} RESET_SCHEDULER=false RESET_BEST_ON_RESUME=false bash ${HN512_LAUNCHER} > ${hn512_log} 2>&1 < /dev/null &" \
        "metadata.direct_launch_pid=${hn512_pid}" \
        "metadata.launch_log=${hn512_log}" \
        "metadata.pid_file=${hn512_pid_file}" \
        "metadata.launched_at_utc=$(ts)"
    register_manifest "${HN512_MANIFEST}"
    log "HN512 launcher pid=${hn512_pid}"

    local new_run_id=""
    if new_run_id="$(wait_for_process_exit_or_startup "${hn512_pid}" "${hn512_log}")"; then
        log "Detected HN512 W&B run id: ${new_run_id}"
        python documents/code/general/wandb_tool.py --project qwen3_rag annotate "${new_run_id}" \
            --set-config "event_id=${HN512_EVENT_ID}" \
            "event_manifest_path=${HN512_MANIFEST}" \
            "source_run_id=${HN512_SOURCE_RUN_ID}" \
            "resume_after_run_id=${HN256_RUN_ID}" \
            "resume_checkpoint=${HN512_RESUME_CKPT}" \
            --set-summary "event_id=${HN512_EVENT_ID}" \
            "resume_after_run_id=${HN256_RUN_ID}" \
            "source_run_id=${HN512_SOURCE_RUN_ID}" \
            --yes || log "[WARN] W&B annotation failed for ${new_run_id}"
        python documents/code/general/wandb_tool.py --project qwen3_rag db-sync --runs "${new_run_id}" --best-bundles || \
            log "[WARN] W&B db-sync failed for ${new_run_id}"
        update_manifest "${HN512_MANIFEST}" \
            "status=running" \
            "wandb_run_id=${new_run_id}" \
            "metadata.wandb_run_id=${new_run_id}" \
            "metadata.wandb_run_url=https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/${new_run_id}" \
            "metadata.wandb_detected_at_utc=$(ts)"
        register_manifest "${HN512_MANIFEST}"
    else
        local status=$?
        log "[WARN] HN512 W&B run id was not detected before wait ended; status=${status}. Check ${hn512_log}"
    fi
}

log "Timer event ${TIMER_EVENT_ID} armed; sleeping ${SLEEP_SECONDS}s before switching ${HN256_RUN_ID} -> HN512."
update_manifest "documents/code/train/term_train/manifests/2026/05/${TIMER_EVENT_ID}.json" \
    "status=armed" \
    "metadata.timer_pid=$$" \
    "metadata.armed_at_utc=$(ts)" \
    "metadata.sleep_seconds=${SLEEP_SECONDS}"
register_manifest "documents/code/train/term_train/manifests/2026/05/${TIMER_EVENT_ID}.json"

sleep "${SLEEP_SECONDS}"
log "Timer fired."
stop_hn256
launch_hn512
log "Timer action complete."
