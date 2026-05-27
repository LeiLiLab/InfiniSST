#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
cd "${REPO_ROOT}"

SLEEP_SECONDS="${SLEEP_SECONDS:-7200}"
NUM_SHARDS="${NUM_SHARDS:-8}"
PREPROCESS_PID="${PREPROCESS_PID:-3481040}"
LOG_DIR="${LOG_DIR:-/mnt/gemini/home/jiaxuanluo/logs}"
mkdir -p "${LOG_DIR}"
NUM_GPUS="${NUM_GPUS:-8}"

DURATION_SECS="${DURATION_SECS:-2.88 3.84 4.80 5.76}"
TRAIN_JSONL="${TRAIN_JSONL:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx2p88_3p84_4p80_5p76.jsonl}"
TRAIN_STATS_JSON="${TRAIN_STATS_JSON:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx2p88_3p84_4p80_5p76_stats.json}"
TRAIN_DIAG_JSON="${TRAIN_DIAG_JSON:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx2p88_3p84_4p80_5p76_diag.json}"
SHARD_DIR="${SHARD_DIR:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx2p88_3p84_4p80_5p76_shards}"
AUDIO_OUTPUT_DIR="${AUDIO_OUTPUT_DIR:-/mnt/gemini/home/jiaxuanluo/term_train_audio_chunks_gsv2full_gsdedup_varctx2p88_3p84_4p80_5p76}"
WIKI_AUDIO_OUTPUT_DIR="${WIKI_AUDIO_OUTPUT_DIR:-${AUDIO_OUTPUT_DIR}/wiki_synth}"

DEV_JSONL="${DEV_JSONL:-/mnt/gemini/home/jiaxuanluo/term_dev_dataset_varctx2p88_3p84_4p80_5p76_new_version.jsonl}"
DEV_STATS_JSON="${DEV_STATS_JSON:-/mnt/gemini/home/jiaxuanluo/term_dev_with_wiki_synth_normalized_varctx2p88_3p84_4p80_5p76_stats.json}"
DEV_DIAG_JSON="${DEV_DIAG_JSON:-/mnt/gemini/home/jiaxuanluo/term_dev_with_wiki_synth_normalized_varctx2p88_3p84_4p80_5p76_diag.json}"
ACL_JSONL="${ACL_JSONL:-/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary_varctx2p88_3p84_4p80_5p76/acl6060_dev_dataset.jsonl}"
ACL_STATS_JSON="${ACL_STATS_JSON:-/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary_varctx2p88_3p84_4p80_5p76/acl6060_dev_dataset_stats.json}"
ACL_DIAG_JSON="${ACL_DIAG_JSON:-/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary_varctx2p88_3p84_4p80_5p76/acl6060_dev_dataset_diag.json}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_gsdedup_varctx_lmlb_v3_tcmoff_ep6_8gpu_aries.sh}"
SMOKE_CHUNKS="${SMOKE_CHUNKS:-512 384 256 128}"
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-3}"
WANDB_PROJECT="${WANDB_PROJECT:-qwen3_rag}"
ENTITY_PATH_FRAGMENT="${ENTITY_PATH_FRAGMENT:-luojiaxuan1215-johns-hopkins-university/${WANDB_PROJECT}/runs/}"
WAIT_FOR_CLEAN_GPUS="${WAIT_FOR_CLEAN_GPUS:-true}"
GPU_FREE_THRESHOLD_MIB="${GPU_FREE_THRESHOLD_MIB:-500}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-300}"

echo "[SCHED] start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[SCHED] sleep_seconds=${SLEEP_SECONDS}"
sleep "${SLEEP_SECONDS}"
echo "[SCHED] woke=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

all_shards_ready() {
  local sid tag
  for sid in $(seq 0 $((NUM_SHARDS - 1))); do
    tag="$(printf '%02d' "${sid}")"
    [ -s "${SHARD_DIR}/part_${tag}.jsonl" ] || return 1
    [ -s "${SHARD_DIR}/part_${tag}_stats.json" ] || return 1
  done
  ! compgen -G "${SHARD_DIR}/part_*.jsonl.tmp" >/dev/null
}

if [ ! -s "${TRAIN_JSONL}" ] || [ ! -s "${TRAIN_STATS_JSON}" ]; then
  echo "[SCHED] train merge output missing; waiting for shard completion"
  while ! all_shards_ready; do
    if ! ps -p "${PREPROCESS_PID}" >/dev/null 2>&1; then
      echo "[SCHED] preprocess pid ${PREPROCESS_PID} is not running; still waiting for complete shards"
    fi
    sleep 600
  done
  if ps -p "${PREPROCESS_PID}" >/dev/null 2>&1; then
    echo "[SCHED] shards ready, waiting briefly for original preprocessor to finish merge"
    for _ in $(seq 1 30); do
      [ -s "${TRAIN_JSONL}" ] && [ -s "${TRAIN_STATS_JSON}" ] && break
      ps -p "${PREPROCESS_PID}" >/dev/null 2>&1 || break
      sleep 60
    done
  fi
fi

if [ ! -s "${TRAIN_JSONL}" ] || [ ! -s "${TRAIN_STATS_JSON}" ]; then
  echo "[SCHED] running explicit shard merge"
  python documents/code/data_pre/training_terms_for_retriever/merge_variable_context_shards.py \
    --shard-dir "${SHARD_DIR}" \
    --num-shards "${NUM_SHARDS}" \
    --output "${TRAIN_JSONL}" \
    --stats-json "${TRAIN_STATS_JSON}" \
    --audio-output-dir "${AUDIO_OUTPUT_DIR}" \
    --wiki-audio-output-dir "${WIKI_AUDIO_OUTPUT_DIR}" \
    --duration-secs "${DURATION_SECS}" \
    --duration-assignment balance_rows
fi

echo "[SCHED] diagnosing train/dev/acl datasets"
python documents/code/data_pre/training_terms_for_retriever/diagnose_variable_context_jsonl.py \
  --input "${TRAIN_JSONL}" \
  --stats-json "${TRAIN_STATS_JSON}" \
  --expected-duration-secs "${DURATION_SECS}" \
  --report-json "${TRAIN_DIAG_JSON}"
python documents/code/data_pre/training_terms_for_retriever/diagnose_variable_context_jsonl.py \
  --input "${DEV_JSONL}" \
  --stats-json "${DEV_STATS_JSON}" \
  --expected-duration-secs "${DURATION_SECS}" \
  --report-json "${DEV_DIAG_JSON}"
python documents/code/data_pre/training_terms_for_retriever/diagnose_variable_context_jsonl.py \
  --input "${ACL_JSONL}" \
  --stats-json "${ACL_STATS_JSON}" \
  --expected-duration-secs "${DURATION_SECS}" \
  --report-json "${ACL_DIAG_JSON}"

echo "[SCHED] canceling current aries jobs before v3 smoke/full run"
mapfile -t aries_jobs < <(squeue -h -u "${USER}" -p aries -o "%A" | sort -u)
if [ "${#aries_jobs[@]}" -gt 0 ]; then
  printf '[SCHED] scancel aries jobs: %s\n' "${aries_jobs[*]}"
  scancel "${aries_jobs[@]}"
  sleep 30
else
  echo "[SCHED] no aries jobs to cancel"
fi

extract_run_id() {
  local log_path="$1"
  if [ ! -f "${log_path}" ]; then
    return 1
  fi
  python - "${log_path}" "${ENTITY_PATH_FRAGMENT}" <<'PY'
import re, sys
path, frag = sys.argv[1], sys.argv[2]
text = open(path, "r", encoding="utf-8", errors="ignore").read()
for pat in [re.escape(frag) + r"([A-Za-z0-9_-]+)", r"run_id[=: ]+([A-Za-z0-9_-]{6,})"]:
    m = re.search(pat, text)
    if m:
        print(m.group(1))
        raise SystemExit(0)
raise SystemExit(1)
PY
}

sync_wandb_when_visible() {
  local job_id="$1"
  local job_name="$2"
  local log_path="${LOG_DIR}/${job_id}_${job_name}.out"
  local run_id=""
  for _ in $(seq 1 30); do
    run_id="$(extract_run_id "${log_path}" || true)"
    if [ -n "${run_id}" ]; then
      echo "[SCHED] wandb run id for job ${job_id}: ${run_id}"
      python documents/code/general/wandb_tool.py --project "${WANDB_PROJECT}" db-sync --runs "${run_id}" || true
      return 0
    fi
    sleep 60
  done
  echo "[SCHED][WARN] no WandB run id found yet in ${log_path}"
  return 1
}

wait_slurm_done() {
  local job_id="$1"
  while squeue -h -j "${job_id}" >/dev/null 2>&1 && [ -n "$(squeue -h -j "${job_id}")" ]; do
    sleep 60
  done
  sacct -j "${job_id}" --format=JobID,State,ExitCode,Elapsed -P -n | head -20 || true
}

clean_gpu_count() {
  python - "${NUM_GPUS}" "${GPU_FREE_THRESHOLD_MIB}" <<'PY'
import subprocess, sys
needed = int(sys.argv[1])
threshold = int(sys.argv[2])
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
except Exception:
    print("0")
    raise SystemExit(0)
free = 0
busy = []
for line in out.strip().splitlines():
    idx, used = [x.strip() for x in line.split(",")]
    used_i = int(used)
    if used_i <= threshold:
        free += 1
    else:
        busy.append((idx, used_i))
print(free)
if free < needed:
    print(f"[SCHED] waiting for clean GPUs: free={free}/{needed} threshold={threshold}MiB busy={busy}", file=sys.stderr)
PY
}

wait_for_clean_gpus() {
  if [ "${WAIT_FOR_CLEAN_GPUS}" != "true" ]; then
    return 0
  fi
  local free
  while true; do
    free="$(clean_gpu_count)"
    if [ "${free}" -ge "${NUM_GPUS}" ]; then
      echo "[SCHED] clean GPUs ready: free=${free}/${NUM_GPUS}"
      return 0
    fi
    sleep "${GPU_WAIT_SECONDS}"
  done
}

selected_chunk=""
for chunk in ${SMOKE_CHUNKS}; do
  wait_for_clean_gpus
  smoke_name="q3_varctx_v3_smoke_gc${chunk}"
  smoke_out="${LOG_DIR}/%j_${smoke_name}.out"
  smoke_err="${LOG_DIR}/%j_${smoke_name}.err"
  echo "[SCHED] submitting smoke grad_cache_chunk_size=${chunk}"
  smoke_job="$(sbatch --parsable \
    --job-name="${smoke_name}" \
    --output="${smoke_out}" \
    --error="${smoke_err}" \
    --export=ALL,TASK_TAG=smoke,MAX_STEPS=${SMOKE_MAX_STEPS},EPOCHS=1,SCHEDULER_EPOCHS=1,GRAD_CACHE_CHUNK_SIZE=${chunk},WANDB_EXP_NAME=variantE_varctx576_v3_smoke_gc${chunk},VERSION=3var_gsv2full_gsdedup_varctx576_gc${chunk}_v3_smoke \
    "${TRAIN_SCRIPT}")"
  smoke_job="${smoke_job%%;*}"
  echo "[SCHED] smoke_job=${smoke_job}"
  wait_slurm_done "${smoke_job}"
  smoke_log="${LOG_DIR}/${smoke_job}_${smoke_name}.out"
  smoke_state="$(sacct -j "${smoke_job}" --format=State -P -n | head -1 | cut -d'|' -f1 || true)"
  if [[ "${smoke_state}" == COMPLETED* ]]; then
    sync_wandb_when_visible "${smoke_job}" "${smoke_name}" || true
    selected_chunk="${chunk}"
    echo "[SCHED] selected grad_cache_chunk_size=${selected_chunk}"
    break
  fi
  if grep -Eiq "out of memory|cuda oom|CUDA.*memory|OutOfMemoryError" "${smoke_log}" "${LOG_DIR}/${smoke_job}_${smoke_name}.err" 2>/dev/null; then
    echo "[SCHED] smoke gc=${chunk} OOM; trying smaller chunk"
  else
    echo "[SCHED][WARN] smoke gc=${chunk} ended state=${smoke_state}; trying smaller chunk"
  fi
done

if [ -z "${selected_chunk}" ]; then
  echo "[SCHED][FATAL] all smoke grad_cache_chunk_size candidates failed" >&2
  exit 3
fi

full_name="q3_varctx_v3_gc${selected_chunk}"
wait_for_clean_gpus
echo "[SCHED] submitting full v3 training grad_cache_chunk_size=${selected_chunk}"
full_job="$(sbatch --parsable \
  --job-name="${full_name}" \
  --output="${LOG_DIR}/%j_${full_name}.out" \
  --error="${LOG_DIR}/%j_${full_name}.err" \
  --export=ALL,TASK_TAG=train,MAX_STEPS=0,EPOCHS=6,SCHEDULER_EPOCHS=6,GRAD_CACHE_CHUNK_SIZE=${selected_chunk} \
  "${TRAIN_SCRIPT}")"
full_job="${full_job%%;*}"
echo "[SCHED] full_job=${full_job}"
sync_wandb_when_visible "${full_job}" "${full_name}" || true
echo "[SCHED] done full_job=${full_job} selected_grad_cache_chunk_size=${selected_chunk}"
