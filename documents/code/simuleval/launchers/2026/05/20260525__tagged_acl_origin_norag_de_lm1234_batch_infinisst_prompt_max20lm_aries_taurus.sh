#!/usr/bin/env bash
set -euo pipefail

# En-De tagged ACL raw InfiniSST/no-RAG batch rerun with serial-compatible
# no-RAG prompt policy. Submit mode launches lm1/lm2 on Taurus and lm3/lm4 on
# Aries; child mode runs one same-LM five-sample batch.

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

MAX_NEW_TOKENS_PER_LM="${MAX_NEW_TOKENS_PER_LM:-40}"
RUN_STAMP="${RUN_STAMP_OVERRIDE:-20260525T115605_origin_norag_de_batch_infprompt_max${MAX_NEW_TOKENS_PER_LM}lm}"
MODE="${MODE:-submit}"
EVENT_ID="${EVENT_ID_OVERRIDE:-20260525T115605__simuleval__tagged_acl_origin_norag_de_lm1234_batch_infinisst_prompt_max${MAX_NEW_TOKENS_PER_LM}lm_aries_taurus}"

BATCH_LAUNCHER="${BATCH_LAUNCHER_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__batched_vllm_rag_eval.sh}"
MODEL_NAME="${MODEL_NAME_OVERRIDE:-/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_origin-bsz4}"
INPUT_DIR="${INPUT_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_de_lm1to4_20260524T075533_tagacl_newv9_hn1024_tau078_raw_de_retry1/__inputs__/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/full_all/de/all}"
RAW_GLOSSARY="${RAW_GLOSSARY_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
NOTES_FILE="${NOTES_FILE_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260525__tagged_acl_origin_norag_de_lm1234_batch_infinisst_prompt_max${MAX_NEW_TOKENS_PER_LM}lm.md}"
PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python}"

OUT_ROOT_BASE="${OUT_ROOT_BASE_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_origin_norag_de_batch_infinisst_prompt_max${MAX_NEW_TOKENS_PER_LM}lm_${RUN_STAMP}}"
LOG_ROOT_BASE="${LOG_ROOT_BASE_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_origin_norag_de_batch_infinisst_prompt_max${MAX_NEW_TOKENS_PER_LM}lm_${RUN_STAMP}}"
CACHE_BASE="${CACHE_BASE_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/cache/tagged_acl_origin_norag_de_batch_infinisst_prompt_max${MAX_NEW_TOKENS_PER_LM}lm_${RUN_STAMP}}"
INPUT_WORK_DIR="${OUT_ROOT_BASE}/__inputs__"
SRC_LIST_PORTABLE="${INPUT_WORK_DIR}/source.portable.list"

GLOSSARY_TAG="acl6060_tagged_gt_raw_min_norm2"
DENSITY_TAG_PREFIX="tagacl_origin_norag_infprompt_max${MAX_NEW_TOKENS_PER_LM}lm"

fail() {
  echo "[ERROR] $*" >&2
  exit 3
}

require_file() {
  local path="$1"
  [[ -s "${path}" ]] || fail "Missing/empty required file: ${path}"
}

gpu_is_idle() {
  local gpu="$1" csv line mem util
  csv="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"
  line="$(awk -F, -v g="${gpu}" '$1 + 0 == g {print $0}' <<< "${csv}")"
  [[ -n "${line}" ]] || return 1
  mem="$(awk -F, '{gsub(/[[:space:]]/, "", $2); print $2}' <<< "${line}")"
  util="$(awk -F, '{gsub(/[[:space:]]/, "", $3); print $3}' <<< "${line}")"
  (( mem <= 2048 && util <= 25 ))
}

wait_pair_idle() {
  local pair="$1" g0 g1
  IFS=',' read -r g0 g1 <<< "${pair}"
  while true; do
    if gpu_is_idle "${g0}" && gpu_is_idle "${g1}"; then
      return 0
    fi
    echo "[WAIT] host=$(hostname -s) gpu_pair=${pair} busy; retry in 20s" >&2
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,name --format=csv,noheader,nounits >&2 || true
    sleep 20
  done
}

wait_no_own_vllm() {
  local poll_secs="${WAIT_OWN_VLLM_POLL_SECS_OVERRIDE:-60}"
  local lines
  while true; do
    lines="$(
      ps -u "$(id -u)" -o pid=,ppid=,stat=,cmd= \
        | awk '
          /documents\/code\/simuleval\/src\/batched_vllm_rag_eval.py|VLLM::Worker|VLLM::EngineCore/ {
            print
          }
        ' \
        | grep -v "awk " || true
    )"
    if [[ -z "${lines}" ]]; then
      return 0
    fi
    echo "[WAIT] host=$(hostname -s) existing own vLLM/batched eval still running; retry in ${poll_secs}s" >&2
    echo "${lines}" >&2
    sleep "${poll_secs}"
  done
}

clean_stale_shm() {
  local me
  me="$(id -un)"
  find /dev/shm -maxdepth 1 -user "${me}" -mmin +1 \
    \( -name 'psm_*' -o -name 'loky-*' -o -name 'torch_*' -o -name 'vllm_*' \) \
    -delete 2>/dev/null || true
  df -h /dev/shm >&2 || true
}

pick_tmp_root() {
  local explicit="${EVAL_TMPDIR_ROOT_OVERRIDE:-}"
  if [[ -n "${explicit}" ]]; then
    printf '%s\n' "${explicit}"
    return 0
  fi
  local free_tmp_mb
  free_tmp_mb="$(df -Pm /tmp 2>/dev/null | awk 'NR == 2 {print $4}')"
  if [[ -n "${free_tmp_mb}" && "${free_tmp_mb}" -ge 2048 ]]; then
    printf '/tmp/jxnoi_seq_%s\n' "$(hostname -s)"
  else
    printf '/dev/shm/jxnoi_seq_%s\n' "$(hostname -s)"
  fi
}

prepare_common_inputs() {
  mkdir -p "${OUT_ROOT_BASE}" "${LOG_ROOT_BASE}" "${CACHE_BASE}" "${INPUT_WORK_DIR}"
  require_file "${BATCH_LAUNCHER}"
  require_file "${MODEL_NAME}/config.json"
  require_file "${RAW_GLOSSARY}"
  require_file "${NOTES_FILE}"
  for f in source.list target.list source_text.txt ref.txt audio.yaml; do
    require_file "${INPUT_DIR}/${f}"
  done
  sed \
    -e 's#^/mnt/data/siqiouyang/#/mnt/taurus/data/siqiouyang/#' \
    -e 's#^/mnt/data1/siqiouyang/#/mnt/taurus/data1/siqiouyang/#' \
    "${INPUT_DIR}/source.list" > "${SRC_LIST_PORTABLE}"
  if grep -q '^/mnt/data/' "${SRC_LIST_PORTABLE}"; then
    fail "portable source rewrite left node-local /mnt/data paths in ${SRC_LIST_PORTABLE}"
  fi
  while IFS= read -r wav_path; do
    [[ -n "${wav_path}" ]] || continue
    require_file "${wav_path}"
  done < "${SRC_LIST_PORTABLE}"
}

run_one() {
  local lm="${LM_VALUE:?LM_VALUE required}"
  local gpu_pair="${GPU_PAIR:?GPU_PAIR required}"
  local max_new=$((MAX_NEW_TOKENS_PER_LM * lm))
  local host_slug
  host_slug="$(hostname -s)"
  local child_id="lm${lm}_${host_slug}_g${gpu_pair//,/}"
  local out_root="${OUT_ROOT_BASE}/${child_id}"
  local output_base="${out_root}/origin_norag_de_batch_infprompt_max${MAX_NEW_TOKENS_PER_LM}lm"
  local log_root="${LOG_ROOT_BASE}/${child_id}"
  local cache_root="${CACHE_BASE}/${child_id}"
  local eval_tmpdir="${EVAL_TMPDIR_OVERRIDE:-/tmp/jxnoi${lm}}"
  local density_tag="${DENSITY_TAG_PREFIX}_lm${lm}"

  prepare_common_inputs
  mkdir -p "${out_root}" "${output_base}" "${log_root}" "${cache_root}" "${eval_tmpdir}"
  wait_pair_idle "${gpu_pair}"

  export XDG_CACHE_HOME="${cache_root}/xdg"
  export TRITON_CACHE_DIR="${cache_root}/triton"
  export TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor"
  export CUDA_CACHE_PATH="${cache_root}/cuda"
  export HF_HOME="${cache_root}/hf"
  export HF_HUB_CACHE="${cache_root}/hf/hub"
  export TRANSFORMERS_CACHE="${cache_root}/hf/transformers"
  export VLLM_CACHE_ROOT="${cache_root}/vllm"
  export NUMBA_CACHE_DIR="${cache_root}/numba"
  export WANDB_CACHE_DIR="${cache_root}/wandb_cache"
  mkdir -p "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
    "${CUDA_CACHE_PATH}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${VLLM_CACHE_ROOT}" \
    "${NUMBA_CACHE_DIR}" "${WANDB_CACHE_DIR}"

  {
    echo "event_id=${EVENT_ID}"
    echo "run_stamp=${RUN_STAMP}"
    echo "host=${host_slug}"
    echo "gpu_pair=${gpu_pair}"
    echo "model=${MODEL_NAME}"
    echo "input_dir=${INPUT_DIR}"
    echo "source_list_portable=${SRC_LIST_PORTABLE}"
    echo "raw_glossary=${RAW_GLOSSARY}"
    echo "output_base=${output_base}"
    echo "lang=de"
    echo "lm=${lm}"
    echo "talks_per_lm=5"
    echo "disable_rag=1"
    echo "empty_term_map_policy=omit"
    echo "norag_prompt_policy=serial_compat"
    echo "max_new_tokens_policy=${MAX_NEW_TOKENS_PER_LM}*lm"
    echo "max_new_tokens=${max_new}"
  } | tee "${out_root}/run_meta.txt"

  ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
  RUN_TAG_OVERRIDE="${RUN_STAMP}_${child_id}" \
  LANG_CODE_OVERRIDE="de" \
  LMS_OVERRIDE="${lm}" \
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${gpu_pair}" \
  VLLM_TP_SIZE_OVERRIDE=2 \
  MAX_NUM_SEQS_OVERRIDE=5 \
  SCHEDULER_BATCH_SIZE_OVERRIDE=5 \
  SCHEDULE_MODE_OVERRIDE=round_robin \
  DISABLE_RAG_OVERRIDE=1 \
  RAG_TOP_K_OVERRIDE=0 \
  RAG_SCORE_THRESHOLD_OVERRIDE=0 \
  RAG_BATCH_RETRIEVAL_OVERRIDE=0 \
  MAX_NEW_TOKENS_OVERRIDE="${max_new}" \
  MAX_NEW_TOKENS_POLICY_OVERRIDE=fixed \
  TEMPERATURE_OVERRIDE=0.6 \
  TOP_P_OVERRIDE=0.95 \
  TOP_K_DECODE_OVERRIDE=20 \
  MAX_CACHE_SECONDS_OVERRIDE=80 \
  KEEP_CACHE_SECONDS_OVERRIDE=60 \
  MIN_CACHE_CHUNKS_OVERRIDE=1 \
  VLLM_LIMIT_AUDIO_OVERRIDE=128 \
  VLLM_ENFORCE_EAGER_OVERRIDE=1 \
  VLLM_ENABLE_PREFIX_CACHING=1 \
  VLLM_MAX_MODEL_LEN_OVERRIDE=12288 \
  MAX_MODEL_LEN_OVERRIDE=12288 \
  VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
  VLLM_MOE_USE_DEEP_GEMM=0 \
  VLLM_USE_FUSED_MOE_GROUPED_TOPK=0 \
  GPU_MEMORY_UTILIZATION_OVERRIDE=0.78 \
  MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
  SRC_LIST_OVERRIDE="${SRC_LIST_PORTABLE}" \
  TGT_LIST_OVERRIDE="${INPUT_DIR}/target.list" \
  SOURCE_TEXT_FILE_OVERRIDE="${INPUT_DIR}/source_text.txt" \
  REF_FILE_OVERRIDE="${INPUT_DIR}/ref.txt" \
  AUDIO_YAML_OVERRIDE="${INPUT_DIR}/audio.yaml" \
  GLOSSARY_PATH_OVERRIDE="${RAW_GLOSSARY}" \
  EVAL_GLOSSARY_PATH_OVERRIDE="${RAW_GLOSSARY}" \
  OUTPUT_BASE_OVERRIDE="${output_base}" \
  DENSITY_TAG_OVERRIDE="${density_tag}" \
  GLOSSARY_TAG_OVERRIDE="${GLOSSARY_TAG}" \
  EMPTY_TERM_MAP_POLICY_OVERRIDE=omit \
  NORAG_PROMPT_POLICY_OVERRIDE=serial_compat \
  TERM_FCR_POLICY_OVERRIDE=source_ref_negative_sentence \
  STRIP_OUTPUT_TAGS_OVERRIDE=term \
  WANDB_LOG_OVERRIDE=0 \
  NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
  LOG_ROOT_OVERRIDE="${log_root}/batch" \
  EVAL_TMPDIR_OVERRIDE="${eval_tmpdir}" \
  bash "${BATCH_LAUNCHER}" \
    > "${log_root}/launcher.out" \
    2> "${log_root}/launcher.err"

  "${PYTHON_BIN}" - "${output_base}" "${lm}" "${max_new}" <<'PY'
import csv
import sys
from pathlib import Path

output_base = Path(sys.argv[1])
lm = sys.argv[2]
max_new = sys.argv[3]
paths = sorted(output_base.glob(f"de/*_lm{lm}_*/eval_results.tsv"))
if len(paths) != 1:
    raise SystemExit(f"expected one lm{lm} eval_results.tsv, found {len(paths)}: {paths}")
rows = list(csv.DictReader(paths[0].open("r", encoding="utf-8"), delimiter="\t"))
if len(rows) != 1:
    raise SystemExit(f"expected one row in {paths[0]}, got {len(rows)}")
out_dir = paths[0].parent
instances = out_dir / "instances.log"
strip = out_dir / "instances.strip_term.log"
inst_rows = sum(1 for _ in instances.open("r", encoding="utf-8"))
strip_rows = sum(1 for _ in strip.open("r", encoding="utf-8"))
if inst_rows != 5 or strip_rows != 5:
    raise SystemExit(f"expected 5 instance rows, got raw={inst_rows} strip={strip_rows}")
row = rows[0]
summary = output_base / "__summary__" / f"summary_de_lm{lm}.tsv"
summary.parent.mkdir(parents=True, exist_ok=True)
fields = [
    "method_key", "lang", "lm", "max_new_tokens", "prompt_policy",
    "empty_term_map_policy", "BLEU", "StreamLAAL", "StreamLAAL_CA",
    "TERM_ACC", "TERM_CORRECT", "TERM_TOTAL", "REAL_TERM_ADOPT",
    "TERM_FCR", "instances_log_rows", "instances_strip_term_log_rows",
    "eval_results", "instances_log", "instances_strip_term_log",
]
with summary.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    w.writeheader()
    w.writerow({
        "method_key": "origin_norag_batch_infinisst_prompt",
        "lang": row.get("lang_code", "de"),
        "lm": lm,
        "max_new_tokens": max_new,
        "prompt_policy": "serial_compat",
        "empty_term_map_policy": "omit",
        "BLEU": row.get("BLEU", ""),
        "StreamLAAL": row.get("StreamLAAL", ""),
        "StreamLAAL_CA": row.get("StreamLAAL_CA", ""),
        "TERM_ACC": row.get("TERM_ACC", ""),
        "TERM_CORRECT": row.get("TERM_CORRECT", ""),
        "TERM_TOTAL": row.get("TERM_TOTAL", ""),
        "REAL_TERM_ADOPT": row.get("REAL_TERM_ADOPT", ""),
        "TERM_FCR": row.get("TERM_FCR", ""),
        "instances_log_rows": inst_rows,
        "instances_strip_term_log_rows": strip_rows,
        "eval_results": str(paths[0]),
        "instances_log": str(instances),
        "instances_strip_term_log": str(strip),
    })
print(summary)
PY

  date -u +%Y-%m-%dT%H:%M:%SZ > "${out_root}/.success"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | tee "${log_root}/gpu_postrun.csv"
  echo "[ALL DONE] lm=${lm} summary=${output_base}/__summary__/summary_de_lm${lm}.tsv"
}

submit_one() {
  local host="$1" lm="$2" gpu_pair="$3" eval_tmpdir="$4"
  local child_id="lm${lm}_${host}_g${gpu_pair//,/}"
  local log_root="${LOG_ROOT_BASE}/${child_id}"
  mkdir -p "${log_root}"
  local cmd
  cmd="cd '${ROOT_DIR}' && MODE=run_one RUN_STAMP_OVERRIDE='${RUN_STAMP}' EVENT_ID_OVERRIDE='${EVENT_ID}' LM_VALUE='${lm}' GPU_PAIR='${gpu_pair}' EVAL_TMPDIR_OVERRIDE='${eval_tmpdir}' bash '${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260525__tagged_acl_origin_norag_de_lm1234_batch_infinisst_prompt_max20lm_aries_taurus.sh'"
  printf '%s\n' "${cmd}" > "${log_root}/command.sh"
  if [[ "${host}" == "$(hostname -s)" ]]; then
    setsid bash -lc "${cmd}" > "${log_root}/outer.out" 2> "${log_root}/outer.err" < /dev/null &
    pid="$!"
  else
    ssh "${host}" "setsid bash -lc $(printf '%q' "${cmd}") > '${log_root}/outer.out' 2> '${log_root}/outer.err' < /dev/null & echo \$! > '${log_root}/outer.pid'; cat '${log_root}/outer.pid'"
    pid="$(ssh "${host}" "cat '${log_root}/outer.pid'")"
  fi
  echo "${pid}" > "${log_root}/outer.pid"
  echo -e "${lm}\t${host}\t${gpu_pair}\t${pid}\t${log_root}/outer.pid\t${log_root}/outer.out\t${log_root}/outer.err"
}

run_sequence() {
  local host_slug tmp_root pair_idx lm pair
  host_slug="$(hostname -s)"
  tmp_root="$(pick_tmp_root)"
  mkdir -p "${tmp_root}" "${LOG_ROOT_BASE}"
  IFS=';' read -r -a pairs <<< "${GPU_PAIRS_CSV_OVERRIDE:?GPU_PAIRS_CSV_OVERRIDE required for MODE=run_sequence}"
  pair_idx=0
  {
    echo "event_id=${EVENT_ID}"
    echo "run_stamp=${RUN_STAMP}"
    echo "host=${host_slug}"
    echo "lms=${LMS_OVERRIDE}"
    echo "gpu_pairs=${GPU_PAIRS_CSV_OVERRIDE}"
    echo "tmp_root=${tmp_root}"
    echo "sequence_policy=one_vllm_batch_at_a_time_after_existing_own_vllm_exits"
  } | tee "${LOG_ROOT_BASE}/sequence_${host_slug}.meta"
  for lm in ${LMS_OVERRIDE}; do
    if (( pair_idx >= ${#pairs[@]} )); then
      pair_idx=0
    fi
    pair="${pairs[$pair_idx]}"
    pair_idx=$((pair_idx + 1))
    wait_no_own_vllm
    clean_stale_shm
    echo "[SEQUENCE] host=${host_slug} starting lm=${lm} gpu_pair=${pair}" | tee -a "${LOG_ROOT_BASE}/sequence_${host_slug}.log"
    LM_VALUE="${lm}" GPU_PAIR="${pair}" EVAL_TMPDIR_OVERRIDE="${tmp_root}/lm${lm}" run_one
    echo "[SEQUENCE] host=${host_slug} completed lm=${lm} gpu_pair=${pair}" | tee -a "${LOG_ROOT_BASE}/sequence_${host_slug}.log"
  done
  echo "[SEQUENCE DONE] host=${host_slug} lms=${LMS_OVERRIDE}" | tee -a "${LOG_ROOT_BASE}/sequence_${host_slug}.log"
}

if [[ "${MODE}" == "run_one" ]]; then
  run_one
  exit 0
fi

if [[ "${MODE}" == "run_sequence" ]]; then
  run_sequence
  exit 0
fi

if [[ "${MODE}" != "submit" ]]; then
  fail "Unsupported MODE=${MODE}"
fi

prepare_common_inputs
df -h /mnt/gemini/data1 /tmp /dev/shm | tee "${LOG_ROOT_BASE}/prelaunch_df_taurus.txt" || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,pstate,name --format=csv,noheader,nounits \
  | tee "${LOG_ROOT_BASE}/prelaunch_gpus_taurus.csv"
ssh aries 'df -h /mnt/gemini/data1 /tmp /dev/shm; nvidia-smi --query-gpu=index,memory.used,utilization.gpu,pstate,name --format=csv,noheader,nounits' \
  | tee "${LOG_ROOT_BASE}/prelaunch_aries.txt"

{
  echo -e "lm\thost\tgpu_pair\tpid\tpid_file\touter_out\touter_err"
  submit_one "taurus" 1 "2,3" "/tmp/jxnoi1"
  submit_one "taurus" 2 "4,5" "/tmp/jxnoi2"
  submit_one "aries" 3 "0,1" "/dev/shm/jxnoi3"
  submit_one "aries" 4 "2,3" "/dev/shm/jxnoi4"
} | tee "${LOG_ROOT_BASE}/pids.tsv"

echo "[SUBMITTED] ${LOG_ROOT_BASE}/pids.tsv"
