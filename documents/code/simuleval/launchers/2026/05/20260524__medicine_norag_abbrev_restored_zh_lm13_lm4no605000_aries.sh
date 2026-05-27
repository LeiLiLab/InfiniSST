#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
MEDICINE_LAUNCHER="${MEDICINE_LAUNCHER_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260522__medicine_abbrev_restored_norag_streaming_batched_aries67.sh}"
CONDA_PREFIX="${CONDA_PREFIX_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv}"
FBK_FAIRSEQ_ROOT="${FBK_FAIRSEQ_ROOT_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/FBK-fairseq}"
MWERSEGMENTER_ROOT="${MWERSEGMENTER_ROOT_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/mwerSegmenter}"
GLOSSARY="${GLOSSARY_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_hard_manual_glossary_streamlaal_20260524.json}"

RUN_STAMP="${RUN_STAMP_OVERRIDE:-20260524T0610}"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs}"
OUT_ROOT_PREFIX="${OUT_ROOT_PREFIX_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260524_zh}"
CACHE_ROOT="${CACHE_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/cache/medicine_norag_zh_aries_20260524T0610}"
TMP_ROOT="${TMP_ROOT_OVERRIDE:-/dev/shm}"
WAIT_SECONDS="${WAIT_SECONDS_OVERRIDE:-60}"
MAX_POLLS="${MAX_POLLS_OVERRIDE:-720}"
GPU_BUSY_THRESHOLD_MIB="${GPU_BUSY_THRESHOLD_MIB_OVERRIDE:-1000}"
FORCE_RERUN="${FORCE_RERUN_OVERRIDE:-0}"

LM1_SAMPLES="${LM1_SAMPLES_OVERRIDE:-404 545006 596001 605000 606}"
LM3_SAMPLES="${LM3_SAMPLES_OVERRIDE:-404 545006 596001 605000 606}"
LM4_SAMPLES="${LM4_SAMPLES_OVERRIDE:-404 545006 596001 606}"

export CONDA_PREFIX
export FBK_FAIRSEQ_ROOT
export MWERSEGMENTER_ROOT
export STREAM_LAAL_TOOL="${FBK_FAIRSEQ_ROOT}/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py"
export PATH="${CONDA_PREFIX}/bin:${MWERSEGMENTER_ROOT}:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${LOG_ROOT}" "${CACHE_ROOT}"

gpu_pair_is_free() {
  local gpu_csv="$1"
  local gpu_comma="${gpu_csv//:/,}"
  local gpu
  local used
  IFS=',' read -r -a gpu_ids <<< "${gpu_comma}"
  for gpu in "${gpu_ids[@]}"; do
    used="$(
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
        awk -F, -v target="${gpu}" '$1 ~ "^[[:space:]]*" target "[[:space:]]*$" {gsub(/ /, "", $2); print $2}'
    )"
    if [[ -z "${used}" || "${used}" -gt "${GPU_BUSY_THRESHOLD_MIB}" ]]; then
      return 1
    fi
  done
  return 0
}

segment_for_lm() {
  case "$1" in
    1) printf '0.96' ;;
    3) printf '2.88' ;;
    4) printf '3.84' ;;
    *) echo "[ERROR] unsupported lm=$1" >&2; return 2 ;;
  esac
}

setting_dir_for() {
  local output_base="$1"
  local lm="$2"
  local cs
  cs="$(segment_for_lm "${lm}")"
  printf '%s/zh/gigaspeech-zh-s_origin-bsz4_gstrict_fixed_medicine_glossary_abbrev_restored__medicine5_cs%s_hs0.48_lm%s_k210_k110_th0p0' \
    "${output_base}" "${cs}" "${lm}"
}

run_generation() {
  local lm="$1"
  local gpu_csv="$2"
  local port="$3"
  local tag="$4"
  local samples="$5"
  local output_base="${OUT_ROOT_PREFIX}_lm${lm}_${tag}"
  local cache_base="${CACHE_ROOT}/lm${lm}_${tag}"
  local tmp_dir="${TMP_ROOT}/jxzh${lm}_${tag}"
  local out_log="${LOG_ROOT}/${RUN_STAMP}_medicine_norag_abbrev_restored_zh_lm${lm}_${tag}.out"
  local err_log="${LOG_ROOT}/${RUN_STAMP}_medicine_norag_abbrev_restored_zh_lm${lm}_${tag}.err"
  local pid_file="${LOG_ROOT}/${RUN_STAMP}_medicine_norag_abbrev_restored_zh_lm${lm}_${tag}.inner.pid"
  local out_path_file="${LOG_ROOT}/${RUN_STAMP}_medicine_norag_abbrev_restored_zh_lm${lm}_${tag}.output_path"

  if [[ -e "${output_base}" ]]; then
    echo "[ERROR] output_base already exists: ${output_base}" >&2
    return 3
  fi

  if ! gpu_pair_is_free "${gpu_csv}"; then
    echo "[ERROR] requested GPU pair is not free: ${gpu_csv}" >&2
    return 4
  fi

  mkdir -p "${tmp_dir}" "${cache_base}/xdg" "${cache_base}/triton" "${cache_base}/torchinductor" "${cache_base}/cuda"
  printf '%s\n' "${output_base}" > "${out_path_file}"

  (
    cd "${ROOT_DIR}"
    export TMPDIR="${tmp_dir}"
    export XDG_CACHE_HOME="${cache_base}/xdg"
    export TRITON_CACHE_DIR="${cache_base}/triton"
    export TORCHINDUCTOR_CACHE_DIR="${cache_base}/torchinductor"
    export CUDA_CACHE_PATH="${cache_base}/cuda"
    MASTER_PORT="${port}" \
    LANGS_OVERRIDE="zh" \
    TARGET_LMS_OVERRIDE="${lm}" \
    TARGET_SAMPLES_OVERRIDE="${samples}" \
    CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV="${gpu_csv}" \
    OUTPUT_BASE_OVERRIDE="${output_base}" \
    FORCE_RERUN_OVERRIDE="${FORCE_RERUN}" \
    bash "${MEDICINE_LAUNCHER}"
  ) >"${out_log}" 2>"${err_log}" &
  echo "$!" > "${pid_file}"
  printf '[SUBMITTED] lang=zh lm=%s samples="%s" gpu_csv=%s pid=%s output_base=%s tmp_dir=%s cache_base=%s out_log=%s err_log=%s\n' \
    "${lm}" "${samples}" "${gpu_csv}" "$(cat "${pid_file}")" "${output_base}" "${tmp_dir}" "${cache_base}" "${out_log}" "${err_log}"
}

start_lm4_when_free() {
  local pair
  for _ in $(seq 1 "${MAX_POLLS}"); do
    for pair in "2:3" "6:7" "0:1"; do
      if gpu_pair_is_free "${pair}"; then
        local tag="no605000_aries${pair//:/}"
        local output_base="${OUT_ROOT_PREFIX}_lm4_${tag}"
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting zh lm=4 on free GPU pair ${pair}"
        run_generation 4 "${pair}" 20644 "${tag}" "${LM4_SAMPLES}"
        run_post_eval 4 "${output_base}"
        return 0
      fi
    done
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for a free pair for zh lm=4; tried 2:3 6:7 0:1"
    sleep "${WAIT_SECONDS}"
  done
  echo "[ERROR] timed out waiting for a free GPU pair for zh lm=4" >&2
  return 2
}

wait_generation_success() {
  local lm="$1"
  local output_base="$2"
  local setting
  setting="$(setting_dir_for "${output_base}" "${lm}")"
  for _ in $(seq 1 "${MAX_POLLS}"); do
    local ts
    local inst_lines="missing"
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    [[ -f "${setting}/instances.log" ]] && inst_lines="$(wc -l < "${setting}/instances.log")"
    echo "[${ts}] wait generation zh lm=${lm} output=${output_base} instances=${inst_lines}"
    if [[ -s "${output_base}/timing.tsv" ]] && awk -F '\t' 'NR>1 && $5 == "success" {found=1} END {exit found?0:1}' "${output_base}/timing.tsv"; then
      if [[ ! -s "${setting}/instances.log" ]]; then
        echo "[ERROR] timing success but missing instances.log: ${setting}/instances.log" >&2
        return 3
      fi
      if [[ "${lm}" == "4" ]] && grep -q $'\t605000\t' "${output_base}/zh/__medicine_inputs__/combined/medicine5.sample_map.tsv"; then
        echo "[ERROR] lm=4 combined input unexpectedly includes sample 605000" >&2
        return 3
      fi
      return 0
    fi
    if [[ -s "${output_base}/timing.tsv" ]] && awk -F '\t' 'NR>1 && $5 == "failed" {found=1} END {exit found?0:1}' "${output_base}/timing.tsv"; then
      echo "[ERROR] generation failed for zh lm=${lm}: ${output_base}/timing.tsv" >&2
      return 2
    fi
    sleep "${WAIT_SECONDS}"
  done
  echo "[ERROR] generation timed out for zh lm=${lm}: ${output_base}" >&2
  return 2
}

run_post_eval() {
  local lm="$1"
  local output_base="$2"
  local setting
  local combined
  setting="$(setting_dir_for "${output_base}" "${lm}")"
  combined="${output_base}/zh/__medicine_inputs__/combined"

  wait_generation_success "${lm}" "${output_base}"

  if [[ -s "${setting}/eval_results_streamlaal_term.hard_llm_manual_check.tsv" ]]; then
    echo "[INFO] zh lm=${lm} hard-term eval already exists; skip"
    return 0
  fi

  cd "${ROOT_DIR}"
  "${CONDA_PREFIX}/bin/python" documents/code/offline_sst_eval/offline_streamlaal_eval.py \
    --mode acl6060 \
    --instances-log "${setting}/instances.log" \
    --lang-code zh \
    --source-file "${combined}/medicine5.source_text.en.sentences.txt" \
    --ref-file "${combined}/medicine5.ref.zh.sentences.txt" \
    --audio-yaml "${combined}/medicine5.audio.yaml" \
    --glossary-acl6060 "${GLOSSARY}" \
    --fbk-fairseq-root "${FBK_FAIRSEQ_ROOT}" \
    --term-fcr-policy source_ref_negative_sentence \
    --output-tsv "${setting}/eval_results_streamlaal_term.hard_llm_manual_check.tsv" \
    --output-log "${setting}/post_eval_streamlaal_term.hard_llm_manual_check.log" \
    --work-dir "${setting}/work_streamlaal_term.hard_llm_manual_check" \
    --term-mismatch-examples 20

  "${CONDA_PREFIX}/bin/python" documents/code/simuleval/export_streamlaal_term_misses.py \
    --instances-log "${setting}/instances.log" \
    --reference "${combined}/medicine5.ref.zh.sentences.txt" \
    --source-reference "${combined}/medicine5.source_text.en.sentences.txt" \
    --audio-yaml "${combined}/medicine5.audio.yaml" \
    --glossary "${GLOSSARY}" \
    --lang-code zh \
    --stream-laal-tool "${STREAM_LAAL_TOOL}" \
    --mwersegmenter-root "${MWERSEGMENTER_ROOT}" \
    --output-misses "${setting}/term_misses.hard_llm_manual_check.zh_lm${lm}.tsv" \
    --output-summary "${setting}/term_miss_summary.hard_llm_manual_check.zh_lm${lm}.tsv" \
    --output-normalized-glossary "${setting}/hard_medicine_glossary.streamlaal_dict.hard_llm_manual_check.json"

  echo "[DONE] zh lm=${lm} hard-term post-eval"
}

wait_path_file_and_post_eval() {
  local lm="$1"
  local out_path_file="$2"
  for _ in $(seq 1 "${MAX_POLLS}"); do
    if [[ -s "${out_path_file}" ]]; then
      run_post_eval "${lm}" "$(cat "${out_path_file}")"
      return 0
    fi
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for output path file: ${out_path_file}"
    sleep "${WAIT_SECONDS}"
  done
  echo "[ERROR] timed out waiting for output path file: ${out_path_file}" >&2
  return 2
}

notify_done() {
  local status="$1"
  local msg="Codex finished: medicine zh no-RAG baseline ${status}; logs=${LOG_ROOT}/${RUN_STAMP}_medicine_norag_abbrev_restored_zh_lm13_lm4no605000_aries.wrapper.out"
  if command -v "${HOME}/bin/codex-notify" >/dev/null 2>&1; then
    "${HOME}/bin/codex-notify" --delay 8 --detach --workspace "${ROOT_DIR}" "${msg}" || true
  fi
}

main() {
  echo "[START] $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) run_stamp=${RUN_STAMP}"
  echo "[CONFIG] lm1_samples=${LM1_SAMPLES}"
  echo "[CONFIG] lm3_samples=${LM3_SAMPLES}"
  echo "[CONFIG] lm4_samples=${LM4_SAMPLES}"
  echo "[CONFIG] force_rerun=${FORCE_RERUN}"
  echo "[CONFIG] cache_root=${CACHE_ROOT}"

  run_generation 1 "0:1" 20641 "aries01" "${LM1_SAMPLES}"
  run_generation 3 "6:7" 20643 "aries67" "${LM3_SAMPLES}"

  start_lm4_when_free &
  local lm4_watcher_pid="$!"

  wait_path_file_and_post_eval 1 "${LOG_ROOT}/${RUN_STAMP}_medicine_norag_abbrev_restored_zh_lm1_aries01.output_path" &
  local post1_pid="$!"
  wait_path_file_and_post_eval 3 "${LOG_ROOT}/${RUN_STAMP}_medicine_norag_abbrev_restored_zh_lm3_aries67.output_path" &
  local post3_pid="$!"

  local rc=0
  wait "${lm4_watcher_pid}" || rc=1
  wait "${post1_pid}" || rc=1
  wait "${post3_pid}" || rc=1

  if [[ "${rc}" -eq 0 ]]; then
    echo "[ALL DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    notify_done "success"
  else
    echo "[FAILED] $(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
    notify_done "failed"
  fi
  return "${rc}"
}

main "$@"
