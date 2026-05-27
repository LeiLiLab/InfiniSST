#!/usr/bin/env bash
set -euo pipefail

# Quick tagged-ACL rank readout for old new_v3 r64/a128 checkpoints.
# Runs serially on one 2-GPU pair.  It waits for HF exports if needed.

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260520__tagged_acl_origin_bsz4_tau073_sweep_taurus45269.sh}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)}"
GPU_PAIR="${GPU_PAIR:-4,5}"
WAIT_SECONDS="${WAIT_SECONDS_OVERRIDE:-21600}"
POLL_SECONDS="${POLL_SECONDS_OVERRIDE:-60}"

Q159_MODEL="${Q159_MODEL_OVERRIDE:-/mnt/data7/jiaxuanluo/slm/old_newv3_rank_ablation/q159wce4_newv3_r64a128_full-hf}"
RJ1_MODEL="${RJ1_MODEL_OVERRIDE:-/mnt/data7/jiaxuanluo/slm/old_newv3_rank_ablation/rj1v1p7r_newv3_random_r64a128-hf}"
EXTRACTED_110_GLOSSARY="${EXTRACTED_110_GLOSSARY_OVERRIDE:-${ROOT_DIR}/documents/data/data_pre/extracted_glossaries_by_paper/extracted_glossary__2022.acl-long.110.json}"

OUT_ROOT="${OUT_ROOT:-/mnt/aries/data7/jiaxuanluo/slm/tagged_acl_old_newv3_r64_rank_quick_zh_lm2_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_old_newv3_r64_rank_quick_${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/aries/data7/jiaxuanluo/slm/maxsim_index_cache}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260522__tagged_acl_old_newv3_r64_rank_quick_zh_lm2.md}"

wait_for_hf_dir() {
  local model_dir="$1"
  local start now elapsed
  start="$(date +%s)"
  while true; do
    if [[ -f "${model_dir}/config.json" && -f "${model_dir}/generation_config.json" ]] \
       && compgen -G "${model_dir}/*.safetensors" >/dev/null; then
      echo "[READY] HF checkpoint: ${model_dir}"
      return 0
    fi
    now="$(date +%s)"
    elapsed=$((now - start))
    if (( elapsed >= WAIT_SECONDS )); then
      echo "[ERROR] Timed out waiting for HF checkpoint: ${model_dir}" >&2
      return 3
    fi
    echo "[WAIT] ${model_dir} (${elapsed}s/${WAIT_SECONDS}s)"
    sleep "${POLL_SECONDS}"
  done
}

run_base() {
  local variant="$1"
  local model_dir="$2"
  local run_kind="$3"
  local run_granularity="$4"
  local glossary_kinds="$5"
  local papers="$6"

  local output_base="${OUT_ROOT}/${variant}/${run_kind}"
  local input_root="${OUT_ROOT}/__inputs__/${variant}/${run_kind}"
  local summary_dir="${output_base}/__summary__"
  local log_dir="${LOG_ROOT}/${variant}/${run_kind}"
  mkdir -p "${output_base}" "${input_root}" "${summary_dir}" "${log_dir}"

  echo "[RUN] variant=${variant} kind=${run_kind} model=${model_dir}"
  env \
    ROOT_DIR="${ROOT_DIR}" \
    RUN_STAMP="${RUN_STAMP}_${variant}_${run_kind}" \
    MODE=full \
    RUN_GRANULARITY="${run_granularity}" \
    HOLD_JOB_ID=0 \
    INSIDE_HOLD_STEP=1 \
    MAX_PARALLEL_OVERRIDE=1 \
    LANGS_OVERRIDE="zh" \
    LMS_OVERRIDE="2" \
    GLOSSARY_KINDS_OVERRIDE="${glossary_kinds}" \
    PAPERS_OVERRIDE="${papers}" \
    GPU_PAIRS_CSV_OVERRIDE="${GPU_PAIR}" \
    MODEL_NAME_OVERRIDE="${model_dir}" \
    TERM_MAP_FORMAT_OVERRIDE="plain" \
    EXTRACTED_GLOSSARY_OVERRIDE="${EXTRACTED_110_GLOSSARY}" \
    OUTPUT_BASE_OVERRIDE="${output_base}" \
    INPUT_ROOT_OVERRIDE="${input_root}" \
    LOG_DIR_OVERRIDE="${log_dir}" \
    SUMMARY_DIR_OVERRIDE="${summary_dir}" \
    INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}" \
    NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
    DENSITY_TAG_OVERRIDE="tagacl_${variant}_${run_kind}_tau073" \
    WANDB_RUN_PREFIX_OVERRIDE="${variant}_${run_kind}" \
    WANDB_EXPERIMENT_FAMILY_OVERRIDE="tagged_acl_old_newv3_r64_rank_quick" \
    WANDB_VARIANT_PREFIX_OVERRIDE="${variant}_${run_kind}" \
    WANDB_COMPUTE_TAG_OVERRIDE="compute:aries_direct" \
    TERM_FCR_POLICY_OVERRIDE="term_map_source_ref_negative_sentence" \
    bash "${BASE_LAUNCHER}" \
      > "${log_dir}/launcher.out" \
      2> "${log_dir}/launcher.err"
}

for p in "${BASE_LAUNCHER}" "${NOTES_FILE}" "${EXTRACTED_110_GLOSSARY}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
{
  echo "run_stamp=${RUN_STAMP}"
  echo "gpu_pair=${GPU_PAIR}"
  echo "q159_model=${Q159_MODEL}"
  echo "rj1_model=${RJ1_MODEL}"
  echo "extracted_110_glossary=${EXTRACTED_110_GLOSSARY}"
  echo "out_root=${OUT_ROOT}"
  echo "log_root=${LOG_ROOT}"
} | tee "${OUT_ROOT}/run_meta.txt"

wait_for_hf_dir "${Q159_MODEL}"
wait_for_hf_dir "${RJ1_MODEL}"

run_base "q159wce4_newv3_r64a128_full" "${Q159_MODEL}" "full_raw" "full_corpus" "raw" "all"
run_base "q159wce4_newv3_r64a128_full" "${Q159_MODEL}" "paper110_extracted" "per_paper" "extracted" "2022.acl-long.110"
run_base "rj1v1p7r_newv3_random_r64a128" "${RJ1_MODEL}" "full_raw" "full_corpus" "raw" "all"
run_base "rj1v1p7r_newv3_random_r64a128" "${RJ1_MODEL}" "paper110_extracted" "per_paper" "extracted" "2022.acl-long.110"

echo "[DONE] Output root: ${OUT_ROOT}"
