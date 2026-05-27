#!/usr/bin/env bash
set -euo pipefail

# Submit origin Speech-LLM zh main-result one-setting jobs:
#   5 papers x 4 lm x 3 glossaries x 2 tau = 120 jobs.
# All jobs are submitted to Aries. By default they run as one serial 3-GPU lane
# to avoid the vLLM /dev/shm conflicts observed when two vLLM lanes share a node.

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
JOB_SCRIPT="${JOB_SCRIPT:-${ROOT_DIR}/documents/code/simuleval/run_acl_zh_main_origin_bsz4_one_setting_taurus.sh}"

DRY_RUN="${DRY_RUN:-0}"
OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/acl_main_zh_origin_bsz4_srcgated}"
MODEL_NAME="${MODEL_NAME_OVERRIDE:-/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_origin-bsz4}"
RAG_MODEL_PATH="${RAG_MODEL_PATH_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_ntcm_final_v3_n64_p1n4_s2000_aries-8gpu_smallest_dense_smoke4650_best.pt}"
DENSITY_TAG_OVERRIDE="${DENSITY_TAG_OVERRIDE:-aclmain_origin_bsz4}"
MAX_SENTENCES_OVERRIDE="${MAX_SENTENCES_OVERRIDE:-0}"
SWEEP_PARTITION="${SWEEP_PARTITION:-aries}"
GPU_CSV="${GPU_CSV:-0:1:2}"
LANE_NAME="${LANE_NAME:-a012}"
COMPUTE_TAG_OVERRIDE="${COMPUTE_TAG_OVERRIDE:-${SWEEP_PARTITION}3}"
SBATCH_GRES="${SBATCH_GRES:-gpu:3}"
SBATCH_MEM="${SBATCH_MEM:-256G}"
CLEAN_SHM_OVERRIDE="${CLEAN_SHM_OVERRIDE:-0}"
CLEAN_OUTPUT_DIR_OVERRIDE="${CLEAN_OUTPUT_DIR_OVERRIDE:-0}"
SUBMIT_ONLY_INCOMPLETE="${SUBMIT_ONLY_INCOMPLETE:-0}"
LANES_OVERRIDE="${LANES_OVERRIDE:-}"

PAPERS=(2022.acl-long.110 2022.acl-long.117 2022.acl-long.268 2022.acl-long.367 2022.acl-long.590)
LMS=(1 2 3 4)
GLOSSARIES=(raw gs1k gs10k)
TAUS=(0.0 0.75)

# Format: partition|gpu_csv|lane_name. Override with semicolon-separated entries,
# e.g. LANES_OVERRIDE='aries|0:1|a01;aries|2:3|a23'.
if [[ -n "${LANES_OVERRIDE}" ]]; then
  IFS=';' read -r -a LANES <<< "${LANES_OVERRIDE}"
else
  LANES=(
    "${SWEEP_PARTITION}|${GPU_CSV}|${LANE_NAME}"
  )
fi

for p in "${JOB_SCRIPT}" "${MODEL_NAME}" "${RAG_MODEL_PATH}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

declare -a lane_last_job
for ((i = 0; i < ${#LANES[@]}; i++)); do
  lane_last_job[$i]=""
done

task_idx=0
submitted=()
skipped_complete=0

glossary_tag_for_kind() {
  local paper="$1" glossary="$2"
  case "${glossary}" in
    raw) echo "extracted_glossary__${paper}" ;;
    gs1k) echo "glossary_acl6060_gt_union_gs1000" ;;
    gs10k) echo "glossary_acl6060_gt_union_gs10000" ;;
    *)
      echo "[ERROR] Unsupported glossary=${glossary}" >&2
      return 2
      ;;
  esac
}

eval_tsv_for_setting() {
  local paper="$1" lm="$2" glossary="$3" tau="$4"
  local glossary_tag
  glossary_tag="$(glossary_tag_for_kind "${paper}" "${glossary}")"
  echo "${OUTPUT_BASE}/zh/d${DENSITY_TAG_OVERRIDE}_lm${lm}_k10_th${tau}_g${glossary_tag}_pp${paper}/eval_results.tsv"
}

setting_is_complete() {
  local eval_tsv="$1"
  [[ -f "${eval_tsv}" ]] && [[ -s "${eval_tsv}" ]] && ! grep -q $'\tN/A\tN/A\tN/A' "${eval_tsv}"
}

submit_one() {
  local partition="$1" gpu_csv="$2" lane_name="$3" dependency="$4" paper="$5" lm="$6" glossary="$7" tau="$8"
  local tau_name="${tau/./}"
  local job_name="ozh_${glossary}_l${lm}_t${tau_name}_${paper##*.}"
  local export_vars
  export_vars="ALL"
  export_vars+=",MODEL_NAME_OVERRIDE=${MODEL_NAME}"
  export_vars+=",RAG_MODEL_PATH_OVERRIDE=${RAG_MODEL_PATH}"
  export_vars+=",OUTPUT_BASE_OVERRIDE=${OUTPUT_BASE}"
  export_vars+=",TARGET_PAPER=${paper}"
  export_vars+=",TARGET_LM=${lm}"
  export_vars+=",GLOSSARY_KIND=${glossary}"
  export_vars+=",RAG_SCORE_THRESHOLD_OVERRIDE=${tau}"
  export_vars+=",RAG_STREAMING_MODE_OVERRIDE=timeline"
  export_vars+=",RAG_MAXSIM_WINDOWS_OVERRIDE=2 3 4 5 6 7 8 10 12 16 20 24"
  export_vars+=",RAG_MAXSIM_STRIDE_OVERRIDE=2"
  export_vars+=",RAG_TOP_K_OVERRIDE=10"
  export_vars+=",CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV=${gpu_csv}"
  export_vars+=",DENSITY_TAG_OVERRIDE=${DENSITY_TAG_OVERRIDE}"
  export_vars+=",MAX_SENTENCES_OVERRIDE=${MAX_SENTENCES_OVERRIDE}"
  export_vars+=",CLEAN_SHM_OVERRIDE=${CLEAN_SHM_OVERRIDE}"
  export_vars+=",CLEAN_OUTPUT_DIR_OVERRIDE=${CLEAN_OUTPUT_DIR_OVERRIDE}"
  export_vars+=",COMPUTE_TAG_OVERRIDE=${COMPUTE_TAG_OVERRIDE}"
  export_vars+=",RUN_NAME_PREFIX_OVERRIDE=${RUN_NAME_PREFIX_OVERRIDE:-origin_bsz4}"
  export_vars+=",VARIANT_TAG_OVERRIDE=${VARIANT_TAG_OVERRIDE:-origin_bsz4}"
  export_vars+=",TASK_TAG_OVERRIDE=${TASK_TAG_OVERRIDE:-eval}"
  export_vars+=",DATA_TAG_OVERRIDE=${DATA_TAG_OVERRIDE:-acl6060_main_zh}"
  export_vars+=",VLLM_DISABLE_CUSTOM_ALL_REDUCE=1"

  local dep_arg=()
  if [[ -n "${dependency}" ]]; then
    dep_arg=(--dependency="afterany:${dependency}")
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN lane=${lane_name} partition=${partition} dep=${dependency:-none} paper=${paper} lm=${lm} glossary=${glossary} tau=${tau}"
    return 0
  fi

  sbatch --parsable \
    --partition="${partition}" \
    --gres="${SBATCH_GRES}" \
    --mem="${SBATCH_MEM}" \
    --job-name="${job_name}" \
    "${dep_arg[@]}" \
    --export="${export_vars}" \
    "${JOB_SCRIPT}"
}

for tau in "${TAUS[@]}"; do
  for lm in "${LMS[@]}"; do
    for glossary in "${GLOSSARIES[@]}"; do
      for paper in "${PAPERS[@]}"; do
        eval_tsv="$(eval_tsv_for_setting "${paper}" "${lm}" "${glossary}" "${tau}")"
        if [[ "${SUBMIT_ONLY_INCOMPLETE}" == "1" ]] && setting_is_complete "${eval_tsv}"; then
          echo "SKIP complete paper=${paper} lm=${lm} glossary=${glossary} tau=${tau} eval=${eval_tsv}"
          skipped_complete=$((skipped_complete + 1))
          continue
        fi
        lane_idx=$((task_idx % ${#LANES[@]}))
        IFS='|' read -r partition gpu_csv lane_name <<< "${LANES[$lane_idx]}"
        dependency="${lane_last_job[$lane_idx]}"
        job_id="$(submit_one "${partition}" "${gpu_csv}" "${lane_name}" "${dependency}" "${paper}" "${lm}" "${glossary}" "${tau}")"
        if [[ "${DRY_RUN}" == "1" ]]; then
          job_id="dry${task_idx}"
        else
          echo "SUBMITTED lane=${lane_name} job=${job_id} dep=${dependency:-none} paper=${paper} lm=${lm} glossary=${glossary} tau=${tau}"
          submitted+=("${job_id}")
        fi
        lane_last_job[$lane_idx]="${job_id}"
        task_idx=$((task_idx + 1))
      done
    done
  done
done

echo "[INFO] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] Submitted ${task_idx} task(s)."
echo "[INFO] Skipped complete ${skipped_complete} task(s)."
if [[ "${DRY_RUN}" != "1" && ${#submitted[@]} -gt 0 ]]; then
  joined="$(IFS=,; echo "${submitted[*]}")"
  squeue -j "${joined}" -o "%.18i %.9P %.8T %.10M %.6D %R %.30j" || true
fi
