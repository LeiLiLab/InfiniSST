#!/usr/bin/env bash
set -euo pipefail

# Submit zh main-result one-setting jobs for:
#   5 papers x 4 lm x 3 glossaries x 2 tau = 120 jobs.
#
# Usage:
#   MODE=smoke bash documents/code/simuleval/submit_acl_zh_main_v2r32_sweep.sh
#   MODE=full  bash documents/code/simuleval/submit_acl_zh_main_v2r32_sweep.sh
#
# The submitter creates four serial lanes by default:
#   aries GPUs 0,1,2 ; aries GPUs 3,4,5 ; taurus GPUs 0,1,2 ; taurus GPUs 3,4,5
# Each lane runs one 3-GPU setting at a time. Lanes run concurrently.

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
JOB_SCRIPT="${JOB_SCRIPT:-${ROOT_DIR}/documents/code/simuleval/run_acl_zh_main_v2r32_one_setting.sh}"

MODE="${MODE:-smoke}"  # smoke | full
DRY_RUN="${DRY_RUN:-0}"

OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/acl_main_zh_v2_r32_srcgated_no_utterance}"
SMOKE_OUTPUT_BASE="${SMOKE_OUTPUT_BASE_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/acl_main_zh_v2_r32_srcgated_no_utterance_smoke}"
MODEL_NAME="${MODEL_NAME_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/speech_llm_tcm_filtered_wiki100kgt_tau075_v2_sourcefinal_gtzh/keep1.0_r32/v0-20260507-103419-hf}"
RAG_MODEL_PATH="${RAG_MODEL_PATH_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_ntcm_final_v3_n64_p1n4_s2000_aries-8gpu_smallest_dense_smoke4650_best.pt}"

PAPERS=(2022.acl-long.110 2022.acl-long.117 2022.acl-long.268 2022.acl-long.367 2022.acl-long.590)
LMS=(1 2 3 4)
GLOSSARIES=(raw gs1k gs10k)
TAUS=(0.0 0.75)

# Format: partition|gpu_csv|lane_name
LANES=(
  "aries|0:1:2|a012"
  "aries|3:4:5|a345"
  "taurus|0:1:2|t012"
  "taurus|3:4:5|t345"
)

if [[ "${MODE}" == "smoke" ]]; then
  PAPERS=(2022.acl-long.110)
  LMS=(1)
  GLOSSARIES=(raw)
  TAUS=(0.0)
  OUTPUT_BASE="${SMOKE_OUTPUT_BASE}"
  MAX_SENTENCES_OVERRIDE="${MAX_SENTENCES_OVERRIDE:-6}"
  DENSITY_TAG_OVERRIDE="${DENSITY_TAG_OVERRIDE:-aclmain_v2r32_smoke}"
elif [[ "${MODE}" == "full" ]]; then
  MAX_SENTENCES_OVERRIDE="${MAX_SENTENCES_OVERRIDE:-0}"
  DENSITY_TAG_OVERRIDE="${DENSITY_TAG_OVERRIDE:-aclmain_v2r32}"
else
  echo "[ERROR] MODE must be smoke or full, got ${MODE}" >&2
  exit 2
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

submit_one() {
  local partition="$1" gpu_csv="$2" lane_name="$3" dependency="$4" paper="$5" lm="$6" glossary="$7" tau="$8"
  local job_name="mzh_${glossary}_l${lm}_t${tau/./}_${paper##*.}"
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
    --job-name="${job_name}" \
    "${dep_arg[@]}" \
    --export="${export_vars}" \
    "${JOB_SCRIPT}"
}

for tau in "${TAUS[@]}"; do
  for lm in "${LMS[@]}"; do
    for glossary in "${GLOSSARIES[@]}"; do
      for paper in "${PAPERS[@]}"; do
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

echo "[INFO] MODE=${MODE} OUTPUT_BASE=${OUTPUT_BASE}"
echo "[INFO] Submitted ${task_idx} task(s)."
if [[ "${DRY_RUN}" != "1" && ${#submitted[@]} -gt 0 ]]; then
  joined="$(IFS=,; echo "${submitted[*]}")"
  squeue -j "${joined}" -o "%.18i %.9P %.8T %.10M %.6D %R %.30j" || true
fi
