#!/usr/bin/env bash
set -euo pipefail

# Submit the paper-110 full-window one-paper comparison used for the historical
# baseline/new_v2 numbers. Defaults match the original one-line sbatch command.

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
SCRIPT="${SCRIPT:-${ROOT_DIR}/documents/code/simuleval/run_acl_onepaper_lm_raw1k10k_taurus.sh}"

BASE_MODEL="${BASE_MODEL:-/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4}"
NEW_MODEL="${NEW_MODEL:-/mnt/gemini/data2/jiaxuanluo/speech_llm_tcm_filtered_wiki100kgt_tau075_v2_sourcefinal_gtzh/keep1.0_r32/v0-20260507-103419-hf}"

TARGET_PAPER="${TARGET_PAPER:-2022.acl-long.110}"
TARGET_LM="${TARGET_LM:-1}"
PARTITION="${PARTITION:-taurus}"
GPU_CSV="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV:-5:6:7}"
WIN="${RAG_MAXSIM_WINDOWS_OVERRIDE:-2 3 4 5 6 7 8 10 12 16 20 24}"
RAG_TOP_K="${RAG_TOP_K_OVERRIDE:-10}"

OUT_ROOT="${OUT_ROOT:-/mnt/gemini/data2/jiaxuanluo}"
NEW_TAU0_OUT="${NEW_TAU0_OUT:-${OUT_ROOT}/acl_onepaper_lm1_raw1k10k_timeline_fullwin_tau0_tcmwiki100kgt_v2_slm}"
BASE_TAU0_OUT="${BASE_TAU0_OUT:-${OUT_ROOT}/acl_onepaper_lm1_raw1k10k_timeline_fullwin_tau0_v4ner_baseline}"
NEW_TAU075_OUT="${NEW_TAU075_OUT:-${OUT_ROOT}/acl_onepaper_lm1_raw1k10k_timeline_fullwin_tau075_tcmwiki100kgt_v2_slm}"
BASE_TAU075_OUT="${BASE_TAU075_OUT:-${OUT_ROOT}/acl_onepaper_lm1_raw1k10k_timeline_fullwin_tau075_v4ner_baseline}"

for p in "${SCRIPT}" "${BASE_MODEL}" "${NEW_MODEL}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

submit_one() {
  local dependency_arg="$1"
  local job_name="$2"
  local model="$3"
  local output_base="$4"
  local density_tag="$5"
  local tau="$6"

  sbatch --parsable ${dependency_arg} \
    --partition="${PARTITION}" \
    --job-name="${job_name}" \
    --export=ALL,MODEL_NAME_OVERRIDE="${model}",TARGET_PAPER="${TARGET_PAPER}",TARGET_LM="${TARGET_LM}",OUTPUT_BASE_OVERRIDE="${output_base}",DENSITY_TAG_OVERRIDE="${density_tag}",RAG_SCORE_THRESHOLD_OVERRIDE="${tau}",RAG_STREAMING_MODE_OVERRIDE=timeline,RAG_MAXSIM_WINDOWS_OVERRIDE="${WIN}",RAG_MAXSIM_STRIDE_OVERRIDE=2,RAG_TOP_K_OVERRIDE="${RAG_TOP_K}",CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV="${GPU_CSV}",VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
    "${SCRIPT}"
}

j_new0="$(submit_one "" "fw110_newv2_t0" "${NEW_MODEL}" "${NEW_TAU0_OUT}" "aclone_tcmw100kgt_v2_timeline_fullwin_tau0" "0.0")"
j_base0="$(submit_one "--dependency=afterok:${j_new0}" "fw110_base_t0" "${BASE_MODEL}" "${BASE_TAU0_OUT}" "aclone_v4ner_timeline_fullwin_tau0" "0.0")"
j_new75="$(submit_one "--dependency=afterok:${j_base0}" "fw110_newv2_t75" "${NEW_MODEL}" "${NEW_TAU075_OUT}" "aclone_tcmw100kgt_v2_timeline_fullwin_tau075" "0.75")"
j_base75="$(submit_one "--dependency=afterok:${j_new75}" "fw110_base_t75" "${BASE_MODEL}" "${BASE_TAU075_OUT}" "aclone_v4ner_timeline_fullwin_tau075" "0.75")"

echo "NEW_TAU0_FULLWIN=${j_new0}"
echo "BASE_TAU0_FULLWIN=${j_base0}"
echo "NEW_TAU075_FULLWIN=${j_new75}"
echo "BASE_TAU075_FULLWIN=${j_base75}"

squeue -j "${j_new0},${j_base0},${j_new75},${j_base75}" -o "%.18i %.9P %.8T %.10M %.6D %R %.30j"
