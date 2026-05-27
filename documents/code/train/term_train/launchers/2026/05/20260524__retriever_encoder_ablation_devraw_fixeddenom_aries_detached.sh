#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${SSH_HOST:-localhost}"
SSH_PORT="${SSH_PORT:-20042}"
SSH_KNOWN_HOSTS="${SSH_KNOWN_HOSTS:-/tmp/codex_aries_20042_known_hosts}"
REMOTE_REPO="${REMOTE_REPO:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/mnt/gemini/data1/jiaxuanluo/logs/retriever_encoder_ablation_aries_20260524T1725Z}"
REMOTE_LAUNCHER="${REMOTE_REPO}/documents/code/train/term_train/launchers/2026/05/20260524__retriever_encoder_ablation_devraw_fixeddenom_eval.sh"
REMOTE_NOTES_DIR="${REMOTE_REPO}/documents/code/train/term_train/notes/2026/05"

ssh_base=(
  ssh
  -p "${SSH_PORT}"
  -o BatchMode=yes
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile="${SSH_KNOWN_HOSTS}"
  "${SSH_HOST}"
)

start_case() {
  local model_tag="$1"
  local encoder_config="$2"
  local resume="$3"
  local notes_file="$4"
  local run_stamp="$5"
  local gpu_id="$6"
  local master_port="$7"

  "${ssh_base[@]}" bash -s -- \
    "${model_tag}" \
    "${encoder_config}" \
    "${resume}" \
    "${notes_file}" \
    "${run_stamp}" \
    "${gpu_id}" \
    "${master_port}" \
    "${REMOTE_REPO}" \
    "${REMOTE_LAUNCHER}" \
    "${REMOTE_LOG_DIR}" <<'REMOTE'
set -euo pipefail

MODEL_TAG="$1"
ENCODER_CONFIG="$2"
RESUME="$3"
NOTES_FILE="$4"
RUN_STAMP="$5"
GPU_ID="$6"
MASTER_PORT="$7"
REMOTE_REPO="$8"
REMOTE_LAUNCHER="$9"
REMOTE_LOG_DIR="${10}"

mkdir -p "${REMOTE_LOG_DIR}"
cd "${REMOTE_REPO}"

if [ ! -f "${REMOTE_LAUNCHER}" ]; then
  echo "[FATAL] launcher not found: ${REMOTE_LAUNCHER}" >&2
  exit 1
fi
if [ ! -f "${RESUME}" ]; then
  echo "[FATAL] checkpoint not found: ${RESUME}" >&2
  exit 1
fi
if [ ! -f "${NOTES_FILE}" ]; then
  echo "[FATAL] notes file not found: ${NOTES_FILE}" >&2
  exit 1
fi

OUT="${REMOTE_LOG_DIR}/${MODEL_TAG}.out"
ERR="${REMOTE_LOG_DIR}/${MODEL_TAG}.err"
PID_FILE="${REMOTE_LOG_DIR}/${MODEL_TAG}.pid"

setsid env \
  MODEL_TAG="${MODEL_TAG}" \
  ENCODER_CONFIG="${ENCODER_CONFIG}" \
  RESUME="${RESUME}" \
  NOTES_FILE="${NOTES_FILE}" \
  RUN_STAMP="${RUN_STAMP}" \
  NUM_GPUS=1 \
  PER_GPU_BATCH=1 \
  BATCH_SIZE=1 \
  SELECT_CLEAN_GPUS=true \
  CUDA_DEVICE_LIST="${GPU_ID}" \
  MASTER_PORT="${MASTER_PORT}" \
  LOCAL_TMP_DIR="/tmp/jx_encab_${MODEL_TAG}_${RUN_STAMP}" \
  EXTRA_WANDB_TAGS="variant:encab_${MODEL_TAG}_devraw_gs100k compute:aries-detached ablation:encoder protocol:devraw-fixeddenom readout:dev-only launcher:port20042" \
  bash "${REMOTE_LAUNCHER}" >"${OUT}" 2>"${ERR}" < /dev/null &

pid="$!"
echo "${pid}" > "${PID_FILE}"
echo "${MODEL_TAG} pid=${pid} gpu=${GPU_ID} out=${OUT} err=${ERR}"
REMOTE
}

CASE_FILTER="${CASE_FILTER:-}"
should_start() {
  local model_tag="$1"
  if [ -z "${CASE_FILTER}" ]; then
    return 0
  fi
  [[ " ${CASE_FILTER} " == *" ${model_tag} "* ]]
}

main_resume="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt"
text_resume="/mnt/taurus/data/siqiouyang/runs/infinisst_rag/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_txt_e5l_bs8k_gc256_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_dev100Tau1_eval100_babel8_best.pt"
audio_resume="/mnt/taurus/data/siqiouyang/runs/infinisst_rag/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_aud_wavl_fu_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_dev100Tau1_eval100_babel8_best.pt"

if should_start main_q3o_bgem3; then
start_case \
  main_q3o_bgem3 \
  main_q3o_bgem3 \
  "${main_resume}" \
  "${REMOTE_NOTES_DIR}/20260524__retriever_encoder_ablation_main_q3o_bgem3_devraw_fixeddenom.md" \
  encab_main_q3o_bgem3_devraw_ariesfast_20260524T1725Z \
  0 \
  30624
fi

if should_start text_e5; then
start_case \
  text_e5 \
  text_e5 \
  "${text_resume}" \
  "${REMOTE_NOTES_DIR}/20260524__retriever_encoder_ablation_text_e5_devraw_fixeddenom.md" \
  encab_text_e5_devraw_ariesfast_20260524T1726Z \
  1 \
  30625
fi

if should_start audio_wavlm; then
start_case \
  audio_wavlm \
  audio_wavlm \
  "${audio_resume}" \
  "${REMOTE_NOTES_DIR}/20260524__retriever_encoder_ablation_audio_wavlm_devraw_fixeddenom.md" \
  encab_audio_wavlm_devraw_ariesfast_20260524T1727Z \
  2 \
  30626
fi
