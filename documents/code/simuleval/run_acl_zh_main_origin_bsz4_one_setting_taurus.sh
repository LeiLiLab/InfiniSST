#!/usr/bin/env bash
#SBATCH --job-name=aclmain_origin
#SBATCH --partition=taurus
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_aclmain_origin.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_aclmain_origin.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
BASE_ONE_SETTING="${ROOT_DIR}/documents/code/simuleval/run_acl_zh_main_v2r32_one_setting.sh"
WANDB_LOGGER="${ROOT_DIR}/documents/code/offline_evaluation/wandb_eval_logger.py"
WANDB_PYTHON="${WANDB_PYTHON:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python}"
WANDB_HOME="${WANDB_HOME:-/mnt/taurus/home/jiaxuanluo}"

MODEL_NAME="${MODEL_NAME_OVERRIDE:-/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_origin-bsz4}"
RAG_MODEL_PATH="${RAG_MODEL_PATH_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_ntcm_final_v3_n64_p1n4_s2000_aries-8gpu_smallest_dense_smoke4650_best.pt}"
OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/acl_main_zh_origin_bsz4_srcgated}"
DENSITY_TAG="${DENSITY_TAG_OVERRIDE:-aclmain_origin_bsz4}"
COMPUTE_TAG="${COMPUTE_TAG_OVERRIDE:-${SLURM_JOB_PARTITION:-taurus}3}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX_OVERRIDE:-origin_bsz4}"
VARIANT_TAG="${VARIANT_TAG_OVERRIDE:-origin_bsz4}"
TASK_TAG="${TASK_TAG_OVERRIDE:-eval}"
DATA_TAG="${DATA_TAG_OVERRIDE:-acl6060_main_zh}"

TARGET_PAPER="${TARGET_PAPER:?Set TARGET_PAPER, e.g. 2022.acl-long.110}"
TARGET_LM="${TARGET_LM:?Set TARGET_LM, e.g. 1}"
GLOSSARY_KIND="${GLOSSARY_KIND:?Set GLOSSARY_KIND to raw|gs1k|gs10k}"
TAU="${RAG_SCORE_THRESHOLD_OVERRIDE:?Set RAG_SCORE_THRESHOLD_OVERRIDE to 0.0 or 0.75}"

case "${TAU}" in
  0|0.0)
    TAU_TAG="tau0"
    TAU_VALUE="0.0"
    NOTES_FILE="${ROOT_DIR}/documents/code/simuleval/notes_origin_bsz4_acl_main_zh_tau0.md"
    ;;
  0.75)
    TAU_TAG="tau075"
    TAU_VALUE="0.75"
    NOTES_FILE="${ROOT_DIR}/documents/code/simuleval/notes_origin_bsz4_acl_main_zh_tau075.md"
    ;;
  *)
    echo "[ERROR] Unsupported tau: ${TAU}" >&2
    exit 2
    ;;
esac

case "${GLOSSARY_KIND}" in
  raw)
    GLOSSARY_TAG="extracted_glossary__${TARGET_PAPER}"
    ;;
  gs1k)
    GLOSSARY_TAG="glossary_acl6060_gt_union_gs1000"
    ;;
  gs10k)
    GLOSSARY_TAG="glossary_acl6060_gt_union_gs10000"
    ;;
  *)
    echo "[ERROR] Unsupported GLOSSARY_KIND=${GLOSSARY_KIND}" >&2
    exit 2
    ;;
esac

for p in "${BASE_ONE_SETTING}" "${WANDB_LOGGER}" "${NOTES_FILE}" "${MODEL_NAME}" "${RAG_MODEL_PATH}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

echo "[ORIGIN_MAIN] paper=${TARGET_PAPER} lm=${TARGET_LM} glossary=${GLOSSARY_KIND} tau=${TAU_VALUE}"
echo "[ORIGIN_MAIN] model=${MODEL_NAME}"
echo "[ORIGIN_MAIN] rag=${RAG_MODEL_PATH}"
echo "[ORIGIN_MAIN] output_base=${OUTPUT_BASE}"
echo "[ORIGIN_MAIN] density=${DENSITY_TAG}"
echo "[ORIGIN_MAIN] compute_tag=${COMPUTE_TAG}"
echo "[ORIGIN_MAIN] run_name_prefix=${RUN_NAME_PREFIX} variant=${VARIANT_TAG} task=${TASK_TAG}"

MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
RAG_MODEL_PATH_OVERRIDE="${RAG_MODEL_PATH}" \
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
DENSITY_TAG_OVERRIDE="${DENSITY_TAG}" \
TARGET_PAPER="${TARGET_PAPER}" \
TARGET_LM="${TARGET_LM}" \
GLOSSARY_KIND="${GLOSSARY_KIND}" \
RAG_SCORE_THRESHOLD_OVERRIDE="${TAU_VALUE}" \
RAG_STREAMING_MODE_OVERRIDE="${RAG_STREAMING_MODE_OVERRIDE:-timeline}" \
RAG_MAXSIM_WINDOWS_OVERRIDE="${RAG_MAXSIM_WINDOWS_OVERRIDE:-2 3 4 5 6 7 8 10 12 16 20 24}" \
RAG_MAXSIM_STRIDE_OVERRIDE="${RAG_MAXSIM_STRIDE_OVERRIDE:-2}" \
RAG_TOP_K_OVERRIDE="${RAG_TOP_K_OVERRIDE:-10}" \
GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}" \
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV:-0:1:2}" \
MAX_SENTENCES_OVERRIDE="${MAX_SENTENCES_OVERRIDE:-0}" \
CLEAN_SHM_OVERRIDE="${CLEAN_SHM_OVERRIDE:-0}" \
CLEAN_OUTPUT_DIR_OVERRIDE="${CLEAN_OUTPUT_DIR_OVERRIDE:-0}" \
VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
bash "${BASE_ONE_SETTING}"

HOME="${WANDB_HOME}" \
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_HOME}/.config/wandb}" \
"${WANDB_PYTHON}" "${WANDB_LOGGER}" \
  --project simuleval_eval \
  --run-name "${RUN_NAME_PREFIX}__${TARGET_PAPER}__lm${TARGET_LM}__${TAU_TAG}__${GLOSSARY_KIND}" \
  --experiment-family acl_main_zh \
  --data-tag "${DATA_TAG}" \
  --task-tag "${TASK_TAG}" \
  --notes-file "${NOTES_FILE}" \
  --baseline-run-ids "3fic89wn" "djcp4rmt" \
  --extra-tags "variant:${VARIANT_TAG}" "compute:${COMPUTE_TAG}" "tau:${TAU_TAG}" "paper:${TARGET_PAPER}" "glossary:${GLOSSARY_KIND}" \
  --density "${DENSITY_TAG}" \
  --rag-top-k "${RAG_TOP_K_OVERRIDE:-10}" \
  --rag-score-threshold "${TAU_VALUE}" \
  --paper-id "${TARGET_PAPER}" \
  --output-base "${OUTPUT_BASE}" \
  --lang-code zh \
  --latency-multipliers "${TARGET_LM}" \
  --glossary-tag "${GLOSSARY_TAG}" \
  --model-name "${MODEL_NAME}" \
  --rag-model-path "${RAG_MODEL_PATH}" \
  --verdict "Logged origin_bsz4 ACL main zh SimulEval for ${TARGET_PAPER}, lm=${TARGET_LM}, tau=${TAU_VALUE}, glossary=${GLOSSARY_KIND}."

HOME="${WANDB_HOME}" \
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_HOME}/.config/wandb}" \
"${WANDB_PYTHON}" "${ROOT_DIR}/documents/code/general/wandb_tool.py" --project simuleval_eval db-sync \
  --family acl_main_zh --best-bundles --limit 120 || true

echo "[ALL DONE] origin_bsz4 paper=${TARGET_PAPER} lm=${TARGET_LM} glossary=${GLOSSARY_KIND} tau=${TAU_VALUE}"
