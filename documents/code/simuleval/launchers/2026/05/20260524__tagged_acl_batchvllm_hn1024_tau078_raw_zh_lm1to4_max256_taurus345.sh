#!/usr/bin/env bash
set -euo pipefail

# Throughput run after serial-compatible batch-vLLM validation.
# This keeps the same model/RAG/cache semantics but raises max_new_tokens to
# handle term-tag-heavy outputs and schedules all zh lm=1..4 streams together.

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

RUN_TAG="${RUN_TAG_OVERRIDE:-20260524T1300_tagacl_batchvllm_hn1024_tau078_raw_zh_lm1to4_max256_taurus345}"
MODEL_LABEL="new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078_batchvllm_max256"
INPUT_DIR="/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_zh_lm1_20260524T0507_tagacl_newv9_hn1024_tau078_raw_zh_lm1_aries01/__inputs__/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/full_all/zh/all"

MODEL_NAME="/mnt/gemini/data1/jiaxuanluo/slm_exports/speech_llm_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_zh_r32a64_tp2_taurus8/keep1.0_r32/v0-20260524-062743-hf"
RAW_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json"
OUT_ROOT="/mnt/gemini/data1/jiaxuanluo/tagged_acl_batchvllm_hn1024_tau078_raw_zh_lm1to4_max256_${RUN_TAG}"
OUTPUT_BASE="${OUT_ROOT}/${MODEL_LABEL}"
LOG_ROOT="/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_batchvllm_hn1024_tau078_raw_zh_lm1to4_max256_${RUN_TAG}"
NOTES_FILE="${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260524__tagged_acl_batchvllm_hn1024_tau078_raw_zh_lm1to4_max256.md"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

fail() {
  echo "[ERROR] $*" >&2
  exit 3
}

for p in \
  "${MODEL_NAME}/config.json" \
  "${RAW_GLOSSARY}" \
  "${INPUT_DIR}/source.list" \
  "${INPUT_DIR}/target.list" \
  "${INPUT_DIR}/source_text.txt" \
  "${INPUT_DIR}/ref.txt" \
  "${INPUT_DIR}/audio.yaml" \
  "${NOTES_FILE}"; do
  [[ -s "${p}" ]] || fail "missing/empty required path: ${p}"
done

shards="$(find "${MODEL_NAME}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
[[ "${shards}" == "15" ]] || fail "expected 15 model shards, found ${shards}: ${MODEL_NAME}"

{
  echo "run_tag=${RUN_TAG}"
  echo "host=$(hostname -s)"
  echo "model=${MODEL_NAME}"
  echo "input_dir=${INPUT_DIR}"
  echo "raw_glossary=${RAW_GLOSSARY}"
  echo "output_base=${OUTPUT_BASE}"
  echo "lms=1 2 3 4"
  echo "tau=0.78"
  echo "vllm_tp_size=2"
  echo "max_num_seqs=8"
  echo "scheduler_batch_size=8"
  echo "schedule_mode=round_robin"
  echo "max_new_tokens=256"
  echo "max_cache_seconds=80"
  echo "keep_cache_seconds=60"
} | tee "${OUT_ROOT}/run_meta.txt"

RUN_TAG_OVERRIDE="${RUN_TAG}" \
LANG_CODE_OVERRIDE="zh" \
LMS_OVERRIDE="1 2 3 4" \
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE:-3,4,5}" \
VLLM_TP_SIZE_OVERRIDE=2 \
MAX_NUM_SEQS_OVERRIDE=8 \
SCHEDULER_BATCH_SIZE_OVERRIDE=8 \
SCHEDULE_MODE_OVERRIDE=round_robin \
VLLM_ENFORCE_EAGER_OVERRIDE=1 \
VLLM_ENABLE_PREFIX_CACHING=1 \
VLLM_LIMIT_AUDIO_OVERRIDE=96 \
MAX_CACHE_SECONDS_OVERRIDE=80 \
KEEP_CACHE_SECONDS_OVERRIDE=60 \
MIN_CACHE_CHUNKS_OVERRIDE=1 \
MAX_NEW_TOKENS_OVERRIDE=256 \
TEMPERATURE_OVERRIDE=0.6 \
TOP_P_OVERRIDE=0.95 \
TOP_K_DECODE_OVERRIDE=20 \
GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.80}" \
VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN_OVERRIDE:-16384}" \
VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}" \
VLLM_USE_FUSED_MOE_GROUPED_TOPK="${VLLM_USE_FUSED_MOE_GROUPED_TOPK:-0}" \
MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
SRC_LIST_OVERRIDE="${INPUT_DIR}/source.list" \
TGT_LIST_OVERRIDE="${INPUT_DIR}/target.list" \
SOURCE_TEXT_FILE_OVERRIDE="${INPUT_DIR}/source_text.txt" \
REF_FILE_OVERRIDE="${INPUT_DIR}/ref.txt" \
AUDIO_YAML_OVERRIDE="${INPUT_DIR}/audio.yaml" \
GLOSSARY_PATH_OVERRIDE="${RAW_GLOSSARY}" \
EVAL_GLOSSARY_PATH_OVERRIDE="${RAW_GLOSSARY}" \
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
DENSITY_TAG_OVERRIDE="tagacl_batchvllm_hn1024_tau078_max256" \
GLOSSARY_TAG_OVERRIDE="acl6060_tagged_gt_raw_min_norm2" \
RAG_SCORE_THRESHOLD_OVERRIDE=0.78 \
RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE=1.92 \
RAG_TOP_K_OVERRIDE=10 \
RAG_DEVICE_OVERRIDE="${RAG_DEVICE_OVERRIDE:-cuda:2}" \
INDEX_BUILD_DEVICE_OVERRIDE="${INDEX_BUILD_DEVICE_OVERRIDE:-cuda:2}" \
INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/batched_vllm}" \
TERM_MAP_FORMAT_OVERRIDE=plain \
TERM_FCR_POLICY_OVERRIDE=term_map_source_ref_negative_sentence \
STRIP_OUTPUT_TAGS_OVERRIDE=term \
WANDB_LOG_OVERRIDE="${WANDB_LOG_OVERRIDE:-0}" \
NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
LOG_ROOT_OVERRIDE="${LOG_ROOT}" \
EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR_OVERRIDE:-/tmp/jx_bvm256}" \
bash "${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__batched_vllm_rag_eval.sh" \
  > "${LOG_ROOT}/launcher.out" \
  2> "${LOG_ROOT}/launcher.err"

echo "[DONE] batch max256 run complete"
echo "[DONE] output=${OUTPUT_BASE}"
