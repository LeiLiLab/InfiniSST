#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

RUN_TAG="${RUN_TAG_OVERRIDE:-20260525T0340_tagacl_oldzh_v2_lm2_hn1024_tau078_batch}"
MODEL_LABEL="oldzh_v2_sourcefinal_gtzh_hn1024_tau078_batch_max80"
INPUT_DIR="/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_zh_lm1_20260524T0507_tagacl_newv9_hn1024_tau078_raw_zh_lm1_aries01/__inputs__/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/full_all/zh/all"

MODEL_NAME="/mnt/gemini/data2/jiaxuanluo/speech_llm_tcm_filtered_wiki100kgt_tau075_v2_sourcefinal_gtzh/keep1.0_r32/v0-20260507-103419-hf"
RAW_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json"
OUT_ROOT="/mnt/gemini/data1/jiaxuanluo/tagged_acl_oldzh_v2_lm2_hn1024_tau078_batch_${RUN_TAG}"
OUTPUT_BASE="${OUT_ROOT}/${MODEL_LABEL}"
LOG_ROOT="/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_oldzh_v2_lm2_hn1024_tau078_batch_${RUN_TAG}"
NOTES_FILE="${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260525__tagged_acl_oldzh_v2_lm2_hn1024_tau078_batch.md"

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
  echo "lms=2"
  echo "tau=0.78"
  echo "vllm_tp_size=2"
  echo "physical_gpus=${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE:-1,7}"
  echo "max_num_seqs=5"
  echo "vllm_limit_audio=128"
  echo "scheduler_batch_size=5"
  echo "schedule_mode=round_robin"
  echo "rag_batch_retrieval=1"
  echo "rag_device=${RAG_DEVICE_OVERRIDE:-cuda:0}"
  echo "max_cache_seconds=80"
  echo "keep_cache_seconds=60"
  echo "max_new_tokens=80"
  echo "max_new_tokens_policy=fixed"
} | tee "${OUT_ROOT}/run_meta.txt"

RUN_TAG_OVERRIDE="${RUN_TAG}" \
LANG_CODE_OVERRIDE="zh" \
LMS_OVERRIDE="2" \
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE:-1,7}" \
VLLM_TP_SIZE_OVERRIDE=2 \
MAX_NUM_SEQS_OVERRIDE=5 \
SCHEDULER_BATCH_SIZE_OVERRIDE=5 \
SCHEDULE_MODE_OVERRIDE=round_robin \
VLLM_ENFORCE_EAGER_OVERRIDE=1 \
VLLM_ENABLE_PREFIX_CACHING=1 \
VLLM_LIMIT_AUDIO_OVERRIDE=128 \
MAX_CACHE_SECONDS_OVERRIDE=80 \
KEEP_CACHE_SECONDS_OVERRIDE=60 \
MIN_CACHE_CHUNKS_OVERRIDE=1 \
MAX_NEW_TOKENS_OVERRIDE=80 \
MAX_NEW_TOKENS_POLICY_OVERRIDE=fixed \
TEMPERATURE_OVERRIDE=0.6 \
TOP_P_OVERRIDE=0.95 \
TOP_K_DECODE_OVERRIDE=20 \
GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}" \
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
DENSITY_TAG_OVERRIDE="tagacl_oldzh_v2_hn1024_tau078_m80" \
GLOSSARY_TAG_OVERRIDE="acl6060_tagged_gt_raw_min_norm2" \
RAG_SCORE_THRESHOLD_OVERRIDE=0.78 \
RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE=1.92 \
RAG_TOP_K_OVERRIDE=10 \
RAG_DEVICE_OVERRIDE="${RAG_DEVICE_OVERRIDE:-cuda:0}" \
RAG_BATCH_RETRIEVAL_OVERRIDE=1 \
INDEX_BUILD_DEVICE_OVERRIDE="${INDEX_BUILD_DEVICE_OVERRIDE:-cuda:0}" \
INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/batched_vllm}" \
TERM_MAP_FORMAT_OVERRIDE=plain \
TERM_FCR_POLICY_OVERRIDE=term_map_source_ref_negative_sentence \
STRIP_OUTPUT_TAGS_OVERRIDE=term \
WANDB_LOG_OVERRIDE="${WANDB_LOG_OVERRIDE:-1}" \
WANDB_RUN_PREFIX_OVERRIDE="oldzh_v2_hn1024_tau078_m80" \
WANDB_EXPERIMENT_FAMILY_OVERRIDE="tagged_acl_oldzh_v2_hn1024_tau078" \
WANDB_VARIANT_PREFIX_OVERRIDE="oldzh_v2_raw_m80" \
WANDB_RUNTIME_GLOSSARY_LABEL_OVERRIDE="raw" \
WANDB_DATA_TAG_OVERRIDE="tagged_acl_raw_zh" \
WANDB_COMPUTE_TAG_OVERRIDE="compute:taurus_oldzh_v2_batch" \
NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
LOG_ROOT_OVERRIDE="${LOG_ROOT}" \
EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR_OVERRIDE:-/tmp/jx_ozh2l2}" \
bash "${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__batched_vllm_rag_eval.sh" \
  > "${LOG_ROOT}/launcher.out" \
  2> "${LOG_ROOT}/launcher.err"

mapfile -t eval_files < <(find "${OUTPUT_BASE}/zh" -type f -name eval_results.tsv | sort)
[[ "${#eval_files[@]}" -eq 1 ]] || fail "expected exactly one eval_results.tsv under ${OUTPUT_BASE}/zh, found ${#eval_files[@]}"
BATCH_DIR="$(dirname "${eval_files[0]}")"
for p in \
  "${BATCH_DIR}/eval_results.tsv" \
  "${BATCH_DIR}/instances.log" \
  "${BATCH_DIR}/instances.strip_term.log"; do
  [[ -s "${p}" ]] || fail "missing/empty batch output: ${p}"
done

tail -n 1 "${BATCH_DIR}/eval_results.tsv" > "${OUT_ROOT}/eval_results.last.tsv"
echo "[DONE] old zh v2 tagged-ACL lm2 max80 batch complete"
echo "[DONE] eval=${BATCH_DIR}/eval_results.tsv"
