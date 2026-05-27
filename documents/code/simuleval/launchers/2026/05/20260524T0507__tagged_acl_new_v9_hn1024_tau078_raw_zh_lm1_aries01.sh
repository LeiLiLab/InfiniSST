#!/usr/bin/env bash
set -euo pipefail

# zh tagged ACL RASST main-result first setting:
# Speech LLM new_v9, HN1024 lh1b88kw retriever, raw glossary, lm=1.
# This launcher is intentionally Aries GPU 0,1 only and keeps generated data off
# /mnt/aries/data7 because that mount is full.

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
QUICK="${QUICK:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260522__tagged_acl_speech_llm_quick_zh_lm2_raw_wait_hf.sh}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260520__tagged_acl_origin_bsz4_tau073_sweep_taurus45269.sh}"
RUN_STAMP="${RUN_STAMP:-20260524T0507_tagacl_newv9_hn1024_tau078_raw_zh_lm1_aries01}"

MODEL_LABEL="${MODEL_LABEL:-new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/gemini/data1/jiaxuanluo/slm_exports/speech_llm_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_zh_r32a64_tp2_taurus8/keep1.0_r32}"
EXPECTED_MODEL_DIR="${EXPECTED_MODEL_DIR:-${MODEL_ROOT}/v0-20260524-062743-hf}"
HN1024_CKPT="${HN1024_CKPT:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
RAW_GLOSSARY="${RAW_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260524T0507__tagged_acl_new_v9_hn1024_tau078_raw_zh_lm1_aries01.md}"

OUT_ROOT="${OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_zh_lm1_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_new_v9_hn1024_tau078_raw_zh_lm1_${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache}"
EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/tmp/tacl0507_lm1}"

GPU_PAIR="${GPU_PAIR:-0,1}"
MIN_DATA1_FREE_MB="${MIN_DATA1_FREE_MB:-200000}"
MAX_IDLE_GPU_MEM_MB="${MAX_IDLE_GPU_MEM_MB:-1024}"
MAX_IDLE_GPU_UTIL="${MAX_IDLE_GPU_UTIL:-20}"

fail() {
  echo "[ERROR] $*" >&2
  exit 3
}

require_path() {
  local path="$1"
  [[ -e "${path}" ]] || fail "Missing required path: ${path}"
}

validate_model_dir() {
  require_path "${EXPECTED_MODEL_DIR}/config.json"
  local shard_count
  shard_count="$(find "${EXPECTED_MODEL_DIR}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
  [[ "${shard_count}" == "15" ]] || fail "Expected 15 HF safetensor shards in ${EXPECTED_MODEL_DIR}, found ${shard_count}"
  local latest
  latest="$(find "${MODEL_ROOT}" -maxdepth 1 -type d -name '*-hf' 2>/dev/null | sort | tail -n 1 || true)"
  [[ "${latest}" == "${EXPECTED_MODEL_DIR}" ]] || fail "Latest HF dir under MODEL_ROOT is ${latest}; expected ${EXPECTED_MODEL_DIR}"
}

check_host() {
  local host
  host="$(hostname -s)"
  [[ "${host}" == aries* ]] || fail "This launcher must run on aries; current host=${host}"
}

check_data1_space() {
  local free_mb
  free_mb="$(df -Pm /mnt/gemini/data1 | awk 'NR == 2 {print $4}')"
  [[ -n "${free_mb}" ]] || fail "Could not read free space for /mnt/gemini/data1"
  (( free_mb >= MIN_DATA1_FREE_MB )) || fail "/mnt/gemini/data1 free space ${free_mb} MB < ${MIN_DATA1_FREE_MB} MB"
}

check_gpu_idle() {
  command -v nvidia-smi >/dev/null || fail "nvidia-smi not found"
  local gpu_csv
  gpu_csv="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"
  local gpu line mem util
  for gpu in 0 1; do
    line="$(awk -F, -v g="${gpu}" '$1 + 0 == g {print $0}' <<< "${gpu_csv}")"
    [[ -n "${line}" ]] || fail "GPU ${gpu} not found in nvidia-smi output"
    mem="$(awk -F, '{gsub(/[[:space:]]/, "", $2); print $2}' <<< "${line}")"
    util="$(awk -F, '{gsub(/[[:space:]]/, "", $3); print $3}' <<< "${line}")"
    (( mem <= MAX_IDLE_GPU_MEM_MB )) || fail "GPU ${gpu} memory used ${mem} MiB > ${MAX_IDLE_GPU_MEM_MB} MiB"
    (( util <= MAX_IDLE_GPU_UTIL )) || fail "GPU ${gpu} utilization ${util}% > ${MAX_IDLE_GPU_UTIL}%"
  done
}

validate_tags() {
  local tags=(
    "family:tagged_acl_new_v9_hn1024_tau078"
    "task:eval"
    "data:tagged_acl_strict_raw_zh"
    "variant:new_v9_hn1024_tau078_zh_raw_lm1"
    "compute:aries_direct_gpu01"
  )
  local tag
  for tag in "${tags[@]}"; do
    (( ${#tag} >= 1 && ${#tag} <= 64 )) || fail "W&B tag length invalid (${#tag}): ${tag}"
  done
}

preflight() {
  check_host
  check_data1_space
  check_gpu_idle
  validate_tags
  for p in \
    "${ROOT_DIR}" \
    "${QUICK}" \
    "${BASE_LAUNCHER}" \
    "${NOTES_FILE}" \
    "${HN1024_CKPT}" \
    "${RAW_GLOSSARY}" \
    "${ROOT_DIR}/documents/code/simuleval/eval_density_unified.sh" \
    "${ROOT_DIR}/documents/code/offline_evaluation/wandb_eval_logger.py" \
    "${ROOT_DIR}/documents/code/general/wandb_tool.py" \
    "/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python"; do
    require_path "${p}"
  done
  validate_model_dir
}

preflight

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[PREFLIGHT] ok"
  exit 0
fi

mkdir -p \
  "${OUT_ROOT}" \
  "${LOG_ROOT}" \
  "${INDEX_CACHE_DIR}" \
  "${EVAL_TMPDIR_OVERRIDE}" \
  "/mnt/gemini/data1/jiaxuanluo/wandb_runs" \
  "/mnt/gemini/data1/jiaxuanluo/wandb_cache" \
  "/mnt/gemini/data1/jiaxuanluo/wandb_data" \
  "/mnt/gemini/data1/jiaxuanluo/wandb_artifacts"

export BASE_LAUNCHER
export ROOT_DIR
export RUN_STAMP
export MODEL_ROOT
export MODEL_LABEL
export GPU_PAIR
export MODE="full"
export WAIT_FOR_HF_SECS="0"
export OUT_ROOT
export LOG_ROOT
export INDEX_CACHE_DIR
export NOTES_FILE
export LANGS="zh"
export LMS="1"
export GLOSSARY_KINDS="raw"
export RUN_GRANULARITY="full_corpus"
export PAPERS="2022.acl-long.268 2022.acl-long.367 2022.acl-long.590 2022.acl-long.110 2022.acl-long.117"
export RAW_GLOSSARY_OVERRIDE="${RAW_GLOSSARY}"
export EVAL_GLOSSARY_PATH_GLOBAL="${RAW_GLOSSARY}"
export EVAL_GLOSSARY_FOLLOWS_KIND="0"
export RAG_MODEL_PATH_OVERRIDE="${HN1024_CKPT}"
export RAG_SCORE_THRESHOLD_OVERRIDE="0.78"
export RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE="1.92"
export RAG_TOP_K_OVERRIDE="10"
export STRIP_OUTPUT_TAGS_OVERRIDE="term"
export RAG_GPU_OVERRIDE="cuda:1"
export VLLM_TP_SIZE_OVERRIDE="2"
export VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_USE_FUSED_MOE_GROUPED_TOPK="${VLLM_USE_FUSED_MOE_GROUPED_TOPK:-0}"
export EVAL_TMPDIR_OVERRIDE
export CLEAN_OUTPUT_DIR_OVERRIDE="${CLEAN_OUTPUT_DIR_OVERRIDE:-1}"
export DENSITY_TAG_OVERRIDE="tagacl_new_v9_hn1024_tau078"
export WANDB_RUN_PREFIX_OVERRIDE="new_v9_hn1024_tau078"
export WANDB_EXPERIMENT_FAMILY_OVERRIDE="tagged_acl_new_v9_hn1024_tau078"
export WANDB_VARIANT_PREFIX_OVERRIDE="new_v9_hn1024_tau078"
export WANDB_COMPUTE_TAG_OVERRIDE="compute:aries_direct_gpu01"
export TERM_FCR_POLICY_OVERRIDE="term_map_source_ref_negative_sentence"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/mnt/taurus/home/jiaxuanluo/.config/wandb}"
export WANDB_DIR="${WANDB_DIR:-/mnt/gemini/data1/jiaxuanluo/wandb_runs}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/mnt/gemini/data1/jiaxuanluo/wandb_cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-/mnt/gemini/data1/jiaxuanluo/wandb_data}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-/mnt/gemini/data1/jiaxuanluo/wandb_artifacts}"

echo "[INFO] host=$(hostname -s)"
echo "[INFO] RUN_STAMP=${RUN_STAMP}"
echo "[INFO] GPUs=${GPU_PAIR}"
echo "[INFO] model=${EXPECTED_MODEL_DIR}"
echo "[INFO] retriever=${HN1024_CKPT}"
echo "[INFO] glossary=${RAW_GLOSSARY}"
echo "[INFO] out=${OUT_ROOT}"
echo "[INFO] logs=${LOG_ROOT}"
echo "[INFO] tmp=${EVAL_TMPDIR_OVERRIDE}"
echo "[INFO] VLLM_DISABLE_CUSTOM_ALL_REDUCE=${VLLM_DISABLE_CUSTOM_ALL_REDUCE}"
echo "[INFO] VLLM_MOE_USE_DEEP_GEMM=${VLLM_MOE_USE_DEEP_GEMM}"
echo "[INFO] VLLM_USE_FUSED_MOE_GROUPED_TOPK=${VLLM_USE_FUSED_MOE_GROUPED_TOPK}"

bash "${QUICK}"

EVAL_TSV="${OUT_ROOT}/${MODEL_LABEL}/zh/dtagacl_new_v9_hn1024_tau078_lm1_k10_th0.78_gacl6060_tagged_gt_raw_min_norm2/eval_results.tsv"
INSTANCES_LOG="${OUT_ROOT}/${MODEL_LABEL}/zh/dtagacl_new_v9_hn1024_tau078_lm1_k10_th0.78_gacl6060_tagged_gt_raw_min_norm2/instances.log"
TERM_ADOPTION="${OUT_ROOT}/${MODEL_LABEL}/zh/dtagacl_new_v9_hn1024_tau078_lm1_k10_th0.78_gacl6060_tagged_gt_raw_min_norm2/term_adoption.json"

[[ -s "${EVAL_TSV}" ]] || fail "Missing eval_results.tsv after run: ${EVAL_TSV}"
[[ -s "${INSTANCES_LOG}" ]] || fail "Missing instances.log after run: ${INSTANCES_LOG}"
[[ -s "${TERM_ADOPTION}" ]] || fail "Missing term_adoption.json after run: ${TERM_ADOPTION}"

echo "[ALL DONE] Tagged ACL New V9 HN1024 tau0.78 raw zh lm1 complete"
echo "[RESULT] ${EVAL_TSV}"
