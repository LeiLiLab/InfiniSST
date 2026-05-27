#!/usr/bin/env bash
set -euo pipefail

# One-lm same-lm batch eval for clean ja New V9 MFA/npfilter Speech LLM.
# The caller supplies LM_OVERRIDE and CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE.

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

BATCH_LAUNCHER="${BATCH_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__batched_vllm_rag_eval.sh}"
LM="${LM_OVERRIDE:?LM_OVERRIDE is required}"
RUN_STAMP="${RUN_STAMP_OVERRIDE:-$(date -u +%Y%m%dT%H%M%S)_tagacl_newv9_mfa_npfilter_ja_lm${LM}}"

MODEL_LABEL="${MODEL_LABEL:-new_v9_mfa_npfilter_lexexact_oldnewv3_ja_r32a64_hn1024_tau078_same_lm_batch_v1}"
MODEL_NAME="${MODEL_NAME_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_ja_r32a64_tp2_aries4/keep1.0_r32/v0-20260525-020815-hf}"
INPUT_DIR="${INPUT_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_ja_lm1to4_20260524T0933_tagacl_newv9_hn1024_tau078_raw_ja_taurus8/__inputs__/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/full_all/ja/all}"
RAW_GLOSSARY="${RAW_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
HN1024_CKPT="${HN1024_CKPT:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"

NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260524__tagged_acl_new_v9_mfa_npfilter_lexexact_hn1024_tau078_raw_ja_after_train.md}"
OUT_ROOT="${OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_mfa_npfilter_lexexact_hn1024_tau078_raw_ja_lm1to4_${RUN_STAMP}}"
OUTPUT_BASE="${OUTPUT_BASE:-${OUT_ROOT}/${MODEL_LABEL}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_new_v9_mfa_npfilter_lexexact_hn1024_tau078_raw_ja_lm1to4_${RUN_STAMP}/lm${LM}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/batched_vllm}"
EVAL_TMPDIR="${EVAL_TMPDIR_OVERRIDE:-/tmp/jx_bvja_${LM}_${USER:-u}}"

fail() {
  echo "[ERROR] $*" >&2
  exit 3
}

require_file() {
  local path="$1"
  [[ -s "${path}" ]] || fail "Missing/empty required file: ${path}"
}

hf_ready() {
  [[ -s "${MODEL_NAME}/config.json" ]] || return 1
  [[ -s "${MODEL_NAME}/generation_config.json" ]] || return 1
  [[ -s "${MODEL_NAME}/model.safetensors.index.json" ]] || return 1
  [[ -s "${MODEL_NAME}/tokenizer_config.json" ]] || return 1
  local shard_count
  shard_count="$(find "${MODEL_NAME}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
  [[ "${shard_count}" == "15" ]] || return 1
}

require_file "${BATCH_LAUNCHER}"
require_file "${RAW_GLOSSARY}"
require_file "${HN1024_CKPT}"
require_file "${NOTES_FILE}"
for f in source.list target.list source_text.txt ref.txt audio.yaml; do
  require_file "${INPUT_DIR}/${f}"
done
hf_ready || fail "HF export is incomplete: ${MODEL_NAME}"

mkdir -p "${OUT_ROOT}" "${OUTPUT_BASE}/__summary__" "${LOG_ROOT}" "${EVAL_TMPDIR}"

{
  echo "run_stamp=${RUN_STAMP}"
  echo "host=$(hostname -s)"
  echo "lm=${LM}"
  echo "gpu_pair=${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE:-<unset>}"
  echo "model=${MODEL_NAME}"
  echo "input_dir=${INPUT_DIR}"
  echo "raw_glossary=${RAW_GLOSSARY}"
  echo "retriever=${HN1024_CKPT}"
  echo "output_base=${OUTPUT_BASE}"
  echo "eval_mode=same_lm_batch_v1"
  echo "talks_per_lm=5"
  echo "tau=0.78"
  echo "lookback_sec=1.92"
  echo "max_new_tokens=80"
  echo "vllm_limit_audio=128"
  echo "strip_output_tags=term"
} | tee "${LOG_ROOT}/run_meta_lm${LM}.txt"

RUN_TAG_OVERRIDE="${RUN_STAMP}_ja_lm${LM}" \
LANG_CODE_OVERRIDE="ja" \
LMS_OVERRIDE="${LM}" \
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE:?CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE is required}" \
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
DENSITY_TAG_OVERRIDE="tagacl_bv1_mfa_np_ja_hn1024_tau078" \
GLOSSARY_TAG_OVERRIDE="acl6060_tagged_gt_raw_min_norm2" \
RAG_MODEL_PATH_OVERRIDE="${HN1024_CKPT}" \
RAG_SCORE_THRESHOLD_OVERRIDE=0.78 \
RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE=1.92 \
RAG_TOP_K_OVERRIDE=10 \
RAG_DEVICE_OVERRIDE="${RAG_DEVICE_OVERRIDE:-cuda:0}" \
RAG_BATCH_RETRIEVAL_OVERRIDE=1 \
INDEX_BUILD_DEVICE_OVERRIDE="${INDEX_BUILD_DEVICE_OVERRIDE:-cuda:0}" \
INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}" \
TERM_MAP_FORMAT_OVERRIDE=plain \
TERM_FCR_POLICY_OVERRIDE=term_map_source_ref_negative_sentence \
STRIP_OUTPUT_TAGS_OVERRIDE=term \
WANDB_LOG_OVERRIDE="${WANDB_LOG_OVERRIDE:-1}" \
WANDB_RUN_PREFIX_OVERRIDE="mfa_np_bv1_hn1024_t078" \
WANDB_EXPERIMENT_FAMILY_OVERRIDE="tagged_acl_same_lm_batch_v1_hn1024_tau078" \
WANDB_VARIANT_PREFIX_OVERRIDE="mfa_np_bv1_hn1024_t078" \
WANDB_COMPUTE_TAG_OVERRIDE="${WANDB_COMPUTE_TAG_OVERRIDE:-compute:same_lm_batch_v1}" \
WANDB_RUNTIME_GLOSSARY_LABEL_OVERRIDE="raw" \
WANDB_DATA_TAG_OVERRIDE="tagged_acl_strict_raw_ja" \
NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
LOG_ROOT_OVERRIDE="${LOG_ROOT}" \
EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR}" \
bash "${BATCH_LAUNCHER}" \
  > "${LOG_ROOT}/launcher.out" \
  2> "${LOG_ROOT}/launcher.err"

python - "${OUTPUT_BASE}" "${LM}" "${LOG_ROOT}" <<'PY'
import csv
import re
import sys
from pathlib import Path

output_base = Path(sys.argv[1])
lm = sys.argv[2]
log_root = Path(sys.argv[3])
paths = sorted(output_base.glob(f"ja/**_lm{lm}_*/eval_results.tsv"))
if len(paths) != 1:
    raise SystemExit(f"expected one eval_results.tsv for lm={lm}, found {len(paths)}: {paths}")
with paths[0].open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
if len(rows) != 1:
    raise SystemExit(f"expected one row in {paths[0]}, got {len(rows)}")
summary = log_root / f"lm{lm}_eval_result_path.txt"
summary.write_text(str(paths[0]) + "\n", encoding="utf-8")
print(paths[0])
PY

echo "[DONE] same-lm batch ja lm${LM} complete"
