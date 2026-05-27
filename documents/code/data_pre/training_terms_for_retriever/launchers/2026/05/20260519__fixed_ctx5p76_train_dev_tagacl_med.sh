#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd)}"
DATA_PRE_DIR="${REPO_ROOT}/documents/code/data_pre/training_terms_for_retriever"
ACL_PRE_DIR="${REPO_ROOT}/documents/code/data_pre/acl"
LOG_DIR="${LOG_DIR:-/mnt/gemini/data1/jiaxuanluo/logs/fixed_ctx5p76_data_prep}"
mkdir -p "${LOG_DIR}"

export DURATION_SECS="${DURATION_SECS:-5.76}"
export DURATION_ASSIGNMENT="${DURATION_ASSIGNMENT:-balance_rows}"
export NUM_SHARDS="${NUM_SHARDS:-8}"
export PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
export NUM_WORKERS="${NUM_WORKERS:-12}"
export OVERWRITE_AUDIO="${OVERWRITE_AUDIO:-false}"
export FORCE_DEV="${FORCE_DEV:-true}"
export COPY_UNEXPANDABLE_GS="${COPY_UNEXPANDABLE_GS:-false}"
export WIKI_EXPAND_FAILURE_POLICY="${WIKI_EXPAND_FAILURE_POLICY:-drop}"
export DIAG_TARGET_FRAC="${DIAG_TARGET_FRAC:-1.0}"
export DIAG_NO_FAIL="${DIAG_NO_FAIL:-true}"
export REPAIR_TRAIN_INVALID_ROWS="${REPAIR_TRAIN_INVALID_ROWS:-true}"
export RUN_TRAIN_CONTEXT="${RUN_TRAIN_CONTEXT:-true}"

TRAIN_INPUT_JSONL="${TRAIN_INPUT_JSONL:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup.jsonl}"
TRAIN_OUTPUT_JSONL="${TRAIN_OUTPUT_JSONL:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_ctx5p76.jsonl}"
TRAIN_AUDIO_OUTPUT_DIR="${TRAIN_AUDIO_OUTPUT_DIR:-/mnt/gemini/home/jiaxuanluo/term_train_audio_chunks_gsv2full_gsdedup_ctx5p76}"
TRAIN_WIKI_AUDIO_OUTPUT_DIR="${TRAIN_WIKI_AUDIO_OUTPUT_DIR:-${TRAIN_AUDIO_OUTPUT_DIR}/wiki_synth}"
TRAIN_STATS_JSON="${TRAIN_STATS_JSON:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_ctx5p76_stats.json}"
TRAIN_DIAG_JSON="${TRAIN_DIAG_JSON:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_ctx5p76_diag.json}"

DEV_INPUT_TSV="${DEV_INPUT_TSV:-/mnt/gemini/data1/jiaxuanluo/dev_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv}"
DEV_NER_JSONL="${DEV_NER_JSONL:-/mnt/gemini/data1/jiaxuanluo/ner_candidates_dev_en_core_web_trf.jsonl}"
DEV_M6_JSONL="${DEV_M6_JSONL:-/mnt/gemini/home/jiaxuanluo/term_dev_dataset_ctx5p76_m6.jsonl}"
DEV_M6_AUDIO_DIR="${DEV_M6_AUDIO_DIR:-/mnt/gemini/home/jiaxuanluo/term_dev_audio_chunks_ctx5p76_m6}"
DEV_OUTPUT_JSONL="${DEV_OUTPUT_JSONL:-/mnt/gemini/home/jiaxuanluo/term_dev_dataset_ctx5p76_new_version.jsonl}"
DEV_STATS_JSON="${DEV_STATS_JSON:-/mnt/gemini/home/jiaxuanluo/term_dev_dataset_ctx5p76_stats.json}"
DEV_DIAG_JSON="${DEV_DIAG_JSON:-/mnt/gemini/home/jiaxuanluo/term_dev_dataset_ctx5p76_diag.json}"

TAGGED_ACL_OUTPUT_DIR="${TAGGED_ACL_OUTPUT_DIR:-/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_tagged_glossary_ctx5p76}"
TAGGED_ACL_AUDIO_DIR="${TAGGED_ACL_AUDIO_DIR:-/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_tagged_ctx5p76/audio_chunks}"
TAGGED_ACL_OUTPUT_JSONL_NAME="${TAGGED_ACL_OUTPUT_JSONL_NAME:-acl6060_tagged_dev_dataset.jsonl}"
TAGGED_ACL_OUTPUT_JSONL="${TAGGED_ACL_OUTPUT_JSONL:-${TAGGED_ACL_OUTPUT_DIR}/${TAGGED_ACL_OUTPUT_JSONL_NAME}}"
TAGGED_ACL_STATS_JSON="${TAGGED_ACL_STATS_JSON:-${TAGGED_ACL_OUTPUT_DIR}/acl6060_tagged_dev_dataset_stats.json}"
TAGGED_ACL_DIAG_JSON="${TAGGED_ACL_DIAG_JSON:-${TAGGED_ACL_OUTPUT_DIR}/acl6060_tagged_dev_dataset_diag.json}"
TAGGED_ACL_EVAL_GLOSSARY="${TAGGED_ACL_EVAL_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_union_gs10000_min_norm2_backfill_ctx5p76.json}"
TAGGED_ACL_MIN_NORM_CHARS="${TAGGED_ACL_MIN_NORM_CHARS:-2}"

MEDICINE_OUTPUT_DIR="${MEDICINE_OUTPUT_DIR:-/mnt/gemini/home/jiaxuanluo/medicine_eval_ctx5p76}"
MEDICINE_AUDIO_DIR="${MEDICINE_AUDIO_DIR:-${MEDICINE_OUTPUT_DIR}/audio_chunks}"
MEDICINE_OUTPUT_JSONL="${MEDICINE_OUTPUT_JSONL:-${MEDICINE_OUTPUT_DIR}/medicine_dev_dataset.jsonl}"
MEDICINE_STATS_JSON="${MEDICINE_STATS_JSON:-${MEDICINE_OUTPUT_DIR}/medicine_dev_dataset_stats.json}"
MEDICINE_DIAG_JSON="${MEDICINE_DIAG_JSON:-${MEDICINE_OUTPUT_DIR}/medicine_dev_dataset_diag.json}"
MEDICINE_DROPPED_TERMS_JSON="${MEDICINE_DROPPED_TERMS_JSON:-${MEDICINE_OUTPUT_DIR}/medicine_dev_dataset_dropped_terms.json}"
MEDICINE_GLOSSARY_JSON="${MEDICINE_GLOSSARY_JSON:-${MEDICINE_OUTPUT_DIR}/medicine_glossary_gt_plus_medicine_wiki_gs10000.json}"
MEDICINE_FILLER_GLOSSARY="${MEDICINE_FILLER_GLOSSARY:-${REPO_ROOT}/documents/code/data_pre/glossary_scale/wiki_glossary_medicine_enriched.json}"
MEDICINE_ALLOWED_LOCATE_METHODS="${MEDICINE_ALLOWED_LOCATE_METHODS:-mfa_exact char_proportional}"
MEDICINE_UNMATCHED_TERM_POLICY="${MEDICINE_UNMATCHED_TERM_POLICY:-drop}"

REQUIRED_INPUTS=(
  "${TRAIN_INPUT_JSONL}"
  "${DEV_INPUT_TSV}"
  "${DEV_NER_JSONL}"
  "${MEDICINE_FILLER_GLOSSARY}"
  "${DATA_PRE_DIR}/run_build_gsv2full_gsdedup_varctx_lmlb2p88_3p84_4p80_5p76_parallel.sh"
  "${DATA_PRE_DIR}/build_term_dev_variable_context_from_multipliers.py"
  "${DATA_PRE_DIR}/diagnose_variable_context_jsonl.py"
  "${ACL_PRE_DIR}/run_build_acl6060_tagged_varctx_lmlb2p88_3p84_4p80_5p76.sh"
  "${DATA_PRE_DIR}/run_prepare_medicine_varctx_lmlb2p88_3p84_4p80_5p76.sh"
)
for required in "${REQUIRED_INPUTS[@]}"; do
  if [ ! -e "${required}" ]; then
    echo "[ERROR] required input missing: ${required}" >&2
    exit 2
  fi
done

echo "[CTX5P76-DATA] duration_secs=${DURATION_SECS}"
echo "[CTX5P76-DATA] train_output=${TRAIN_OUTPUT_JSONL}"
echo "[CTX5P76-DATA] dev_output=${DEV_OUTPUT_JSONL}"
echo "[CTX5P76-DATA] tagged_acl_output=${TAGGED_ACL_OUTPUT_JSONL}"
echo "[CTX5P76-DATA] medicine_output=${MEDICINE_OUTPUT_JSONL}"

if [ "${RUN_TRAIN_CONTEXT}" = "true" ]; then
  INPUT_JSONL="${TRAIN_INPUT_JSONL}" \
  OUTPUT_JSONL="${TRAIN_OUTPUT_JSONL}" \
  AUDIO_OUTPUT_DIR="${TRAIN_AUDIO_OUTPUT_DIR}" \
  WIKI_AUDIO_OUTPUT_DIR="${TRAIN_WIKI_AUDIO_OUTPUT_DIR}" \
  STATS_JSON="${TRAIN_STATS_JSON}" \
  DIAG_JSON="${TRAIN_DIAG_JSON}" \
  DURATION_SECS="${DURATION_SECS}" \
  DURATION_ASSIGNMENT="${DURATION_ASSIGNMENT}" \
  NUM_SHARDS="${NUM_SHARDS}" \
  PARALLEL_JOBS="${PARALLEL_JOBS}" \
  SHARD_DIR="${TRAIN_OUTPUT_JSONL%.jsonl}_shards" \
  LOG_DIR="${LOG_DIR}" \
  OVERWRITE_AUDIO="${OVERWRITE_AUDIO}" \
  COPY_UNEXPANDABLE_GS="${COPY_UNEXPANDABLE_GS}" \
  WIKI_EXPAND_FAILURE_POLICY="${WIKI_EXPAND_FAILURE_POLICY}" \
  DIAG_TARGET_FRAC="${DIAG_TARGET_FRAC}" \
  DIAG_NO_FAIL="${DIAG_NO_FAIL}" \
  RUN_DIAG=true \
  bash "${DATA_PRE_DIR}/run_build_gsv2full_gsdedup_varctx_lmlb2p88_3p84_4p80_5p76_parallel.sh"
else
  echo "[CTX5P76-DATA] skip train context build; using existing jsonl=${TRAIN_OUTPUT_JSONL}"
fi

if [ "${REPAIR_TRAIN_INVALID_ROWS}" = "true" ]; then
  echo "[CTX5P76-DATA] repair train invalid rows in-place jsonl=${TRAIN_OUTPUT_JSONL}"
  python "${DATA_PRE_DIR}/repair_variable_context_jsonl.py" \
    --input "${TRAIN_OUTPUT_JSONL}" \
    --output "${TRAIN_OUTPUT_JSONL}" \
    --stats-json "${TRAIN_STATS_JSON}" \
    --source-stats-json "${TRAIN_STATS_JSON}" \
    --expected-duration-secs "${DURATION_SECS}"

  python "${DATA_PRE_DIR}/diagnose_variable_context_jsonl.py" \
    --input "${TRAIN_OUTPUT_JSONL}" \
    --stats-json "${TRAIN_STATS_JSON}" \
    --expected-duration-secs "${DURATION_SECS}" \
    --target-frac "${DIAG_TARGET_FRAC}" \
    --report-json "${TRAIN_DIAG_JSON}"
fi

if [ "${FORCE_DEV}" = "true" ] || [ ! -s "${DEV_M6_JSONL}" ]; then
  echo "[CTX5P76-DATA] rebuild dev multiplier m=6 jsonl=${DEV_M6_JSONL}"
  python "${REPO_ROOT}/retriever/gigaspeech/extract_all_terms_from_tsv.py" \
    --input-tsv "${DEV_INPUT_TSV}" \
    --output-dir "${DEV_M6_AUDIO_DIR}" \
    --output-jsonl "${DEV_M6_JSONL}" \
    --multiplier-merge 6 \
    --ner-candidates-jsonl "${DEV_NER_JSONL}" \
    --num-workers "${NUM_WORKERS}"
else
  echo "[CTX5P76-DATA] reuse dev multiplier m=6 jsonl=${DEV_M6_JSONL}"
fi

python "${DATA_PRE_DIR}/build_term_dev_variable_context_from_multipliers.py" \
  --multiplier-jsonl "6=${DEV_M6_JSONL}" \
  --output "${DEV_OUTPUT_JSONL}" \
  --stats-json "${DEV_STATS_JSON}" \
  --balance all \
  --context-build term_dev_multiplier_ctx5p76

python "${DATA_PRE_DIR}/diagnose_variable_context_jsonl.py" \
  --input "${DEV_OUTPUT_JSONL}" \
  --stats-json "${DEV_STATS_JSON}" \
  --expected-duration-secs "${DURATION_SECS}" \
  --target-frac "${DIAG_TARGET_FRAC}" \
  --report-json "${DEV_DIAG_JSON}"

OUTPUT_DIR="${TAGGED_ACL_OUTPUT_DIR}" \
CHUNK_AUDIO_DIR="${TAGGED_ACL_AUDIO_DIR}" \
OUTPUT_JSONL_NAME="${TAGGED_ACL_OUTPUT_JSONL_NAME}" \
STATS_JSON="${TAGGED_ACL_STATS_JSON}" \
DIAG_JSON="${TAGGED_ACL_DIAG_JSON}" \
EVAL_GLOSSARY_OUTPUT="${TAGGED_ACL_EVAL_GLOSSARY}" \
DURATION_SECS="${DURATION_SECS}" \
MIN_NORM_CHARS="${TAGGED_ACL_MIN_NORM_CHARS}" \
OVERWRITE_AUDIO="${OVERWRITE_AUDIO}" \
DIAG_TARGET_FRAC="${DIAG_TARGET_FRAC}" \
bash "${ACL_PRE_DIR}/run_build_acl6060_tagged_varctx_lmlb2p88_3p84_4p80_5p76.sh"

OUTPUT_DIR="${MEDICINE_OUTPUT_DIR}" \
CHUNK_AUDIO_DIR="${MEDICINE_AUDIO_DIR}" \
FILLER_GLOSSARY="${MEDICINE_FILLER_GLOSSARY}" \
DURATION_SECS="${DURATION_SECS}" \
OVERWRITE_AUDIO="${OVERWRITE_AUDIO}" \
UNMATCHED_TERM_POLICY="${MEDICINE_UNMATCHED_TERM_POLICY}" \
DROPPED_TERMS_JSON="${MEDICINE_DROPPED_TERMS_JSON}" \
ALLOWED_LOCATE_METHODS="${MEDICINE_ALLOWED_LOCATE_METHODS}" \
bash "${DATA_PRE_DIR}/run_prepare_medicine_varctx_lmlb2p88_3p84_4p80_5p76.sh"

python "${DATA_PRE_DIR}/diagnose_variable_context_jsonl.py" \
  --input "${MEDICINE_OUTPUT_JSONL}" \
  --stats-json "${MEDICINE_STATS_JSON}" \
  --expected-duration-secs "${DURATION_SECS}" \
  --target-frac "${DIAG_TARGET_FRAC}" \
  --report-json "${MEDICINE_DIAG_JSON}"

for output in \
  "${TRAIN_OUTPUT_JSONL}" \
  "${TRAIN_STATS_JSON}" \
  "${TRAIN_DIAG_JSON}" \
  "${DEV_OUTPUT_JSONL}" \
  "${DEV_STATS_JSON}" \
  "${DEV_DIAG_JSON}" \
  "${TAGGED_ACL_OUTPUT_JSONL}" \
  "${TAGGED_ACL_STATS_JSON}" \
  "${TAGGED_ACL_DIAG_JSON}" \
  "${TAGGED_ACL_EVAL_GLOSSARY}" \
  "${MEDICINE_OUTPUT_JSONL}" \
  "${MEDICINE_STATS_JSON}" \
  "${MEDICINE_DIAG_JSON}" \
  "${MEDICINE_DROPPED_TERMS_JSON}" \
  "${MEDICINE_GLOSSARY_JSON}"; do
  if [ ! -s "${output}" ]; then
    echo "[ERROR] expected output missing or empty: ${output}" >&2
    exit 2
  fi
done

echo "[CTX5P76-DATA] DONE"
