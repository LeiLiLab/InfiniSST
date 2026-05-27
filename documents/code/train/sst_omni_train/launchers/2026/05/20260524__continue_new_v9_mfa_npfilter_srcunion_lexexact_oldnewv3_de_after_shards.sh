#!/usr/bin/env bash
# Continue the de New V9 data-prep event after manual recovery of retriever shards.
#
# This is only a recovery launcher for:
#   /mnt/gemini/data1/jiaxuanluo/speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_de_20260524
# It waits for both retriever shard logs to report Done, merges the source-copy
# retriever results, then resumes the standard builder from Stage C2.
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
OUT_ROOT="${OUT_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo}"
OUT_NAME_PREFIX="${OUT_NAME_PREFIX_OVERRIDE:-speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3}"
OUT_DIR="${OUT_ROOT}/${OUT_NAME_PREFIX}_de_20260524"
SHARD_DIR="${OUT_DIR}/shards"
LOG_DIR="${OUT_DIR}/logs"
BASE_BUILDER="${ROOT_DIR}/documents/code/train/sst_omni_train/launchers/2026/05/20260524__build_new_v9_mfa_openai_rewrite_oldnewv3_de_ja.sh"
SOURCE_GLOSSARY="${OUT_DIR}/source_glossary_wiki100k_plus_de_oldnewv3_candidates.json"
SOURCE_CANDIDATE_JSONL="${OUT_DIR}/source_candidates/source_candidates_de_oldnewv3_spacy.jsonl"
STAGE1_SOURCECOPY="${OUT_DIR}/stage1_train_s_de_oldnewv3_tcm_sourcecopy_retriever_results.jsonl"

mkdir -p "${LOG_DIR}"

for p in "${BASE_BUILDER}" "${SOURCE_GLOSSARY}" "${SOURCE_CANDIDATE_JSONL}"; do
  if [[ ! -s "${p}" ]]; then
    echo "[ERROR] Missing required file: ${p}" >&2
    exit 3
  fi
done

wait_for_shard_done() {
  local shard="$1"
  local output="${SHARD_DIR}/stage1_retriever_shard_${shard}_of_02.jsonl"
  local log_glob="${LOG_DIR}/stage1_retriever_de_shard${shard}"*.log
  local waited=0
  while true; do
    # shellcheck disable=SC2086
    if [[ -s "${output}" ]] && grep -h -q "^\\[INFO\\] Done:" ${log_glob} 2>/dev/null; then
      echo "[INFO] shard ${shard} done: ${output}"
      return 0
    fi
    if (( waited > 21600 )); then
      echo "[ERROR] Timed out waiting for shard ${shard}: ${output}" >&2
      # shellcheck disable=SC2086
      tail -n 80 ${log_glob} 2>/dev/null || true
      exit 7
    fi
    sleep 30
    waited=$((waited + 30))
  done
}

echo "[INFO] Waiting for retriever shard outputs under ${SHARD_DIR}"
wait_for_shard_done "00"
wait_for_shard_done "01"

echo "[INFO] Merge retriever shards -> ${STAGE1_SOURCECOPY}"
: > "${STAGE1_SOURCECOPY}"
for shard in 00 01; do
  out_shard="${SHARD_DIR}/stage1_retriever_shard_${shard}_of_02.jsonl"
  if [[ ! -s "${out_shard}" ]]; then
    echo "[ERROR] Missing retriever shard output after wait: ${out_shard}" >&2
    exit 8
  fi
  cat "${out_shard}" >> "${STAGE1_SOURCECOPY}"
done

echo "[INFO] Resume standard builder from existing Stage1 sourcecopy"
export ROOT_DIR_OVERRIDE="${ROOT_DIR}"
export OUT_ROOT_OVERRIDE="${OUT_ROOT}"
export OUT_NAME_PREFIX_OVERRIDE="${OUT_NAME_PREFIX}"
export LANGS_OVERRIDE="de"
export RESUME_FROM_STAGE1_SOURCECOPY_OVERRIDE=1
export SOURCE_GLOSSARY_OVERRIDE="${SOURCE_GLOSSARY}"
export SOURCE_CANDIDATE_JSONL_OVERRIDE="${SOURCE_CANDIDATE_JSONL}"
export PREFILTER_REFERENCE_SPANS_FROM_INPUT_TERM_MAP_OVERRIDE=1
export USE_LEGACY_TERMMAP_SPAN_AS_TARGET_WITHOUT_OPENAI_OVERRIDE=1
export OPENAI_TRANSLATE_BATCH_OVERRIDE="${OPENAI_TRANSLATE_BATCH_OVERRIDE:-64}"
export OPENAI_TRANSLATE_WORKERS_OVERRIDE="${OPENAI_TRANSLATE_WORKERS_OVERRIDE:-8}"
export BATCH_ACROSS_CONVERSATIONS_OVERRIDE=1
export AUDIO_ENCODE_BATCH_OVERRIDE="${AUDIO_ENCODE_BATCH_OVERRIDE:-96}"
export MAX_BATCH_SECONDS_OVERRIDE="${MAX_BATCH_SECONDS_OVERRIDE:-180}"

bash "${BASE_BUILDER}"

if command -v "${HOME}/bin/codex-notify" >/dev/null 2>&1; then
  "${HOME}/bin/codex-notify" --delay 8 --detach --workspace "${ROOT_DIR}" \
    "Codex finished: de New V9 SFT data prepared at ${OUT_DIR}"
fi

echo "[OK] de New V9 SFT data prepared: ${OUT_DIR}"
