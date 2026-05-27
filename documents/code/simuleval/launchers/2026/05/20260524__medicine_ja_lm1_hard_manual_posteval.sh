#!/usr/bin/env bash
set -euo pipefail

CONDA_PREFIX="${CONDA_PREFIX_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv}"
FBK_FAIRSEQ_ROOT="${FBK_FAIRSEQ_ROOT_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/FBK-fairseq}"
MWERSEGMENTER_ROOT="${MWERSEGMENTER_ROOT_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/mwerSegmenter}"
ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260524_ja_lm1_aries01}"
GLOSSARY="${GLOSSARY_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_hard_manual_glossary_streamlaal_20260524.json}"
TMPDIR="${TMPDIR_OVERRIDE:-/dev/shm/jxjapost1}"

SETTING="${OUTPUT_BASE}/ja/gigaspeech-ja-s_origin-bsz4_gstrict_fixed_medicine_glossary_abbrev_restored__medicine5_cs0.96_hs0.48_lm1_k210_k110_th0p0"
COMBINED="${OUTPUT_BASE}/ja/__medicine_inputs__/combined"
EVAL_TSV="${SETTING}/eval_results_streamlaal_term.hard_llm_manual_check.tsv"
EVAL_LOG="${SETTING}/post_eval_streamlaal_term.hard_llm_manual_check.log"
WORK_DIR="${SETTING}/work_streamlaal_term.hard_llm_manual_check"
MISS_TSV="${SETTING}/term_misses.hard_llm_manual_check.ja_lm1.tsv"
MISS_SUMMARY="${SETTING}/term_miss_summary.hard_llm_manual_check.ja_lm1.tsv"
NORM_GLOSSARY="${SETTING}/hard_medicine_glossary.streamlaal_dict.hard_llm_manual_check.json"

export CONDA_PREFIX
export FBK_FAIRSEQ_ROOT
export MWERSEGMENTER_ROOT
export STREAM_LAAL_TOOL="${FBK_FAIRSEQ_ROOT}/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py"
export PATH="${CONDA_PREFIX}/bin:${MWERSEGMENTER_ROOT}:${PATH}"
export TMPDIR

mkdir -p "${TMPDIR}"

if [[ ! -s "${OUTPUT_BASE}/timing.tsv" ]] ||
  ! awk -F '\t' 'NR>1 && $2 == "1" && $5 == "success" {found=1} END {exit found?0:1}' "${OUTPUT_BASE}/timing.tsv"; then
  echo "[ERROR] JA lm1 generation timing is not successful: ${OUTPUT_BASE}/timing.tsv" >&2
  exit 2
fi

if [[ ! -s "${SETTING}/instances.log" ]]; then
  echo "[ERROR] Missing instances.log: ${SETTING}/instances.log" >&2
  exit 3
fi

instances_count="$(wc -l < "${SETTING}/instances.log")"
if [[ "${instances_count}" != "5" ]]; then
  echo "[ERROR] Expected 5 instances, got ${instances_count}: ${SETTING}/instances.log" >&2
  exit 4
fi

if [[ -s "${EVAL_TSV}" && "${FORCE_REPOST_EVAL_OVERRIDE:-0}" != "1" ]]; then
  echo "[INFO] Existing hard-manual TSV found; skip: ${EVAL_TSV}"
  exit 0
fi

cd "${ROOT_DIR}"
"${CONDA_PREFIX}/bin/python" documents/code/offline_sst_eval/offline_streamlaal_eval.py \
  --mode acl6060 \
  --instances-log "${SETTING}/instances.log" \
  --lang-code ja \
  --source-file "${COMBINED}/medicine5.source_text.en.sentences.txt" \
  --ref-file "${COMBINED}/medicine5.ref.ja.sentences.txt" \
  --audio-yaml "${COMBINED}/medicine5.audio.yaml" \
  --glossary-acl6060 "${GLOSSARY}" \
  --fbk-fairseq-root "${FBK_FAIRSEQ_ROOT}" \
  --term-fcr-policy source_ref_negative_sentence \
  --output-tsv "${EVAL_TSV}" \
  --output-log "${EVAL_LOG}" \
  --work-dir "${WORK_DIR}" \
  --term-mismatch-examples 20

"${CONDA_PREFIX}/bin/python" documents/code/simuleval/export_streamlaal_term_misses.py \
  --instances-log "${SETTING}/instances.log" \
  --reference "${COMBINED}/medicine5.ref.ja.sentences.txt" \
  --source-reference "${COMBINED}/medicine5.source_text.en.sentences.txt" \
  --audio-yaml "${COMBINED}/medicine5.audio.yaml" \
  --glossary "${GLOSSARY}" \
  --lang-code ja \
  --stream-laal-tool "${STREAM_LAAL_TOOL}" \
  --mwersegmenter-root "${MWERSEGMENTER_ROOT}" \
  --output-misses "${MISS_TSV}" \
  --output-summary "${MISS_SUMMARY}" \
  --output-normalized-glossary "${NORM_GLOSSARY}"

echo "[DONE] JA lm1 hard-manual post-eval: ${EVAL_TSV}"
