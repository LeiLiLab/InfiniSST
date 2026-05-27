#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

INPUT_TSV="${INPUT_TSV:-/mnt/gemini/data1/jiaxuanluo/dev_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv}"
NER_JSONL="${NER_JSONL:-/mnt/gemini/data1/jiaxuanluo/ner_candidates_dev_en_core_web_trf.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-/mnt/gemini/home/jiaxuanluo/term_dev_dataset_varctx2p88_3p84_4p80_5p76_new_version.jsonl}"
AUDIO_OUTPUT_PREFIX="${AUDIO_OUTPUT_PREFIX:-/mnt/gemini/home/jiaxuanluo/term_dev_audio_chunks_varctx_m}"
MULTIPLIER_JSONL_PREFIX="${MULTIPLIER_JSONL_PREFIX:-/mnt/gemini/home/jiaxuanluo/term_dev_dataset_varctx_m}"
STATS_JSON="${STATS_JSON:-/mnt/gemini/home/jiaxuanluo/term_dev_with_wiki_synth_normalized_varctx2p88_3p84_4p80_5p76_stats.json}"
DIAG_JSON="${DIAG_JSON:-/mnt/gemini/home/jiaxuanluo/term_dev_with_wiki_synth_normalized_varctx2p88_3p84_4p80_5p76_diag.json}"
DURATION_SECS="${DURATION_SECS:-2.88 3.84 4.80 5.76}"
MULTIPLIERS="${MULTIPLIERS:-3 4 5 6}"
NUM_WORKERS="${NUM_WORKERS:-12}"
FORCE="${FORCE:-false}"

echo "[DEV-VARCTX] input_tsv=${INPUT_TSV}"
echo "[DEV-VARCTX] ner_jsonl=${NER_JSONL}"
echo "[DEV-VARCTX] output=${OUTPUT_JSONL}"
echo "[DEV-VARCTX] audio_output_prefix=${AUDIO_OUTPUT_PREFIX}"
echo "[DEV-VARCTX] durations=${DURATION_SECS}"

multiplier_args=()
for m in ${MULTIPLIERS}; do
  out_jsonl="${MULTIPLIER_JSONL_PREFIX}${m}.jsonl"
  out_dir="${AUDIO_OUTPUT_PREFIX}${m}"
  if [ "${FORCE}" = "true" ] || [ ! -s "${out_jsonl}" ]; then
    echo "[DEV-VARCTX] build multiplier m=${m} jsonl=${out_jsonl}"
    python "${REPO_ROOT}/retriever/gigaspeech/extract_all_terms_from_tsv.py" \
      --input-tsv "${INPUT_TSV}" \
      --output-dir "${out_dir}" \
      --output-jsonl "${out_jsonl}" \
      --multiplier-merge "${m}" \
      --ner-candidates-jsonl "${NER_JSONL}" \
      --num-workers "${NUM_WORKERS}"
  else
    echo "[DEV-VARCTX] reuse multiplier m=${m} jsonl=${out_jsonl}"
  fi
  multiplier_args+=("${m}=${out_jsonl}")
done

python "${REPO_ROOT}/documents/code/data_pre/training_terms_for_retriever/build_term_dev_variable_context_from_multipliers.py" \
  --multiplier-jsonl "${multiplier_args[@]}" \
  --output "${OUTPUT_JSONL}" \
  --stats-json "${STATS_JSON}" \
  --balance min

python "${REPO_ROOT}/documents/code/data_pre/training_terms_for_retriever/diagnose_variable_context_jsonl.py" \
  --input "${OUTPUT_JSONL}" \
  --stats-json "${STATS_JSON}" \
  --expected-duration-secs "${DURATION_SECS}" \
  --report-json "${DIAG_JSON}"

echo "[DEV-VARCTX] DONE diag=${DIAG_JSON}"
