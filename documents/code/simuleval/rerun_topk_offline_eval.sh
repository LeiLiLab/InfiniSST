#!/usr/bin/env bash
set -euo pipefail

# Recovery step for Phase 0.5 top-k ablation: the inference finished OK but the
# run_one_density_eval.sh offline-eval step silently fell through to base conda
# python (which does not have simuleval). Re-run the extracted_by_paper offline
# eval directly with the spaCyEnv python for each k in {3,5,7,10}, then
# aggregate the summary TSV.

# ======Configuration=====
EXIT_CONFIG_ERROR="2"

ROOT_DIR="/home/jiaxuanluo/InfiniSST"
OFFLINE_EVAL_SCRIPT="${ROOT_DIR}/documents/code/offline_sst_eval/offline_streamlaal_eval.py"

SPACY_PYTHON="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python3.10"

MWERSEGMENTER_ROOT="/mnt/taurus/home/jiaxuanluo/mwerSegmenter"
export MWERSEGMENTER_ROOT
export PATH="${MWERSEGMENTER_ROOT}:${PATH}"

LANG_CODE="zh"
DENSITY="5"
LATENCY_MULTIPLIER="1"
GLOSSARY_ACL6060="${ROOT_DIR}/retriever/gigaspeech/data_pre/glossary_acl6060.json"
EXTRACTED_GLOSSARY_MANIFEST="${ROOT_DIR}/documents/data/data_pre/extracted_glossary_by_paper_manifest.json"

OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/density_eval_maxsim_fixed"

TOPK_VALUES=(3 5 7 10)

SUMMARY_TSV="${OUTPUT_BASE}/${LANG_CODE}/d${DENSITY}_lm${LATENCY_MULTIPLIER}_topk_ablation_summary.tsv"
# ======Configuration=====

if [[ ! -f "${OFFLINE_EVAL_SCRIPT}" ]]; then
  echo "[ERROR] offline eval script missing: ${OFFLINE_EVAL_SCRIPT}" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi
if [[ ! -x "${SPACY_PYTHON}" ]]; then
  echo "[ERROR] spaCyEnv python missing: ${SPACY_PYTHON}" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi

for k in "${TOPK_VALUES[@]}"; do
  COMBINED_DIR="${OUTPUT_BASE}/${LANG_CODE}/d${DENSITY}_lm${LATENCY_MULTIPLIER}_k${k}_per_paper_combined"
  INST_LOG="${COMBINED_DIR}/instances.log"
  OUT_TSV="${COMBINED_DIR}/eval_results_by_paper.tsv"
  OUT_LOG="${COMBINED_DIR}/eval_results_by_paper.log"

  if [[ ! -f "${INST_LOG}" ]] || [[ ! -s "${INST_LOG}" ]]; then
    echo "[WARN] k=${k} missing combined instances.log: ${INST_LOG}" >&2
    continue
  fi
  if [[ -f "${OUT_TSV}" ]] && [[ -s "${OUT_TSV}" ]]; then
    echo "[SKIP] k=${k} already has ${OUT_TSV}"
    continue
  fi

  echo "[RUN] k=${k} offline eval at $(date '+%H:%M:%S')"
  "${SPACY_PYTHON}" "${OFFLINE_EVAL_SCRIPT}" \
    --mode extracted_by_paper \
    --instances-log "${INST_LOG}" \
    --lang-code "${LANG_CODE}" \
    --glossary-acl6060 "${GLOSSARY_ACL6060}" \
    --extracted-glossary-manifest "${EXTRACTED_GLOSSARY_MANIFEST}" \
    --output-tsv "${OUT_TSV}" \
    --output-log "${OUT_LOG}" \
    --python-bin "${SPACY_PYTHON}" \
    2>&1 | tee "${COMBINED_DIR}/rerun_offline_eval.log"
  rc="${PIPESTATUS[0]}"
  if [[ "${rc}" != "0" ]]; then
    echo "[ERROR] k=${k} offline eval rc=${rc}" >&2
  else
    echo "[DONE] k=${k} offline eval"
  fi
done

echo ""
echo "[INFO] Aggregating summary TSV: ${SUMMARY_TSV}"
{
  printf "density\tlm\tk\tBLEU\tStreamLAAL\tStreamLAAL_CA\tTERM_ACC\tTERM_CORRECT\tTERM_TOTAL\tTCR\tTCR_ADOPTED\tTCR_TOTAL\tTERM_FCR\tFALSE_COPY\tNEG_TOTAL\tdir\n"
  for k in "${TOPK_VALUES[@]}"; do
    bp_tsv="${OUTPUT_BASE}/${LANG_CODE}/d${DENSITY}_lm${LATENCY_MULTIPLIER}_k${k}_per_paper_combined/eval_results_by_paper.tsv"
    if [[ -f "${bp_tsv}" ]] && [[ -s "${bp_tsv}" ]]; then
      row="$(tail -1 "${bp_tsv}")"
      bleu="$(printf '%s' "${row}" | cut -f3)"
      slaal="$(printf '%s' "${row}" | cut -f4)"
      slaal_ca="$(printf '%s' "${row}" | cut -f5)"
      term_acc="$(printf '%s' "${row}" | cut -f6)"
      term_correct="$(printf '%s' "${row}" | cut -f7)"
      term_total="$(printf '%s' "${row}" | cut -f8)"
      tcr="$(printf '%s' "${row}" | cut -f9)"
      tcr_adopted="$(printf '%s' "${row}" | cut -f10)"
      tcr_total="$(printf '%s' "${row}" | cut -f11)"
      term_fcr="$(printf '%s' "${row}" | cut -f12)"
      false_copy="$(printf '%s' "${row}" | cut -f13)"
      neg_total="$(printf '%s' "${row}" | cut -f14)"
      dir="${OUTPUT_BASE}/${LANG_CODE}/d${DENSITY}_lm${LATENCY_MULTIPLIER}_k${k}_per_paper_combined"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${DENSITY}" "${LATENCY_MULTIPLIER}" "${k}" \
        "${bleu}" "${slaal}" "${slaal_ca}" \
        "${term_acc}" "${term_correct}" "${term_total}" \
        "${tcr}" "${tcr_adopted}" "${tcr_total}" \
        "${term_fcr}" "${false_copy}" "${neg_total}" \
        "${dir}"
    else
      echo "[WARN] Missing eval_results_by_paper.tsv for k=${k}: ${bp_tsv}" >&2
    fi
  done
} > "${SUMMARY_TSV}"

echo ""
echo "[INFO] ============================================================"
echo "[INFO] Summary TSV: ${SUMMARY_TSV}"
echo "[INFO] ============================================================"
column -ts $'\t' "${SUMMARY_TSV}" | cut -c1-200
