#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
INPUT_JSONL="${INPUT_JSONL:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx0p96_1p92_2p88_3p84.jsonl}"
STATS_JSON="${STATS_JSON:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx0p96_1p92_2p88_3p84_stats.json}"
REPORT_JSON="${REPORT_JSON:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx0p96_1p92_2p88_3p84_diag.json}"
DURATION_SECS="${DURATION_SECS:-0.96 1.92 2.88 3.84}"

python "${REPO_ROOT}/documents/code/data_pre/training_terms_for_retriever/diagnose_variable_context_jsonl.py" \
  --input "${INPUT_JSONL}" \
  --stats-json "${STATS_JSON}" \
  --expected-duration-secs "${DURATION_SECS}" \
  --report-json "${REPORT_JSON}"
