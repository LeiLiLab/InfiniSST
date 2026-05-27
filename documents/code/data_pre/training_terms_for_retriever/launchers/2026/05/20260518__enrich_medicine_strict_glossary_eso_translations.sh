#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/jiaxuanluo/InfiniSST"
SCRIPT="${ROOT_DIR}/documents/code/data_pre/training_terms_for_retriever/src/enrich_medicine_glossary_translations_from_eso.py"

ESO_TEST_ROOT="${ESO_TEST_ROOT_OVERRIDE:-/home/jiaxingxu/rag-sst/eso-dataset/outputs_v2/test}"
INPUT_GLOSSARY="${INPUT_GLOSSARY_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_glossary_gt_plus_medicine_wiki_gs10000.json}"
WIKI_ENRICHED_GLOSSARY="${WIKI_ENRICHED_GLOSSARY_OVERRIDE:-${ROOT_DIR}/documents/code/data_pre/glossary_scale/wiki_glossary_medicine_enriched.json}"
OUTPUT_GLOSSARY="${OUTPUT_GLOSSARY_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_glossary_gt_plus_medicine_wiki_gs10000_translated.json}"
STATS_JSON="${STATS_JSON_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_glossary_gt_plus_medicine_wiki_gs10000_translated_stats.json}"

cd "${ROOT_DIR}"

python3 "${SCRIPT}" \
  --eso-test-root "${ESO_TEST_ROOT}" \
  --input-glossary "${INPUT_GLOSSARY}" \
  --wiki-enriched-glossary "${WIKI_ENRICHED_GLOSSARY}" \
  --output-glossary "${OUTPUT_GLOSSARY}" \
  --stats-json "${STATS_JSON}"
