#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
if [[ -d /mnt/taurus/home/jiaxuanluo/InfiniSST ]]; then
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
fi

SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/augment_term_translation_llm_variants.py"
OUT_DIR="${OUT_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v4_llm_variant_aug_newv3_zh_20260522}"
WORK_DIR="${OUT_DIR}/.work_${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)}_$$"

TRAIN_IN="${TRAIN_IN_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/tcm_wiki100k_gt_zh_tau075_d9_k20_postfiltercap_termmap_v2_sourcefinal_gtzh/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_sourcefinal_tcmwiki100kgt_tau075_d9_k20_postfiltercap_gtzhoverride.jsonl}"
DEV_IN="${DEV_IN_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/dev_s_zh_v4_ner_baseline_aligned_freq_k20_final.jsonl}"
CACHE_JSON="${CACHE_JSON_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_v16_llm_variant_aug_retriever_timeline_zh_lh1b88kw_tau073_20260522/openai_term_variant_cache.json}"

TRAIN_OUT="${OUT_DIR}/train_s_zh_new_v4_llm_variant_aug_newv3_cacheonly.jsonl"
DEV_OUT="${OUT_DIR}/dev_s_zh_new_v4_llm_variant_aug_newv3_cacheonly.jsonl"
TRAIN_STATS="${OUT_DIR}/train_new_v4_llm_variant_aug_stats.json"
DEV_STATS="${OUT_DIR}/dev_new_v4_llm_variant_aug_stats.json"
TRAIN_SAMPLE="${OUT_DIR}/train_new_v4_llm_variant_aug_samples.json"
DEV_SAMPLE="${OUT_DIR}/dev_new_v4_llm_variant_aug_samples.json"
SUMMARY_JSON="${OUT_DIR}/new_v4_llm_variant_aug_summary.json"

for p in "${SCRIPT}" "${TRAIN_IN}" "${DEV_IN}" "${CACHE_JSON}"; do
  if [[ ! -f "${p}" ]]; then
    echo "[ERROR] Missing required file: ${p}" >&2
    exit 3
  fi
done

mkdir -p "${WORK_DIR}"

echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] WORK_DIR=${WORK_DIR}"
echo "[INFO] TRAIN_IN=${TRAIN_IN}"
echo "[INFO] DEV_IN=${DEV_IN}"
echo "[INFO] CACHE_JSON=${CACHE_JSON}"

python "${SCRIPT}" \
  --input-jsonl "${TRAIN_IN}" \
  --output-jsonl "${WORK_DIR}/$(basename "${TRAIN_OUT}")" \
  --stats-json "${WORK_DIR}/$(basename "${TRAIN_STATS}")" \
  --variant-cache-json "${CACHE_JSON}" \
  --sample-json "${WORK_DIR}/$(basename "${TRAIN_SAMPLE}")" \
  --lang-code zh \
  --augment-prob "${AUGMENT_PROB_OVERRIDE:-0.5}" \
  --seed "${TRAIN_SEED_OVERRIDE:-1606}" \
  --max-augmented-terms-per-row "${MAX_AUG_TERMS_PER_ROW_OVERRIDE:-8}" \
  --sample-count "${SAMPLE_COUNT_OVERRIDE:-80}" \
  --cache-only \
  --missing-gt-policy keep_unchanged

python "${SCRIPT}" \
  --input-jsonl "${DEV_IN}" \
  --output-jsonl "${WORK_DIR}/$(basename "${DEV_OUT}")" \
  --stats-json "${WORK_DIR}/$(basename "${DEV_STATS}")" \
  --variant-cache-json "${CACHE_JSON}" \
  --sample-json "${WORK_DIR}/$(basename "${DEV_SAMPLE}")" \
  --lang-code zh \
  --augment-prob "${AUGMENT_PROB_OVERRIDE:-0.5}" \
  --seed "${DEV_SEED_OVERRIDE:-1607}" \
  --max-augmented-terms-per-row "${MAX_AUG_TERMS_PER_ROW_OVERRIDE:-8}" \
  --sample-count "${SAMPLE_COUNT_OVERRIDE:-80}" \
  --cache-only \
  --missing-gt-policy keep_unchanged

python - "${WORK_DIR}" "${OUT_DIR}" "${SUMMARY_JSON}" <<'PY'
import json
import sys
from pathlib import Path

work = Path(sys.argv[1])
out = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
train_stats = json.loads((work / "train_new_v4_llm_variant_aug_stats.json").read_text(encoding="utf-8"))
dev_stats = json.loads((work / "dev_new_v4_llm_variant_aug_stats.json").read_text(encoding="utf-8"))
train_rows = sum(1 for _ in (work / "train_s_zh_new_v4_llm_variant_aug_newv3_cacheonly.jsonl").open(encoding="utf-8"))
dev_rows = sum(1 for _ in (work / "dev_s_zh_new_v4_llm_variant_aug_newv3_cacheonly.jsonl").open(encoding="utf-8"))
if train_rows != 12500:
    raise SystemExit(f"unexpected train row count: {train_rows}")
if dev_rows != 355:
    raise SystemExit(f"unexpected dev row count: {dev_rows}")
summary = {
    "version": "new_v4_llm_variant_aug_newv3_cacheonly",
    "train_rows": train_rows,
    "dev_rows": dev_rows,
    "train_augmented_terms": train_stats.get("augmented_terms", 0),
    "train_selected_terms": train_stats.get("selected_terms", 0),
    "train_augmented_over_selected_rate": train_stats.get("augmented_over_selected_rate", 0.0),
    "train_rows_missing_gt_terms_by_chunk": train_stats.get("rows_missing_gt_terms_by_chunk", 0),
    "dev_rows_missing_gt_terms_by_chunk": dev_stats.get("rows_missing_gt_terms_by_chunk", 0),
    "cache_only": True,
}
out.mkdir(parents=True, exist_ok=True)
(work / "new_v4_llm_variant_aug_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

mkdir -p "${OUT_DIR}"
for f in \
  "$(basename "${TRAIN_OUT}")" \
  "$(basename "${DEV_OUT}")" \
  "$(basename "${TRAIN_STATS}")" \
  "$(basename "${DEV_STATS}")" \
  "$(basename "${TRAIN_SAMPLE}")" \
  "$(basename "${DEV_SAMPLE}")" \
  "$(basename "${SUMMARY_JSON}")"; do
  mv -f "${WORK_DIR}/${f}" "${OUT_DIR}/${f}"
done
rmdir "${WORK_DIR}"

echo "[INFO] Completed new_v4 data prep:"
cat "${SUMMARY_JSON}"
