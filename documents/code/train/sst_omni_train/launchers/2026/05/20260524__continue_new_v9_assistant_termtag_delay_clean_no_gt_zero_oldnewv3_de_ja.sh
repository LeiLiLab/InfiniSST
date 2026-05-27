#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
if [[ -d /mnt/taurus/home/jiaxuanluo/InfiniSST ]]; then
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
fi

CONDA_PREFIX_OVERRIDE="${CONDA_PREFIX_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv}"
if [[ -d "${CONDA_PREFIX_OVERRIDE}" ]]; then
  export PATH="${CONDA_PREFIX_OVERRIDE}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX_OVERRIDE}/lib:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUT_ROOT="${OUT_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo}"
DATA_ROOT="${DATA_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo}"
LANGS="${LANGS_OVERRIDE:-de ja}"
DEV_ROWS="${DEV_ROWS_OVERRIDE:-355}"
FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"
CACHE_ONLY="${CACHE_ONLY_OVERRIDE:-0}"
DRY_RUN="${DRY_RUN_OVERRIDE:-0}"

AUGMENT_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/augment_term_translation_llm_variants.py"
ZERO_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/zero_no_gt_termmap_chunks.py"
WRAP_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/wrap_assistant_term_targets.py"

EXCLUDE_SOURCE_TOKENS="${EXCLUDE_SOURCE_TOKENS_OVERRIDE:-this,that,these,those,his,her,hers,him,he,she,it,its,they,them,their,theirs,you,your,yours,we,our,ours,i,me,my,mine,myself,yourself,himself,herself,itself,ourselves,yourselves,themselves,what,which,who,whom,whose,someone,somebody,something,anyone,anybody,anything,everyone,everybody,everything}"

for p in "${AUGMENT_SCRIPT}" "${ZERO_SCRIPT}" "${WRAP_SCRIPT}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

if [[ "${CACHE_ONLY}" != "1" && "${DRY_RUN}" != "1" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ERROR] OPENAI_API_KEY is required unless CACHE_ONLY=1 or DRY_RUN=1." >&2
  exit 5
fi

echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] LANGS=${LANGS}"
echo "[INFO] CACHE_ONLY=${CACHE_ONLY} DRY_RUN=${DRY_RUN}"

for lang in ${LANGS}; do
  source_jsonl="${DATA_ROOT}/train_s_${lang}_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl"
  out_dir="${OUT_ROOT}/speech_llm_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_${lang}_20260524"
  oldnewv3_jsonl="${out_dir}/stage2_train_s_${lang}_oldnewv3_equiv_tcmwiki_termmap_gtbackfill.jsonl"
  new_v4_train="${out_dir}/train_s_${lang}_new_v4_llm_variant_aug_oldnewv3_cache.jsonl"
  new_v4_dev="${out_dir}/dev_s_${lang}_new_v4_llm_variant_aug_oldnewv3_cache_first${DEV_ROWS}.jsonl"
  new_v5_train="${out_dir}/train_s_${lang}_new_v5_no_gt_zero_llm_variant_aug_oldnewv3.jsonl"
  new_v5_dev="${out_dir}/dev_s_${lang}_new_v5_no_gt_zero_llm_variant_aug_oldnewv3_first${DEV_ROWS}.jsonl"
  final_train="${out_dir}/train_s_${lang}_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3.jsonl"
  final_dev="${out_dir}/dev_s_${lang}_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_first${DEV_ROWS}.jsonl"
  variant_cache="${out_dir}/openai_term_variant_cache_${lang}.json"
  summary="${out_dir}/new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_${lang}_summary.json"

  if [[ ! -s "${oldnewv3_jsonl}" ]]; then
    echo "[ERROR] Missing prerequisite Stage 2 JSONL for ${lang}: ${oldnewv3_jsonl}" >&2
    exit 3
  fi

  outputs=(
    "${new_v4_train}"
    "${new_v4_dev}"
    "${new_v5_train}"
    "${new_v5_dev}"
    "${final_train}"
    "${final_dev}"
    "${summary}"
    "${out_dir}/train_new_v4_llm_variant_aug_stats.json"
    "${out_dir}/dev_new_v4_llm_variant_aug_first${DEV_ROWS}_stats.json"
    "${out_dir}/train_new_v5_no_gt_zero_stats.json"
    "${out_dir}/dev_new_v5_no_gt_zero_first${DEV_ROWS}_stats.json"
    "${out_dir}/train_new_v9_assistant_termtag_delay_clean_stats.json"
    "${out_dir}/dev_new_v9_assistant_termtag_delay_clean_first${DEV_ROWS}_stats.json"
    "${out_dir}/train_new_v4_llm_variant_aug_samples.json"
    "${out_dir}/dev_new_v4_llm_variant_aug_first${DEV_ROWS}_samples.json"
    "${out_dir}/train_new_v5_no_gt_zero_samples.json"
    "${out_dir}/dev_new_v5_no_gt_zero_first${DEV_ROWS}_samples.json"
    "${out_dir}/train_new_v9_assistant_termtag_delay_clean_samples.json"
    "${out_dir}/dev_new_v9_assistant_termtag_delay_clean_first${DEV_ROWS}_samples.json"
  )
  if [[ "${FORCE_OVERWRITE}" == "1" ]]; then
    rm -f "${outputs[@]}"
  else
    for p in "${outputs[@]}"; do
      if [[ -e "${p}" ]]; then
        echo "[ERROR] Output exists for ${lang}: ${p}" >&2
        echo "[ERROR] Set FORCE_OVERWRITE=1 only if this rerun is intentional." >&2
        exit 4
      fi
    done
  fi

  echo "[STAGE 3] ${lang}: New V4 LLM-variant augmentation"
  augment_common=(
    --lang-code "${lang}"
    --augment-prob "${AUGMENT_PROB_OVERRIDE:-0.5}"
    --max-augmented-terms-per-row "${MAX_AUG_TERMS_PER_ROW_OVERRIDE:-8}"
    --sample-count "${SAMPLE_COUNT_OVERRIDE:-80}"
    --missing-gt-policy error
  )
  if [[ "${CACHE_ONLY}" == "1" ]]; then
    augment_common+=(--cache-only)
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    augment_common+=(--dry-run)
  fi

  python3 "${AUGMENT_SCRIPT}" \
    --input-jsonl "${oldnewv3_jsonl}" \
    --output-jsonl "${new_v4_train}" \
    --stats-json "${out_dir}/train_new_v4_llm_variant_aug_stats.json" \
    --variant-cache-json "${variant_cache}" \
    --sample-json "${out_dir}/train_new_v4_llm_variant_aug_samples.json" \
    --seed "${TRAIN_SEED_OVERRIDE:-1606}" \
    "${augment_common[@]}"

  python3 "${AUGMENT_SCRIPT}" \
    --input-jsonl "${oldnewv3_jsonl}" \
    --output-jsonl "${new_v4_dev}" \
    --stats-json "${out_dir}/dev_new_v4_llm_variant_aug_first${DEV_ROWS}_stats.json" \
    --variant-cache-json "${variant_cache}" \
    --sample-json "${out_dir}/dev_new_v4_llm_variant_aug_first${DEV_ROWS}_samples.json" \
    --seed "${DEV_SEED_OVERRIDE:-1607}" \
    --max-rows "${DEV_ROWS}" \
    "${augment_common[@]}"

  echo "[STAGE 4] ${lang}: New V5 no-GT-zero"
  python3 "${ZERO_SCRIPT}" \
    --input-jsonl "${new_v4_train}" \
    --output-jsonl "${new_v5_train}" \
    --stats-json "${out_dir}/train_new_v5_no_gt_zero_stats.json" \
    --sample-json "${out_dir}/train_new_v5_no_gt_zero_samples.json" \
    --missing-gt-policy error

  python3 "${ZERO_SCRIPT}" \
    --input-jsonl "${new_v4_dev}" \
    --output-jsonl "${new_v5_dev}" \
    --stats-json "${out_dir}/dev_new_v5_no_gt_zero_first${DEV_ROWS}_stats.json" \
    --sample-json "${out_dir}/dev_new_v5_no_gt_zero_first${DEV_ROWS}_samples.json" \
    --missing-gt-policy error

  echo "[STAGE 5] ${lang}: New V9 assistant <term> tagging"
  wrap_common=(
    --lang-code "${lang}"
    --tag-template '<term>{translation}</term>'
    --min-target-chars 2
    --max-tags-per-row 16
    --missing-gt-policy error
    --exclude-source-tokens "${EXCLUDE_SOURCE_TOKENS}"
    --enable-local-rewrite
    --rewrite-avoid-boundary-overlap
    --rewrite-delay-boundary-prefix
    --rewrite-delay-boundary-min-prefix-chars 2
    --rewrite-require-text-boundaries
    --rewrite-min-target-chars 4
    --rewrite-min-score 0.58
    --rewrite-min-coverage 0.40
    --rewrite-max-span-ratio 1.60
    --rewrite-max-span-extra-chars 4
    --sample-count 200
  )
  python3 "${WRAP_SCRIPT}" \
    --input-jsonl "${new_v5_train}" \
    --output-jsonl "${final_train}" \
    --stats-json "${out_dir}/train_new_v9_assistant_termtag_delay_clean_stats.json" \
    --sample-json "${out_dir}/train_new_v9_assistant_termtag_delay_clean_samples.json" \
    "${wrap_common[@]}"

  python3 "${WRAP_SCRIPT}" \
    --input-jsonl "${new_v5_dev}" \
    --output-jsonl "${final_dev}" \
    --stats-json "${out_dir}/dev_new_v9_assistant_termtag_delay_clean_first${DEV_ROWS}_stats.json" \
    --sample-json "${out_dir}/dev_new_v9_assistant_termtag_delay_clean_first${DEV_ROWS}_samples.json" \
    "${wrap_common[@]}"

  python3 - "${lang}" "${source_jsonl}" "${out_dir}" "${summary}" <<'PY'
import json
import sys
from pathlib import Path

lang = sys.argv[1]
source = sys.argv[2]
out = Path(sys.argv[3])
summary_path = Path(sys.argv[4])

def load(name):
    p = out / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

summary = {
    "event": f"new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_{lang}",
    "source_jsonl": source,
    "policy": {
        "zh_lineage_reference": "sourcefinal_tcmwiki100kgt -> New V4 LLM-variant -> New V5 no-GT-zero -> New V9 assistant term tags",
        "stage2_input": str(out / f"stage2_train_s_{lang}_oldnewv3_equiv_tcmwiki_termmap_gtbackfill.jsonl"),
        "variant_stage": "OpenAI natural target variants",
        "no_gt_zero": True,
        "assistant_tag_template": "<term>{translation}</term>",
    },
    "stats": {
        "derive_gt": load("stage0_derive_gt_stats.json"),
        "glossary": load("stage0_glossary_stats.json"),
        "new_v4": load("train_new_v4_llm_variant_aug_stats.json"),
        "new_v5": load("train_new_v5_no_gt_zero_stats.json"),
        "new_v9": load("train_new_v9_assistant_termtag_delay_clean_stats.json"),
    },
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
PY

  echo "[OK] ${lang}: ${final_train}"
  echo "[OK] ${lang}: ${final_dev}"
  echo "[OK] ${lang}: ${summary}"
done
