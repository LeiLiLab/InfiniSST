#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/wrap_assistant_term_targets.py"

BASE_DIR="${BASE_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v5_no_gt_zero_llm_variant_aug_oldnewv3_zh_20260522}"
OUT_DIR="${OUT_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_zh_20260523}"

TRAIN_IN="${TRAIN_IN_OVERRIDE:-${BASE_DIR}/train_s_zh_new_v5_no_gt_zero_llm_variant_aug_oldnewv3_cacheonly.jsonl}"
DEV_IN="${DEV_IN_OVERRIDE:-${BASE_DIR}/dev_s_zh_new_v5_no_gt_zero_llm_variant_aug_oldnewv3_cacheonly.jsonl}"
TRAIN_OUT="${TRAIN_OUT_OVERRIDE:-${OUT_DIR}/train_s_zh_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3.jsonl}"
DEV_OUT="${DEV_OUT_OVERRIDE:-${OUT_DIR}/dev_s_zh_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3.jsonl}"

TRAIN_STATS="${OUT_DIR}/train_new_v9_assistant_termtag_delay_clean_stats.json"
DEV_STATS="${OUT_DIR}/dev_new_v9_assistant_termtag_delay_clean_stats.json"
TRAIN_SAMPLES="${OUT_DIR}/train_new_v9_assistant_termtag_delay_clean_samples.json"
DEV_SAMPLES="${OUT_DIR}/dev_new_v9_assistant_termtag_delay_clean_samples.json"
SUMMARY="${OUT_DIR}/new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_summary.json"

EXCLUDE_SOURCE_TOKENS="${EXCLUDE_SOURCE_TOKENS_OVERRIDE:-this,that,these,those,his,her,hers,him,he,she,it,its,they,them,their,theirs,you,your,yours,we,our,ours,i,me,my,mine,myself,yourself,himself,herself,itself,ourselves,yourselves,themselves,what,which,who,whom,whose,someone,somebody,something,anyone,anybody,anything,everyone,everybody,everything}"

for p in "${SCRIPT}" "${TRAIN_IN}" "${DEV_IN}"; do
  if [[ ! -f "${p}" ]]; then
    echo "[ERROR] Missing required file: ${p}" >&2
    exit 3
  fi
done
mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
  --lang-code zh
  --tag-template '<term>{translation}</term>'
  --min-target-chars 2
  --max-tags-per-row 16
  --missing-gt-policy keep_unchanged
  --exclude-source-tokens "${EXCLUDE_SOURCE_TOKENS}"
  --enable-local-rewrite
  --rewrite-avoid-boundary-overlap
  --rewrite-delay-boundary-prefix
  --rewrite-delay-boundary-min-prefix-chars 2
  --rewrite-min-target-chars 4
  --rewrite-min-score 0.58
  --rewrite-min-coverage 0.40
  --rewrite-max-span-ratio 1.60
  --rewrite-max-span-extra-chars 4
  --sample-count 200
)

python3 "${SCRIPT}" \
  --input-jsonl "${TRAIN_IN}" \
  --output-jsonl "${TRAIN_OUT}" \
  --stats-json "${TRAIN_STATS}" \
  --sample-json "${TRAIN_SAMPLES}" \
  "${COMMON_ARGS[@]}"

python3 "${SCRIPT}" \
  --input-jsonl "${DEV_IN}" \
  --output-jsonl "${DEV_OUT}" \
  --stats-json "${DEV_STATS}" \
  --sample-json "${DEV_SAMPLES}" \
  "${COMMON_ARGS[@]}"

python3 - "${TRAIN_STATS}" "${DEV_STATS}" "${SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

train = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
dev = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
summary = {
    "event": "new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_zh",
    "base_data": "new_v5_no_gt_zero_llm_variant_aug_oldnewv3_zh",
    "policy": {
        "user_input_unchanged": True,
        "term_map_unchanged": True,
        "exact_tagging": True,
        "local_rewrite_fallback": True,
        "rewrite_avoid_boundary_overlap": True,
        "rewrite_delay_boundary_prefix": True,
        "rewrite_delay_boundary_min_prefix_chars": 2,
        "exclude_source_tokens": train.get("exclude_source_tokens", []),
        "tag_template": "<term>{translation}</term>",
        "rewrite_min_target_chars": 4,
        "rewrite_min_score": 0.58,
        "rewrite_min_coverage": 0.40,
        "rewrite_max_span_ratio": 1.60,
        "rewrite_max_span_extra_chars": 4,
    },
    "train": train,
    "dev": dev,
}
Path(sys.argv[3]).write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
PY

echo "[OK] Wrote ${TRAIN_OUT}"
echo "[OK] Wrote ${DEV_OUT}"
echo "[OK] Wrote ${SUMMARY}"
