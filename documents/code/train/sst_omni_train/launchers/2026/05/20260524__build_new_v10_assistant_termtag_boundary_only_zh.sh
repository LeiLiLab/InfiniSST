#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/wrap_assistant_term_targets.py"

BASE_DIR="${BASE_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v5_no_gt_zero_llm_variant_aug_oldnewv3_zh_20260522}"
OUT_DIR="${OUT_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v10_assistant_termtag_boundary_only_no_gt_zero_oldnewv3_zh_20260524}"

TRAIN_IN="${TRAIN_IN_OVERRIDE:-${BASE_DIR}/train_s_zh_new_v5_no_gt_zero_llm_variant_aug_oldnewv3_cacheonly.jsonl}"
DEV_IN="${DEV_IN_OVERRIDE:-${BASE_DIR}/dev_s_zh_new_v5_no_gt_zero_llm_variant_aug_oldnewv3_cacheonly.jsonl}"
TRAIN_OUT="${TRAIN_OUT_OVERRIDE:-${OUT_DIR}/train_s_zh_new_v10_assistant_termtag_boundary_only_no_gt_zero_oldnewv3.jsonl}"
DEV_OUT="${DEV_OUT_OVERRIDE:-${OUT_DIR}/dev_s_zh_new_v10_assistant_termtag_boundary_only_no_gt_zero_oldnewv3.jsonl}"

TRAIN_STATS="${OUT_DIR}/train_new_v10_assistant_termtag_boundary_only_stats.json"
DEV_STATS="${OUT_DIR}/dev_new_v10_assistant_termtag_boundary_only_stats.json"
TRAIN_SAMPLES="${OUT_DIR}/train_new_v10_assistant_termtag_boundary_only_samples.json"
DEV_SAMPLES="${OUT_DIR}/dev_new_v10_assistant_termtag_boundary_only_samples.json"
SUMMARY="${OUT_DIR}/new_v10_assistant_termtag_boundary_only_no_gt_zero_oldnewv3_summary.json"

EXCLUDE_SOURCE_TOKENS="${EXCLUDE_SOURCE_TOKENS_OVERRIDE:-this,that,these,those,his,her,hers,him,he,she,it,its,they,them,their,theirs,you,your,yours,we,our,ours,i,me,my,mine,myself,yourself,himself,herself,itself,ourselves,yourselves,themselves,what,which,who,whom,whose,someone,somebody,something,anyone,anybody,anything,everyone,everybody,everything}"

for p in "${SCRIPT}" "${TRAIN_IN}" "${DEV_IN}"; do
  if [[ ! -f "${p}" ]]; then
    echo "[ERROR] Missing required file: ${p}" >&2
    exit 3
  fi
done
mkdir -p "${OUT_DIR}"

if [[ "${FORCE_OVERWRITE:-0}" == "1" ]]; then
  rm -f "${TRAIN_OUT}" "${DEV_OUT}" "${TRAIN_STATS}" "${DEV_STATS}" "${TRAIN_SAMPLES}" "${DEV_SAMPLES}" "${SUMMARY}"
else
  for p in "${TRAIN_OUT}" "${DEV_OUT}" "${TRAIN_STATS}" "${DEV_STATS}" "${TRAIN_SAMPLES}" "${DEV_SAMPLES}" "${SUMMARY}"; do
    if [[ -e "${p}" ]]; then
      echo "[ERROR] Output exists: ${p}" >&2
      echo "[ERROR] Set FORCE_OVERWRITE=1 for an intentional rebuild." >&2
      exit 4
    fi
  done
fi

COMMON_ARGS=(
  --lang-code zh
  --tag-template '<term>{translation}</term>'
  --min-target-chars 2
  --max-tags-per-row 16
  --missing-gt-policy keep_unchanged
  --exclude-source-tokens "${EXCLUDE_SOURCE_TOKENS}"
  --enable-local-rewrite
  --rewrite-boundary-only
  --rewrite-delay-boundary-min-prefix-chars 2
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

python3 - "${TRAIN_STATS}" "${DEV_STATS}" "${TRAIN_OUT}" "${DEV_OUT}" "${SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

train_stats = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
dev_stats = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
train_rows = sum(1 for _ in Path(sys.argv[3]).open(encoding="utf-8"))
dev_rows = sum(1 for _ in Path(sys.argv[4]).open(encoding="utf-8"))
summary = {
    "event": "new_v10_assistant_termtag_boundary_only_no_gt_zero_oldnewv3_zh",
    "base_data": "new_v5_no_gt_zero_llm_variant_aug_oldnewv3_zh",
    "policy": {
        "user_input_unchanged": True,
        "term_map_unchanged": True,
        "exact_tagging": True,
        "local_rewrite_fallback": "adjacent_assistant_boundary_only",
        "sequence_matcher_fuzzy_rewrite": False,
        "tag_template": "<term>{translation}</term>",
    },
    "train_rows": train_rows,
    "dev_rows": dev_rows,
    "train": train_stats,
    "dev": dev_stats,
}
Path(sys.argv[5]).write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
PY

echo "[OK] Wrote ${TRAIN_OUT}"
echo "[OK] Wrote ${DEV_OUT}"
echo "[OK] Wrote ${SUMMARY}"
