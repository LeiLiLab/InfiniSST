#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
DATA_ROOT="${DATA_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo}"
SRC_DIR="${DATA_ROOT}/speech_llm_deja_termmap_ablation_cap16_exactboundary_20260525/ja/retriever_hn1024_tau078_cap16_exactboundary"
OUT_ROOT="${OUT_ROOT_OVERRIDE:-${DATA_ROOT}/speech_llm_ja_cap16_denoise_budget_20260525}"
BRANCH_DIR="${OUT_ROOT}/ja/hn1024_tau078_cap16_denoise_budget_ttag_v1"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-${DATA_ROOT}/logs/ja_cap16_denoise_budget_ttag_20260525}"

REBUILD_SCRIPT="${ROOT_DIR}/documents/code/data_pre/hard_negative_jsonl_for_speech_llm/rebuild_termmap_denoise_budget.py"
WRAP_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/wrap_assistant_term_targets.py"

TRAIN_RETRIEVED="${SRC_DIR}/train_s_ja_retriever_results_hn1024_tau078.jsonl"
DEV_RETRIEVED="${SRC_DIR}/dev_s_ja_retriever_results_hn1024_tau078_first355.jsonl"
STAGE1_TRAIN="${BRANCH_DIR}/train_s_ja_retriever_hn1024_tau078_cap16_denoise_budget_stage1.jsonl"
STAGE1_DEV="${BRANCH_DIR}/dev_s_ja_retriever_hn1024_tau078_cap16_denoise_budget_stage1_first355.jsonl"
FINAL_TRAIN="${BRANCH_DIR}/train_s_ja_retriever_hn1024_tau078_cap16_denoise_budget_ttag_exactboundary.jsonl"
FINAL_DEV="${BRANCH_DIR}/dev_s_ja_retriever_hn1024_tau078_cap16_denoise_budget_ttag_exactboundary_first355.jsonl"

EXCLUDE_SOURCE_TOKENS="${EXCLUDE_SOURCE_TOKENS_OVERRIDE:-this,that,these,those,his,her,hers,him,he,she,it,its,they,them,their,theirs,you,your,yours,we,our,ours,i,me,my,mine,myself,yourself,himself,herself,itself,ourselves,yourselves,themselves,what,which,who,whom,whose,someone,somebody,something,anyone,anybody,anything,everyone,everybody,everything}"
FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${BRANCH_DIR}" "${LOG_ROOT}"

for p in "${REBUILD_SCRIPT}" "${WRAP_SCRIPT}" "${TRAIN_RETRIEVED}" "${DEV_RETRIEVED}"; do
  [[ -s "${p}" ]] || { echo "[ERROR] Missing required path: ${p}" >&2; exit 3; }
done

maybe_rm_outputs() {
  if [[ "${FORCE_OVERWRITE}" == "1" ]]; then
    rm -f "$@"
    return
  fi
  for p in "$@"; do
    if [[ -e "${p}" ]]; then
      echo "[ERROR] Output exists: ${p}" >&2
      echo "[ERROR] Set FORCE_OVERWRITE=1 only for an intentional rebuild." >&2
      exit 4
    fi
  done
}

run_rebuild() {
  local input_jsonl="$1"
  local output_jsonl="$2"
  local stats_json="$3"
  local sample_json="$4"
  python3 "${REBUILD_SCRIPT}" \
    --input-jsonl "${input_jsonl}" \
    --output-jsonl "${output_jsonl}" \
    --stats-json "${stats_json}" \
    --sample-json "${sample_json}" \
    --target-lang ja \
    --budget-choices "6,8,10" \
    --budget-weights "0.45,0.35,0.20" \
    --no-gt-max-terms 4 \
    --no-gt-empty-prob 0.35 \
    --low-score-cutoff 0.82 \
    --mid-score-cutoff 0.85 \
    --low-score-keep-prob 0.25 \
    --mid-score-keep-prob 0.60 \
    --high-score-keep-prob 0.90 \
    --supported-non-gt-keep-prob 0.85 \
    --missing-score-keep-prob 0.50 \
    --min-target-chars 2 \
    --seed 42 \
    --sample-count 200
}

run_wrap() {
  local input_jsonl="$1"
  local output_jsonl="$2"
  local stats_json="$3"
  local sample_json="$4"
  python3 "${WRAP_SCRIPT}" \
    --input-jsonl "${input_jsonl}" \
    --output-jsonl "${output_jsonl}" \
    --stats-json "${stats_json}" \
    --sample-json "${sample_json}" \
    --lang-code ja \
    --tag-template '<t>{translation}</t>' \
    --min-target-chars 2 \
    --max-tags-per-row 16 \
    --missing-gt-policy error \
    --exclude-source-tokens "${EXCLUDE_SOURCE_TOKENS}" \
    --exact-require-text-boundaries \
    --enable-local-rewrite \
    --rewrite-boundary-only \
    --rewrite-delay-boundary-prefix \
    --rewrite-delay-boundary-min-prefix-chars 2 \
    --rewrite-require-text-boundaries \
    --sample-count 200
}

maybe_rm_outputs \
  "${STAGE1_TRAIN}" "${STAGE1_DEV}" "${FINAL_TRAIN}" "${FINAL_DEV}" \
  "${BRANCH_DIR}/train_rebuild_stats.json" "${BRANCH_DIR}/train_rebuild_samples.json" \
  "${BRANCH_DIR}/dev_rebuild_stats.json" "${BRANCH_DIR}/dev_rebuild_samples.json" \
  "${BRANCH_DIR}/train_wrap_stats.json" "${BRANCH_DIR}/train_wrap_samples.json" \
  "${BRANCH_DIR}/dev_wrap_stats.json" "${BRANCH_DIR}/dev_wrap_samples.json" \
  "${BRANCH_DIR}/runtime_termmap_budget_schedule.json" \
  "${BRANCH_DIR}/validation_summary.json"

echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] SRC_DIR=${SRC_DIR}"
echo "[INFO] BRANCH_DIR=${BRANCH_DIR}"
df -h /mnt/gemini/data1

echo "[STAGE] Rebuild train/dev term maps with denoise budget policy"
run_rebuild "${TRAIN_RETRIEVED}" "${STAGE1_TRAIN}" "${BRANCH_DIR}/train_rebuild_stats.json" "${BRANCH_DIR}/train_rebuild_samples.json"
run_rebuild "${DEV_RETRIEVED}" "${STAGE1_DEV}" "${BRANCH_DIR}/dev_rebuild_stats.json" "${BRANCH_DIR}/dev_rebuild_samples.json"

echo "[STAGE] Wrap assistant GT target translations with short <t> tags"
run_wrap "${STAGE1_TRAIN}" "${FINAL_TRAIN}" "${BRANCH_DIR}/train_wrap_stats.json" "${BRANCH_DIR}/train_wrap_samples.json"
run_wrap "${STAGE1_DEV}" "${FINAL_DEV}" "${BRANCH_DIR}/dev_wrap_stats.json" "${BRANCH_DIR}/dev_wrap_samples.json"

echo "[STAGE] Write runtime budget schedule and validation summary"
python3 - "${BRANCH_DIR}" "${FINAL_TRAIN}" "${FINAL_DEV}" <<'PY'
import json
import re
import sys
from pathlib import Path

branch = Path(sys.argv[1])
train = Path(sys.argv[2])
dev = Path(sys.argv[3])

schedule = {
    "version": "cap16_denoise_budget_ttag_v1",
    "dataset": "acl_tagged_raw",
    "lang": "ja",
    "retriever": "HN1024",
    "tau": 0.78,
    "empty_term_map_policy": "omit",
    "runtime_budget_by_lm": {
        "1": {"max_terms": 6, "note": "lowest latency; strongest noise pressure"},
        "2": {"max_terms": 8, "note": "balanced low-latency budget"},
        "3": {"max_terms": 10, "note": "moderate latency budget"},
        "4": {"max_terms": 10, "note": "BLEU-preserving budget from denoise SFT"},
    },
    "training_budget_mix": {"choices": [6, 8, 10], "weights": [0.45, 0.35, 0.20]},
    "no_gt_max_terms": 4,
    "no_gt_empty_prob": 0.35,
    "score_dropout": {
        "low_score_cutoff": 0.82,
        "mid_score_cutoff": 0.85,
        "low_score_keep_prob": 0.25,
        "mid_score_keep_prob": 0.60,
        "high_score_keep_prob": 0.90,
        "supported_non_gt_keep_prob": 0.85,
    },
    "assistant_tag_template": "<t>{translation}</t>",
}
branch.joinpath("runtime_termmap_budget_schedule.json").write_text(
    json.dumps(schedule, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

latin = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]")

def is_latin(ch: str) -> bool:
    return bool(ch) and bool(latin.fullmatch(ch))

def count_terms(content: str) -> int:
    content = str(content or "")
    marker = "term_map:"
    idx = content.find(marker)
    if idx < 0:
        return 0
    body = content[idx + len(marker):].strip()
    return sum(1 for line in body.splitlines() if "=" in line)

def validate(path: Path) -> dict:
    rows = chunks = term_chunks = max_terms = tagged_rows = 0
    malformed = latin_boundary_cuts = legacy_term_messages = 0
    for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
        obj = json.loads(line)
        rows += 1
        messages = obj.get("messages")
        audios = obj.get("audios")
        gt_terms_by_chunk = obj.get("gt_terms_by_chunk")
        if not isinstance(messages, list) or not isinstance(audios, list) or not isinstance(gt_terms_by_chunk, list):
            raise SystemExit(f"Malformed row at {path}:{line_no}")
        user_indices = [
            i for i, m in enumerate(messages)
            if m.get("role") == "user" and str(m.get("content") or "").startswith("<audio>")
        ]
        if len(user_indices) != len(audios) or len(gt_terms_by_chunk) != len(audios):
            raise SystemExit(f"user/audio/gt mismatch at {path}:{line_no}")
        chunks += len(audios)
        for idx in user_indices:
            n = count_terms(messages[idx].get("content", ""))
            term_chunks += int(n > 0)
            max_terms = max(max_terms, n)
        row_has_tag = False
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            text = str(msg.get("content") or "")
            if "<term>" in text or "</term>" in text:
                legacy_term_messages += 1
            row_has_tag = row_has_tag or "<t>" in text
            if text.count("<t>") != text.count("</t>"):
                malformed += 1
            search = 0
            while True:
                start = text.find("<t>", search)
                if start < 0:
                    break
                end = text.find("</t>", start + 3)
                if end < 0:
                    malformed += 1
                    break
                inner_start = start + 3
                inner_end = end
                before = text[start - 1] if start > 0 else ""
                after_i = end + 4
                after = text[after_i] if after_i < len(text) else ""
                if inner_start < inner_end:
                    latin_boundary_cuts += int(is_latin(before) and is_latin(text[inner_start]))
                    latin_boundary_cuts += int(is_latin(after) and is_latin(text[inner_end - 1]))
                search = after_i
        tagged_rows += int(row_has_tag)
    return {
        "path": str(path),
        "rows": rows,
        "chunks": chunks,
        "termmap_chunks": term_chunks,
        "termmap_chunk_rate": term_chunks / max(1, chunks),
        "max_termmap_entries": max_terms,
        "tagged_rows": tagged_rows,
        "malformed_ttag_messages": malformed,
        "legacy_term_tag_messages": legacy_term_messages,
        "latin_boundary_cut_messages": latin_boundary_cuts,
    }

summary = {
    "status": "success",
    "short_tag": "<t>{translation}</t>",
    "train": validate(train),
    "dev": validate(dev),
}
for split in ("train", "dev"):
    row = summary[split]
    if row["malformed_ttag_messages"] or row["legacy_term_tag_messages"] or row["latin_boundary_cut_messages"]:
        raise SystemExit(f"Validation failed for {split}: {row}")
    if row["max_termmap_entries"] > 16:
        raise SystemExit(f"Unexpected max termmap entries for {split}: {row['max_termmap_entries']}")
branch.joinpath("validation_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "[DONE] JA cap16 denoise-budget short-tag data ready: ${BRANCH_DIR}"
