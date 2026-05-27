#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
DATA_ROOT="${DATA_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo}"
PARENT_DIR="${PARENT_DIR_OVERRIDE:-${DATA_ROOT}/speech_llm_de_cap16_denoise_budget_20260525/de/hn1024_tau078_cap16_denoise_budget_v1}"
BRANCH_DIR="${BRANCH_DIR_OVERRIDE:-${DATA_ROOT}/speech_llm_de_cap16_denoise_budget_20260525/de/hn1024_tau078_cap16_denoise_budget_ttag_v1}"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-${DATA_ROOT}/logs/de_cap16_denoise_budget_ttag_20260525}"

WRAP_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/wrap_assistant_term_targets.py"

STAGE1_TRAIN="${PARENT_DIR}/train_s_de_retriever_hn1024_tau078_cap16_denoise_budget_stage1.jsonl"
STAGE1_DEV="${PARENT_DIR}/dev_s_de_retriever_hn1024_tau078_cap16_denoise_budget_stage1_first355.jsonl"
FINAL_TRAIN="${BRANCH_DIR}/train_s_de_retriever_hn1024_tau078_cap16_denoise_budget_ttag_exactboundary.jsonl"
FINAL_DEV="${BRANCH_DIR}/dev_s_de_retriever_hn1024_tau078_cap16_denoise_budget_ttag_exactboundary_first355.jsonl"

EXCLUDE_SOURCE_TOKENS="${EXCLUDE_SOURCE_TOKENS_OVERRIDE:-this,that,these,those,his,her,hers,him,he,she,it,its,they,them,their,theirs,you,your,yours,we,our,ours,i,me,my,mine,myself,yourself,himself,herself,itself,ourselves,yourselves,themselves,what,which,who,whom,whose,someone,somebody,something,anyone,anybody,anything,everyone,everybody,everything}"
FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${BRANCH_DIR}" "${LOG_ROOT}"

for p in "${WRAP_SCRIPT}" "${STAGE1_TRAIN}" "${STAGE1_DEV}"; do
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
    --lang-code de \
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
  "${FINAL_TRAIN}" "${FINAL_DEV}" \
  "${BRANCH_DIR}/train_wrap_stats.json" "${BRANCH_DIR}/train_wrap_samples.json" \
  "${BRANCH_DIR}/dev_wrap_stats.json" "${BRANCH_DIR}/dev_wrap_samples.json" \
  "${BRANCH_DIR}/validation_summary.json"

echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] PARENT_DIR=${PARENT_DIR}"
echo "[INFO] BRANCH_DIR=${BRANCH_DIR}"
df -h /mnt/gemini/data1

echo "[STAGE] Wrap assistant GT target translations with short <t> tags"
run_wrap "${STAGE1_TRAIN}" "${FINAL_TRAIN}" "${BRANCH_DIR}/train_wrap_stats.json" "${BRANCH_DIR}/train_wrap_samples.json"
run_wrap "${STAGE1_DEV}" "${FINAL_DEV}" "${BRANCH_DIR}/dev_wrap_stats.json" "${BRANCH_DIR}/dev_wrap_samples.json"

echo "[STAGE] Validate short-tag JSONLs"
python3 - "${PARENT_DIR}" "${BRANCH_DIR}" "${FINAL_TRAIN}" "${FINAL_DEV}" <<'PY'
import json
import re
import sys
from pathlib import Path

parent = Path(sys.argv[1])
branch = Path(sys.argv[2])
train = Path(sys.argv[3])
dev = Path(sys.argv[4])

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
    "parent_dir": str(parent),
    "short_tag": "<t>{translation}</t>",
    "train": validate(train),
    "dev": validate(dev),
}
for split in ("train", "dev"):
    row = summary[split]
    if row["malformed_ttag_messages"] or row["legacy_term_tag_messages"] or row["latin_boundary_cut_messages"]:
        raise SystemExit(f"Validation failed for {split}: {row}")
    if row["max_termmap_entries"] > 12:
        raise SystemExit(f"Unexpected max termmap entries for {split}: {row['max_termmap_entries']}")
branch.joinpath("validation_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "[DONE] Short-tag data ready: ${BRANCH_DIR}"
