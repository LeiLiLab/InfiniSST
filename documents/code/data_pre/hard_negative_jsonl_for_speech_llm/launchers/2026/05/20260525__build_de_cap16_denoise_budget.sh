#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
DATA_ROOT="${DATA_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo}"
SRC_DIR="${DATA_ROOT}/speech_llm_deja_termmap_ablation_cap16_exactboundary_20260525/de/retriever_hn1024_tau078_cap16_exactboundary"
OUT_ROOT="${OUT_ROOT_OVERRIDE:-${DATA_ROOT}/speech_llm_de_cap16_denoise_budget_20260525}"
BRANCH_DIR="${OUT_ROOT}/de/hn1024_tau078_cap16_denoise_budget_v1"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-${DATA_ROOT}/logs/de_cap16_denoise_budget_20260525}"

REBUILD_SCRIPT="${ROOT_DIR}/documents/code/data_pre/hard_negative_jsonl_for_speech_llm/rebuild_termmap_denoise_budget.py"
WRAP_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/wrap_assistant_term_targets.py"
AUDIT_SCRIPT="${ROOT_DIR}/documents/code/tools/audit_training_jsonl.py"

TRAIN_RETRIEVED="${SRC_DIR}/train_s_de_retriever_results_hn1024_tau078.jsonl"
DEV_RETRIEVED="${SRC_DIR}/dev_s_de_retriever_results_hn1024_tau078_first355.jsonl"
STAGE1_TRAIN="${BRANCH_DIR}/train_s_de_retriever_hn1024_tau078_cap16_denoise_budget_stage1.jsonl"
STAGE1_DEV="${BRANCH_DIR}/dev_s_de_retriever_hn1024_tau078_cap16_denoise_budget_stage1_first355.jsonl"
FINAL_TRAIN="${BRANCH_DIR}/train_s_de_retriever_hn1024_tau078_cap16_denoise_budget_gttermwrap_exactboundary.jsonl"
FINAL_DEV="${BRANCH_DIR}/dev_s_de_retriever_hn1024_tau078_cap16_denoise_budget_gttermwrap_exactboundary_first355.jsonl"

EXCLUDE_SOURCE_TOKENS="${EXCLUDE_SOURCE_TOKENS_OVERRIDE:-this,that,these,those,his,her,hers,him,he,she,it,its,they,them,their,theirs,you,your,yours,we,our,ours,i,me,my,mine,myself,yourself,himself,herself,itself,ourselves,yourselves,themselves,what,which,who,whom,whose,someone,somebody,something,anyone,anybody,anything,everyone,everybody,everything}"
FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"
RUN_FULL_AUDIT="${RUN_FULL_AUDIT:-0}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${BRANCH_DIR}" "${LOG_ROOT}"

for p in "${REBUILD_SCRIPT}" "${WRAP_SCRIPT}" "${AUDIT_SCRIPT}" "${TRAIN_RETRIEVED}" "${DEV_RETRIEVED}"; do
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
    --target-lang de \
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
    --lang-code de \
    --tag-template '<term>{translation}</term>' \
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
  "${BRANCH_DIR}/train_audit.md" "${BRANCH_DIR}/train_audit_summary.json" \
  "${BRANCH_DIR}/dev_audit.md" "${BRANCH_DIR}/dev_audit_summary.json" \
  "${BRANCH_DIR}/runtime_termmap_budget_schedule.json" \
  "${BRANCH_DIR}/validation_summary.json"

echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] SRC_DIR=${SRC_DIR}"
echo "[INFO] BRANCH_DIR=${BRANCH_DIR}"
df -h /mnt/gemini/data1 || true

echo "[STAGE] Rebuild train/dev term maps with denoise budget policy"
run_rebuild "${TRAIN_RETRIEVED}" "${STAGE1_TRAIN}" "${BRANCH_DIR}/train_rebuild_stats.json" "${BRANCH_DIR}/train_rebuild_samples.json"
run_rebuild "${DEV_RETRIEVED}" "${STAGE1_DEV}" "${BRANCH_DIR}/dev_rebuild_stats.json" "${BRANCH_DIR}/dev_rebuild_samples.json"

echo "[STAGE] Wrap assistant GT target translations"
run_wrap "${STAGE1_TRAIN}" "${FINAL_TRAIN}" "${BRANCH_DIR}/train_wrap_stats.json" "${BRANCH_DIR}/train_wrap_samples.json"
run_wrap "${STAGE1_DEV}" "${FINAL_DEV}" "${BRANCH_DIR}/dev_wrap_stats.json" "${BRANCH_DIR}/dev_wrap_samples.json"

if [[ "${RUN_FULL_AUDIT}" == "1" ]]; then
  echo "[STAGE] Audit final JSONLs"
  python3 "${AUDIT_SCRIPT}" \
    --input-jsonl "${FINAL_TRAIN}" \
    --output-md "${BRANCH_DIR}/train_audit.md" \
    --summary-json "${BRANCH_DIR}/train_audit_summary.json" \
    --wav-sample-rows 0 \
    --whisper-samples 0
  python3 "${AUDIT_SCRIPT}" \
    --input-jsonl "${FINAL_DEV}" \
    --output-md "${BRANCH_DIR}/dev_audit.md" \
    --summary-json "${BRANCH_DIR}/dev_audit_summary.json" \
    --wav-sample-rows 0 \
    --whisper-samples 0
else
  echo "[SKIP] RUN_FULL_AUDIT=0; using built-in lightweight validation only"
fi

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
    "version": "cap16_denoise_budget_v1",
    "dataset": "acl_tagged_raw",
    "lang": "de",
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
}
(branch / "runtime_termmap_budget_schedule.json").write_text(
    json.dumps(schedule, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

latin = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]")

def latin_char(ch):
    return bool(ch) and bool(latin.fullmatch(ch))

def count_terms(content):
    content = str(content or "")
    idx = content.find("term_map:")
    if idx < 0:
        return 0
    body = content[idx + len("term_map:"):].strip()
    return sum(1 for line in body.splitlines() if "=" in line)

def validate(path):
    rows = chunks = term_chunks = max_terms = malformed = latin_cut = tagged_rows = 0
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            obj = json.loads(line)
            rows += 1
            messages = obj.get("messages")
            audios = obj.get("audios")
            gt = obj.get("gt_terms_by_chunk")
            if not isinstance(messages, list) or not isinstance(audios, list) or not isinstance(gt, list):
                raise SystemExit(f"malformed row {path}:{line_no}")
            user_indices = [
                i for i, m in enumerate(messages)
                if m.get("role") == "user" and str(m.get("content") or "").startswith("<audio>")
            ]
            if len(user_indices) != len(audios) or len(gt) != len(audios):
                raise SystemExit(f"user/audio/gt mismatch {path}:{line_no}")
            chunks += len(audios)
            for idx in user_indices:
                n = count_terms(messages[idx].get("content"))
                max_terms = max(max_terms, n)
                term_chunks += int(n > 0)
            row_tagged = False
            for m in messages:
                if m.get("role") != "assistant":
                    continue
                text = str(m.get("content") or "")
                row_tagged |= "<term>" in text
                if text.count("<term>") != text.count("</term>"):
                    malformed += 1
                search = 0
                while True:
                    s = text.find("<term>", search)
                    if s < 0:
                        break
                    e = text.find("</term>", s + 6)
                    if e < 0:
                        malformed += 1
                        break
                    inner_s = s + 6
                    inner_e = e
                    before = text[s - 1] if s > 0 else ""
                    after_i = e + 7
                    after = text[after_i] if after_i < len(text) else ""
                    if inner_s < inner_e:
                        latin_cut += int(latin_char(before) and latin_char(text[inner_s]))
                        latin_cut += int(latin_char(after) and latin_char(text[inner_e - 1]))
                    search = after_i
            tagged_rows += int(row_tagged)
    return {
        "path": str(path),
        "rows": rows,
        "chunks": chunks,
        "termmap_chunks": term_chunks,
        "termmap_chunk_rate": term_chunks / max(1, chunks),
        "max_termmap_entries": max_terms,
        "tagged_rows": tagged_rows,
        "malformed_tag_messages": malformed,
        "latin_boundary_cut_messages": latin_cut,
    }

summary = {
    "status": "success",
    "train": validate(train),
    "dev": validate(dev),
    "runtime_budget_schedule": str(branch / "runtime_termmap_budget_schedule.json"),
    "rebuild_stats": {
        "train": str(branch / "train_rebuild_stats.json"),
        "dev": str(branch / "dev_rebuild_stats.json"),
    },
    "wrap_stats": {
        "train": str(branch / "train_wrap_stats.json"),
        "dev": str(branch / "dev_wrap_stats.json"),
    },
}
if summary["train"]["malformed_tag_messages"] or summary["train"]["latin_boundary_cut_messages"]:
    raise SystemExit("train tag validation failed")
if summary["dev"]["malformed_tag_messages"] or summary["dev"]["latin_boundary_cut_messages"]:
    raise SystemExit("dev tag validation failed")
(branch / "validation_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
PY

echo "[DONE] De cap16 denoise budget data ready: ${BRANCH_DIR}"
