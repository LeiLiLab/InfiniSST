#!/usr/bin/env bash
set -euo pipefail

if [[ -d /mnt/taurus/home/jiaxuanluo/InfiniSST ]]; then
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
else
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
fi

SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/build_source_glossary_exact_gt_terms.py"
OUT_DIR="${OUT_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_srcmatch100k_future_ref_gt_zh_20260522}"

TRAIN_INPUT_JSONL="${TRAIN_INPUT_JSONL_OVERRIDE:-/mnt/gemini/data/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl}"
DEV_INPUT_JSONL="${DEV_INPUT_JSONL_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/dev_s_zh_v4_ner_baseline_aligned_freq_k20.jsonl}"
TRAIN_TSV="${TRAIN_TSV_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv}"
DEV_TSV="${DEV_TSV_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/dev_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv}"
GLOSSARY_JSON="${GLOSSARY_JSON_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample100000_zh100k_train_gt.json}"

TRAIN_OUTPUT_JSONL="${TRAIN_OUTPUT_JSONL_OVERRIDE:-${OUT_DIR}/train_s_zh_srcmatch100k_future_ref_gt.jsonl}"
DEV_OUTPUT_JSONL="${DEV_OUTPUT_JSONL_OVERRIDE:-${OUT_DIR}/dev_s_zh_srcmatch100k_future_ref_gt.jsonl}"
TRAIN_STATS_JSON="${TRAIN_STATS_JSON_OVERRIDE:-${OUT_DIR}/train_s_zh_srcmatch100k_future_ref_gt_stats.json}"
DEV_STATS_JSON="${DEV_STATS_JSON_OVERRIDE:-${OUT_DIR}/dev_s_zh_srcmatch100k_future_ref_gt_stats.json}"
TRAIN_SAMPLE_JSON="${TRAIN_SAMPLE_JSON_OVERRIDE:-${OUT_DIR}/train_s_zh_srcmatch100k_future_ref_gt_samples.json}"
DEV_SAMPLE_JSON="${DEV_SAMPLE_JSON_OVERRIDE:-${OUT_DIR}/dev_s_zh_srcmatch100k_future_ref_gt_samples.json}"
SUMMARY_JSON="${SUMMARY_JSON_OVERRIDE:-${OUT_DIR}/srcmatch100k_future_ref_gt_summary.json}"

PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-python3}"

for p in "${SCRIPT}" "${TRAIN_INPUT_JSONL}" "${DEV_INPUT_JSONL}" "${TRAIN_TSV}" "${DEV_TSV}" "${GLOSSARY_JSON}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

mkdir -p "${OUT_DIR}"
cd "${ROOT_DIR}"

run_one() {
  local split="$1"
  local input_jsonl="$2"
  local input_tsv="$3"
  local output_jsonl="$4"
  local stats_json="$5"
  local sample_json="$6"

  echo "[INFO] Building ${split} future-ref source-match GT: ${output_jsonl}"
  "${PYTHON_BIN}" "${SCRIPT}" \
    --input-jsonl "${input_jsonl}" \
    --input-tsv "${input_tsv}" \
    --glossary-json "${GLOSSARY_JSON}" \
    --output-jsonl "${output_jsonl}" \
    --stats-json "${stats_json}" \
    --sample-json "${sample_json}" \
    --lang-code zh \
    --max-words 6 \
    --min-norm-chars 2 \
    --target-match-policy future_ref
}

run_one train "${TRAIN_INPUT_JSONL}" "${TRAIN_TSV}" "${TRAIN_OUTPUT_JSONL}" "${TRAIN_STATS_JSON}" "${TRAIN_SAMPLE_JSON}"
run_one dev "${DEV_INPUT_JSONL}" "${DEV_TSV}" "${DEV_OUTPUT_JSONL}" "${DEV_STATS_JSON}" "${DEV_SAMPLE_JSON}"

"${PYTHON_BIN}" - "${TRAIN_OUTPUT_JSONL}" "${DEV_OUTPUT_JSONL}" "${TRAIN_STATS_JSON}" "${DEV_STATS_JSON}" "${SUMMARY_JSON}" <<'PY'
import json
import sys
from pathlib import Path

train_jsonl, dev_jsonl, train_stats, dev_stats, summary_json = map(Path, sys.argv[1:])


def assistant_future_text(messages, audio_msg_idx):
    return "\n".join(
        str(messages[idx].get("content") or "")
        for idx in range(audio_msg_idx + 1, len(messages))
        if messages[idx].get("role") == "assistant"
    )


def validate(path):
    checked = 0
    rows = 0
    chunks = 0
    for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        rows += 1
        obj = json.loads(line)
        messages = obj.get("messages")
        gt_by_chunk = obj.get("gt_terms_by_chunk")
        if not isinstance(messages, list) or not isinstance(gt_by_chunk, list):
            raise SystemExit(f"[ERROR] malformed row {path}:{line_no}")
        audio_idxs = [
            idx for idx, msg in enumerate(messages)
            if msg.get("role") == "user" and "<audio>" in str(msg.get("content") or "")
        ]
        if len(audio_idxs) != len(gt_by_chunk):
            raise SystemExit(f"[ERROR] audio/gt mismatch {path}:{line_no}: {len(audio_idxs)} != {len(gt_by_chunk)}")
        chunks += len(audio_idxs)
        for chunk_idx, msg_idx in enumerate(audio_idxs):
            future = assistant_future_text(messages, msg_idx)
            for term in gt_by_chunk[chunk_idx]:
                translation = str(term.get("zh") or term.get("translation") or "").strip()
                checked += 1
                if not translation or translation not in future:
                    raise SystemExit(
                        f"[ERROR] future-ref violation {path}:{line_no} chunk={chunk_idx} term={term!r}"
                    )
    return {"rows": rows, "chunks": chunks, "checked_gt_terms": checked}


summary = {
    "train_stats": json.loads(train_stats.read_text(encoding="utf-8")),
    "dev_stats": json.loads(dev_stats.read_text(encoding="utf-8")),
    "train_validation": validate(train_jsonl),
    "dev_validation": validate(dev_jsonl),
}
for split in ("train", "dev"):
    stats = summary[f"{split}_stats"]
    if stats["dropped_rows"] != 0:
        raise SystemExit(f"[ERROR] {split} dropped_rows={stats['dropped_rows']}")
    if stats["exact_gt_terms_total"] != summary[f"{split}_validation"]["checked_gt_terms"]:
        raise SystemExit(f"[ERROR] {split} stats/output GT count mismatch")
    if stats["target_match_policy"] != "future_ref":
        raise SystemExit(f"[ERROR] {split} target_match_policy={stats['target_match_policy']}")

summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "summary_json": str(summary_json),
    "train": {
        "rows": summary["train_validation"]["rows"],
        "chunks": summary["train_validation"]["chunks"],
        "source_exact_terms_total": summary["train_stats"]["source_exact_terms_total"],
        "future_ref_gt_terms": summary["train_stats"]["exact_gt_terms_total"],
        "target_match_kept_term_rate": summary["train_stats"]["target_match_kept_term_rate"],
        "chunks_with_gt_rate": summary["train_stats"]["chunks_with_exact_gt_rate"],
        "avg_gt_terms_per_chunk": summary["train_stats"]["avg_exact_gt_terms_per_chunk"],
    },
    "dev": {
        "rows": summary["dev_validation"]["rows"],
        "chunks": summary["dev_validation"]["chunks"],
        "source_exact_terms_total": summary["dev_stats"]["source_exact_terms_total"],
        "future_ref_gt_terms": summary["dev_stats"]["exact_gt_terms_total"],
        "target_match_kept_term_rate": summary["dev_stats"]["target_match_kept_term_rate"],
        "chunks_with_gt_rate": summary["dev_stats"]["chunks_with_exact_gt_rate"],
        "avg_gt_terms_per_chunk": summary["dev_stats"]["avg_exact_gt_terms_per_chunk"],
    },
}, ensure_ascii=False, indent=2), flush=True)
PY

echo "[INFO] Future-ref source-match GT ready: ${OUT_DIR}"
