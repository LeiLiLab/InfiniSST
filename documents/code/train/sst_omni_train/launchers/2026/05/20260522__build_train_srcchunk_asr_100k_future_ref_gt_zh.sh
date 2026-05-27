#!/usr/bin/env bash
set -euo pipefail

if [[ -d /mnt/taurus/home/jiaxuanluo/InfiniSST ]]; then
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
else
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
fi

SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/build_source_glossary_exact_gt_terms.py"
OUT_DIR="${OUT_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_train_srcchunk_asr_100k_future_ref_gt_zh_20260522}"

TRAIN_INPUT_JSONL="${TRAIN_INPUT_JSONL_OVERRIDE:-/mnt/gemini/data/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl}"
TRAIN_TSV="${TRAIN_TSV_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv}"
GLOSSARY_JSON="${GLOSSARY_JSON_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample100000_zh100k_train_gt.json}"

OUTPUT_JSONL="${OUTPUT_JSONL_OVERRIDE:-${OUT_DIR}/train_s_zh_srcchunk_asr_100k_future_ref_gt_termmap_none.jsonl}"
STATS_JSON="${STATS_JSON_OVERRIDE:-${OUT_DIR}/train_s_zh_srcchunk_asr_100k_future_ref_gt_stats.json}"
SAMPLE_JSON="${SAMPLE_JSON_OVERRIDE:-${OUT_DIR}/train_s_zh_srcchunk_asr_100k_future_ref_gt_samples.json}"
SUMMARY_JSON="${SUMMARY_JSON_OVERRIDE:-${OUT_DIR}/train_srcchunk_asr_100k_future_ref_gt_summary.json}"

PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-python3}"

for p in "${SCRIPT}" "${TRAIN_INPUT_JSONL}" "${TRAIN_TSV}" "${GLOSSARY_JSON}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

mkdir -p "${OUT_DIR}"
cd "${ROOT_DIR}"

echo "[INFO] Building train source-chunk-ASR 100k future-ref GT: ${OUTPUT_JSONL}"
"${PYTHON_BIN}" "${SCRIPT}" \
  --input-jsonl "${TRAIN_INPUT_JSONL}" \
  --input-tsv "${TRAIN_TSV}" \
  --glossary-json "${GLOSSARY_JSON}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --stats-json "${STATS_JSON}" \
  --sample-json "${SAMPLE_JSON}" \
  --lang-code zh \
  --max-words 6 \
  --min-norm-chars 2 \
  --target-match-policy future_ref \
  --term-map-output-policy none

"${PYTHON_BIN}" - "${OUTPUT_JSONL}" "${STATS_JSON}" "${SUMMARY_JSON}" <<'PY'
import json
import sys
from pathlib import Path

output_jsonl, stats_json, summary_json = map(Path, sys.argv[1:])
stats = json.loads(stats_json.read_text(encoding="utf-8"))

rows = chunks = checked_terms = 0
termmap_none_violations = 0
source_chunk_field_violations = 0
future_ref_violations = 0

with output_jsonl.open(encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue
        rows += 1
        obj = json.loads(line)
        messages = obj.get("messages")
        audios = obj.get("audios")
        gt_by_chunk = obj.get("gt_terms_by_chunk")
        src_by_chunk = obj.get("source_chunk_asr_by_chunk")
        if not isinstance(messages, list) or not isinstance(audios, list):
            raise SystemExit(f"[ERROR] malformed messages/audios at line {line_no}")
        if not isinstance(gt_by_chunk, list) or not isinstance(src_by_chunk, list):
            raise SystemExit(f"[ERROR] missing gt/source chunk fields at line {line_no}")
        audio_idxs = [
            idx for idx, msg in enumerate(messages)
            if msg.get("role") == "user" and "<audio>" in str(msg.get("content") or "")
        ]
        if len(audio_idxs) != len(audios) or len(gt_by_chunk) != len(audios) or len(src_by_chunk) != len(audios):
            raise SystemExit(
                f"[ERROR] chunk count mismatch line {line_no}: "
                f"audios={len(audios)} audio_msgs={len(audio_idxs)} gt={len(gt_by_chunk)} src={len(src_by_chunk)}"
            )
        chunks += len(audios)
        for chunk_idx, msg_idx in enumerate(audio_idxs):
            if not isinstance(src_by_chunk[chunk_idx], str):
                source_chunk_field_violations += 1
            content = str(messages[msg_idx].get("content") or "")
            if "term_map:NONE" not in content:
                termmap_none_violations += 1
            future = "\n".join(
                str(messages[j].get("content") or "")
                for j in range(msg_idx + 1, len(messages))
                if messages[j].get("role") == "assistant"
            )
            for term in gt_by_chunk[chunk_idx]:
                checked_terms += 1
                translation = str(term.get("zh") or term.get("translation") or "").strip()
                if not translation or translation not in future:
                    future_ref_violations += 1

if rows != stats["rows_written"] or chunks != stats["chunks_total"]:
    raise SystemExit(f"[ERROR] stats/output mismatch rows={rows}/{stats['rows_written']} chunks={chunks}/{stats['chunks_total']}")
if termmap_none_violations or source_chunk_field_violations or future_ref_violations:
    raise SystemExit(
        f"[ERROR] validation failed: termmap_none={termmap_none_violations}, "
        f"source_field={source_chunk_field_violations}, future_ref={future_ref_violations}"
    )
summary = {
    "output_jsonl": str(output_jsonl),
    "stats_json": str(stats_json),
    "rows": rows,
    "chunks": chunks,
    "checked_gt_terms": checked_terms,
    "source_exact_terms_total": stats["source_exact_terms_total"],
    "future_ref_gt_terms": stats["exact_gt_terms_total"],
    "target_match_kept_term_rate": stats["target_match_kept_term_rate"],
    "chunks_with_gt_rate": stats["chunks_with_exact_gt_rate"],
    "avg_gt_terms_per_chunk": stats["avg_exact_gt_terms_per_chunk"],
    "termmap_none_violations": termmap_none_violations,
    "source_chunk_field_violations": source_chunk_field_violations,
    "future_ref_violations": future_ref_violations,
}
summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
PY

echo "[INFO] Train source-chunk-ASR future-ref GT ready: ${OUT_DIR}"
