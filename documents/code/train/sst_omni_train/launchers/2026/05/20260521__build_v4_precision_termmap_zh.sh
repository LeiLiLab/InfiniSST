#!/usr/bin/env bash
set -euo pipefail

if [[ -d /mnt/taurus/home/jiaxuanluo/InfiniSST ]]; then
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
else
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
fi

PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-python3}"
SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/build_robust_termmap_sft.py"

SOURCE_DIR="${SOURCE_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_retriever_timeline_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260519}"
OUT_DIR="${OUT_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_v4_precision_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260521}"

TRAIN_INPUT="${TRAIN_INPUT_OVERRIDE:-${SOURCE_DIR}/train_s_zh_retriever_timeline_lh1b88kw_tau073_srcmatch100k_k10_lb1p92.jsonl}"
DEV_INPUT="${DEV_INPUT_OVERRIDE:-${SOURCE_DIR}/dev_s_zh_retriever_timeline_lh1b88kw_tau073_srcmatch100k_k10_lb1p92.jsonl}"

for p in "${SCRIPT}" "${TRAIN_INPUT}" "${DEV_INPUT}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

mkdir -p "${OUT_DIR}"
cd "${ROOT_DIR}"

run_one() {
  local split="$1"
  local input="$2"
  local output="${OUT_DIR}/${split}_s_zh_v4_precision_termmap_lh1b88kw_tau073_srcmatch100k.jsonl"
  local stats="${OUT_DIR}/${split}_s_zh_v4_precision_termmap_stats.json"
  local sample="${OUT_DIR}/${split}_s_zh_v4_precision_termmap_samples.json"
  echo "[INFO] Building ${split}/precision: ${output}"
  "${PYTHON_BIN}" "${SCRIPT}" \
    --input-jsonl "${input}" \
    --output-jsonl "${output}" \
    --stats-json "${stats}" \
    --sample-json "${sample}" \
    --variant precision \
    --term-map-style plain \
    --lang-code zh \
    --seed 20260521
}

run_one train "${TRAIN_INPUT}"
run_one dev "${DEV_INPUT}"

"${PYTHON_BIN}" - "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
summary = {}
for p in sorted(out_dir.glob("*_stats.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    if d["dropped_rows"]:
        raise SystemExit(f"[ERROR] dropped rows in {p}: {d['dropped_rows']} {d.get('dropped_reasons')}")
    if d["rows_written"] <= 0 or d["chunks"] <= 0:
        raise SystemExit(f"[ERROR] empty output stats in {p}")
    key = p.name.replace("_stats.json", "")
    summary[key] = {
        "rows_written": d["rows_written"],
        "chunks": d["chunks"],
        "mode_counts": d["mode_counts"],
        "gt_term_in_term_map_rate": d["gt_term_in_term_map_rate"],
        "gt_chunk_any_term_in_map_rate": d["gt_chunk_any_term_in_map_rate"],
        "gt_chunk_all_terms_in_map_rate": d["gt_chunk_all_terms_in_map_rate"],
        "no_gt_nonempty_term_map_rate": d["no_gt_nonempty_term_map_rate"],
        "avg_term_map_entries_per_chunk": d["avg_term_map_entries_per_chunk"],
        "avg_non_gt_entries_per_chunk": d["avg_non_gt_entries_per_chunk"],
        "term_map_size_hist": d["term_map_size_hist"],
    }
summary_path = out_dir / "v4_precision_termmap_summary.json"
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"[INFO] Wrote summary: {summary_path}")
PY

echo "[INFO] V4 precision retriever term_map data ready: ${OUT_DIR}"
