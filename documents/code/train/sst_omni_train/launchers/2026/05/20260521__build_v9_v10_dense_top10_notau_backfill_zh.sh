#!/usr/bin/env bash
set -euo pipefail

if [[ -d /mnt/taurus/home/jiaxuanluo/InfiniSST ]]; then
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
else
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
fi

BASE_TIMELINE_LAUNCHER="${ROOT_DIR}/documents/code/train/sst_omni_train/launchers/2026/05/20260519__build_retriever_timeline_termmap_zh_lh1b88kw_tau073.sh"
DENSE_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/build_dense_topk_backfill_sft.py"

SRCMATCH_DIR="${SRCMATCH_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_retriever_timeline_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260519}"
NO_TAU_DIR="${NO_TAU_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_retriever_timeline_termmap_zh_lh1b88kw_top10_notau_srcmatch100k_20260521}"
OUT_DIR="${OUT_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_v9_v10_dense_top10_notau_backfill_zh_lh1b88kw_srcmatch100k_20260521}"

SRCMATCH_TRAIN_JSONL="${SRCMATCH_TRAIN_JSONL_OVERRIDE:-${SRCMATCH_DIR}/train_s_zh_srcmatch100k_gt.jsonl}"
SRCMATCH_DEV_JSONL="${SRCMATCH_DEV_JSONL_OVERRIDE:-${SRCMATCH_DIR}/dev_s_zh_srcmatch100k_gt.jsonl}"
GLOSSARY_JSON="${GLOSSARY_JSON_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample100000_zh100k_train_gt.json}"
MODEL_PATH="${MODEL_PATH_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"

NO_TAU_TRAIN_JSONL="${NO_TAU_DIR}/train_s_zh_retriever_timeline_lh1b88kw_top10_notau_srcmatch100k_lb1p92.jsonl"
NO_TAU_DEV_JSONL="${NO_TAU_DIR}/dev_s_zh_retriever_timeline_lh1b88kw_top10_notau_srcmatch100k_lb1p92.jsonl"

CONDA_PREFIX="${CONDA_PREFIX_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv}"
PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-${CONDA_PREFIX}/bin/python3}"
GPU_DEVICES_CSV="${GPU_DEVICES_CSV_OVERRIDE:-6,7}"
NUM_SHARDS="${NUM_SHARDS_OVERRIDE:-2}"

for p in "${BASE_TIMELINE_LAUNCHER}" "${DENSE_SCRIPT}" "${SRCMATCH_TRAIN_JSONL}" "${SRCMATCH_DEV_JSONL}" "${GLOSSARY_JSON}" "${MODEL_PATH}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Missing executable PYTHON_BIN: ${PYTHON_BIN}" >&2
  exit 3
fi

mkdir -p "${NO_TAU_DIR}" "${OUT_DIR}"
cd "${ROOT_DIR}"

echo "[INFO] Building/rebuilding no-tau top10 timeline retriever source"
OUT_DIR_OVERRIDE="${NO_TAU_DIR}" \
SHARD_DIR_OVERRIDE="${NO_TAU_DIR}/shards" \
TRAIN_INPUT_JSONL_OVERRIDE="${SRCMATCH_TRAIN_JSONL}" \
VAL_INPUT_JSONL_OVERRIDE="${SRCMATCH_DEV_JSONL}" \
GLOSSARY_JSON_OVERRIDE="${GLOSSARY_JSON}" \
MODEL_PATH_OVERRIDE="${MODEL_PATH}" \
TEXT_INDEX_PATH_OVERRIDE="${NO_TAU_DIR}/lh1b88kw_zh100k_text_index.pt" \
TRAIN_OUTPUT_JSONL_OVERRIDE="${NO_TAU_TRAIN_JSONL}" \
VAL_OUTPUT_JSONL_OVERRIDE="${NO_TAU_DEV_JSONL}" \
TRAIN_STATS_JSON_OVERRIDE="${NO_TAU_DIR}/train_retriever_timeline_top10_notau_stats.json" \
VAL_STATS_JSON_OVERRIDE="${NO_TAU_DIR}/dev_retriever_timeline_top10_notau_stats.json" \
TRAIN_SAMPLE_JSON_OVERRIDE="${NO_TAU_DIR}/train_retriever_timeline_top10_notau_samples.json" \
VAL_SAMPLE_JSON_OVERRIDE="${NO_TAU_DIR}/dev_retriever_timeline_top10_notau_samples.json" \
TRAIN_SAMPLE_CHUNKS_JSON_OVERRIDE="${NO_TAU_DIR}/train_retriever_timeline_top10_notau_sample_chunks.json" \
VAL_SAMPLE_CHUNKS_JSON_OVERRIDE="${NO_TAU_DIR}/dev_retriever_timeline_top10_notau_sample_chunks.json" \
SUMMARY_JSON_OVERRIDE="${NO_TAU_DIR}/retriever_timeline_top10_notau_manifest.json" \
TOP_K_OVERRIDE=10 \
SCORE_THRESHOLD_OVERRIDE=-1000000000 \
LOOKBACK_SEC_OVERRIDE=1.92 \
GPU_DEVICES_CSV_OVERRIDE="${GPU_DEVICES_CSV}" \
NUM_SHARDS_OVERRIDE="${NUM_SHARDS}" \
PYTHON_BIN_OVERRIDE="${PYTHON_BIN}" \
CONDA_PREFIX_OVERRIDE="${CONDA_PREFIX}" \
bash "${BASE_TIMELINE_LAUNCHER}"

build_variant() {
  local split="$1"
  local variant="$2"
  local input="$3"
  local output="${OUT_DIR}/${split}_s_zh_${variant}_dense_top10_notau_backfill_lh1b88kw_srcmatch100k.jsonl"
  local stats="${OUT_DIR}/${split}_s_zh_${variant}_dense_top10_notau_backfill_stats.json"
  local sample="${OUT_DIR}/${split}_s_zh_${variant}_dense_top10_notau_backfill_samples.json"
  local py_variant="${variant}"
  if [[ "${variant}" == "v10" ]]; then
    py_variant="v10_marker"
  fi
  echo "[INFO] Building ${split}/${variant}: ${output}"
  "${PYTHON_BIN}" "${DENSE_SCRIPT}" \
    --input-jsonl "${input}" \
    --output-jsonl "${output}" \
    --stats-json "${stats}" \
    --sample-json "${sample}" \
    --variant "${py_variant}" \
    --lang-code zh \
    --max-terms 20 \
    --seed 20260521
}

build_variant train v9 "${NO_TAU_TRAIN_JSONL}"
build_variant dev v9 "${NO_TAU_DEV_JSONL}"
build_variant train v10 "${NO_TAU_TRAIN_JSONL}"
build_variant dev v10 "${NO_TAU_DEV_JSONL}"

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
        "exact_ref_gt_keep_rate": d["exact_ref_gt_keep_rate"],
        "gt_term_in_term_map_rate": d["gt_term_in_term_map_rate"],
        "nonempty_term_map_rate": d["nonempty_term_map_rate"],
        "no_gt_nonempty_term_map_rate": d["no_gt_nonempty_term_map_rate"],
        "avg_retrieved_entries_per_chunk": d["avg_retrieved_entries_per_chunk"],
        "avg_term_map_entries_per_chunk": d["avg_term_map_entries_per_chunk"],
        "avg_non_gt_entries_per_chunk": d["avg_non_gt_entries_per_chunk"],
        "backfilled_gt_terms": d["backfilled_gt_terms"],
        "marker_augmented_gt_entries": d["marker_augmented_gt_entries"],
        "assistant_marker_replacements": d["assistant_marker_replacements"],
        "term_map_size_hist": d["term_map_size_hist"],
    }
summary_path = out_dir / "v9_v10_dense_top10_notau_backfill_summary.json"
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"[INFO] Wrote summary: {summary_path}")
PY

echo "[DONE] V9/V10 dense top10 no-tau data ready: ${OUT_DIR}"
