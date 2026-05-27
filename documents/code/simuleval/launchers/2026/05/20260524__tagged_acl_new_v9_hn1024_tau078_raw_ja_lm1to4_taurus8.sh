#!/usr/bin/env bash
set -euo pipefail

# Tagged ACL Japanese main result on taurus.
# Runs lm=1,2,3,4 in parallel, one 2-GPU pair per setting.

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260520__tagged_acl_origin_bsz4_tau073_sweep_taurus45269.sh}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)_tagacl_newv9_hn1024_tau078_raw_ja_taurus8}"

MODEL_LABEL="${MODEL_LABEL:-new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078}"
MODEL_NAME="${MODEL_NAME:-/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_ja_r32a64_tp2_taurus4/keep1.0_r32/v0-20260524-123624-hf}"
HN1024_CKPT="${HN1024_CKPT:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
RAW_GLOSSARY="${RAW_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
GS10K_GLOSSARY="${GS10K_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_union_gs10000_min_norm2_backfill.json}"

OUT_ROOT="${OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_ja_lm1to4_${RUN_STAMP}}"
OUTPUT_BASE="${OUTPUT_BASE:-${OUT_ROOT}/${MODEL_LABEL}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_new_v9_hn1024_tau078_raw_ja_lm1to4_${RUN_STAMP}}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260524__tagged_acl_new_v9_hn1024_tau078_raw_ja_lm1to4_taurus8.md}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache}"
# Keep this short: vLLM uses Unix-domain IPC sockets under TMPDIR and the
# sockaddr path limit is 107 bytes.
EVAL_TMPDIR="${EVAL_TMPDIR:-/tmp/jxja_t8}"

fail() {
  echo "[ERROR] $*" >&2
  exit 3
}

for p in "${ROOT_DIR}" "${BASE_LAUNCHER}" "${MODEL_NAME}/config.json" "${HN1024_CKPT}" "${RAW_GLOSSARY}" "${GS10K_GLOSSARY}"; do
  [[ -e "${p}" ]] || fail "Missing required path: ${p}"
done

shard_count="$(find "${MODEL_NAME}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
[[ "${shard_count}" == "15" ]] || fail "Expected 15 HF shards under ${MODEL_NAME}, got ${shard_count}"

mkdir -p "${OUT_ROOT}" "${OUTPUT_BASE}/__summary__" "${LOG_ROOT}" "${EVAL_TMPDIR}" "$(dirname "${NOTES_FILE}")"
touch "${NOTES_FILE}"

{
  echo "run_stamp=${RUN_STAMP}"
  echo "host=$(hostname -s)"
  echo "model=${MODEL_NAME}"
  echo "retriever=${HN1024_CKPT}"
  echo "raw_glossary=${RAW_GLOSSARY}"
  echo "output_base=${OUTPUT_BASE}"
  echo "log_root=${LOG_ROOT}"
  echo "lang=ja"
  echo "lms=1 2 3 4"
  echo "glossary=raw"
  echo "tau=0.78"
  echo "lookback_sec=1.92"
  echo "strip_output_tags=term"
} | tee "${OUTPUT_BASE}/run_meta.txt"

MODE="full" \
RUN_GRANULARITY="full_corpus" \
HOLD_JOB_ID=0 \
INSIDE_HOLD_STEP=1 \
MAX_PARALLEL_OVERRIDE=4 \
LANGS_OVERRIDE="ja" \
LMS_OVERRIDE="1 2 3 4" \
GLOSSARY_KINDS_OVERRIDE="raw" \
GPU_PAIRS_CSV_OVERRIDE="0,1;2,3;4,5;6,7" \
MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
TERM_MAP_FORMAT_OVERRIDE="plain" \
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
INPUT_ROOT_OVERRIDE="${OUT_ROOT}/__inputs__/${MODEL_LABEL}" \
LOG_DIR_OVERRIDE="${LOG_ROOT}/${MODEL_LABEL}" \
SUMMARY_DIR_OVERRIDE="${OUTPUT_BASE}/__summary__" \
INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}" \
NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
RAW_GLOSSARY_OVERRIDE="${RAW_GLOSSARY}" \
GS10K_GLOSSARY_OVERRIDE="${GS10K_GLOSSARY}" \
EVAL_GLOSSARY_PATH_GLOBAL_OVERRIDE="${RAW_GLOSSARY}" \
EVAL_GLOSSARY_FOLLOWS_KIND_OVERRIDE=0 \
EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR}" \
CLEAN_OUTPUT_DIR_OVERRIDE="${CLEAN_OUTPUT_DIR_OVERRIDE:-1}" \
DENSITY_TAG_OVERRIDE="tagacl_new_v9_hn1024_tau078" \
WANDB_RUN_PREFIX_OVERRIDE="new_v9_hn1024_tau078" \
WANDB_EXPERIMENT_FAMILY_OVERRIDE="tagged_acl_new_v9_hn1024_tau078" \
WANDB_VARIANT_PREFIX_OVERRIDE="new_v9_hn1024_tau078" \
WANDB_COMPUTE_TAG_OVERRIDE="compute:taurus8_direct" \
TERM_FCR_POLICY_OVERRIDE="term_map_source_ref_negative_sentence" \
RAG_MODEL_PATH_OVERRIDE="${HN1024_CKPT}" \
RAG_SCORE_THRESHOLD_OVERRIDE="0.78" \
RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE="1.92" \
RAG_TOP_K_OVERRIDE=10 \
RAG_GPU_OVERRIDE="cuda:1" \
STRIP_OUTPUT_TAGS_OVERRIDE="term" \
VLLM_TP_SIZE_OVERRIDE=2 \
GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}" \
VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}" \
VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}" \
VLLM_USE_FUSED_MOE_GROUPED_TOPK="${VLLM_USE_FUSED_MOE_GROUPED_TOPK:-0}" \
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/mnt/taurus/home/jiaxuanluo/.config/wandb}" \
WANDB_DIR="${WANDB_DIR:-/mnt/gemini/data1/jiaxuanluo/wandb_runs}" \
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/mnt/gemini/data1/jiaxuanluo/wandb_cache}" \
WANDB_DATA_DIR="${WANDB_DATA_DIR:-/mnt/gemini/data1/jiaxuanluo/wandb_data}" \
WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-/mnt/gemini/data1/jiaxuanluo/wandb_artifacts}" \
bash "${BASE_LAUNCHER}"

python - "${OUTPUT_BASE}" <<'PY'
import csv
import re
import sys
from pathlib import Path

output_base = Path(sys.argv[1])
summary_dir = output_base / "__summary__"
summary_dir.mkdir(parents=True, exist_ok=True)
rows = []
for path in sorted(output_base.glob("ja/**/eval_results.tsv")):
    m = re.search(r"_lm(\d+)_", str(path))
    if not m:
        continue
    with path.open("r", encoding="utf-8", newline="") as f:
        data = list(csv.DictReader(f, delimiter="\t"))
    if len(data) != 1:
        raise SystemExit(f"expected one row in {path}, got {len(data)}")
    row = data[0]
    rows.append({
        "lang": "ja",
        "lm": int(m.group(1)),
        "glossary": "raw",
        "BLEU": row.get("BLEU", ""),
        "TERM_ACC": row.get("TERM_ACC", ""),
        "REAL_TERM_ADOPT": row.get("REAL_TERM_ADOPT", ""),
        "TERM_FCR": row.get("TERM_FCR", ""),
        "StreamLAAL": row.get("StreamLAAL", ""),
        "eval_results": str(path),
    })
rows.sort(key=lambda r: r["lm"])
if len(rows) != 4:
    raise SystemExit(f"expected 4 ja raw lm eval rows, found {len(rows)} under {output_base}")

fields = ["lang", "lm", "glossary", "BLEU", "TERM_ACC", "REAL_TERM_ADOPT", "TERM_FCR", "StreamLAAL", "eval_results"]
tsv = summary_dir / "summary_ja_raw_lm1to4.tsv"
with tsv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    w.writeheader()
    w.writerows(rows)

def pct(x):
    try:
        return f"{float(x) * 100:.2f}"
    except Exception:
        return str(x)

lines = [
    "# Tagged ACL ja raw: New V9 + HN1024 tau=0.78",
    "",
    "| lang | lm | glossary | BLEU | TERM_ACC | REAL_ADOPT | TERM_FCR | StreamLAAL |",
    "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
]
for r in rows:
    lines.append(
        f"| {r['lang']} | {r['lm']} | {r['glossary']} | "
        f"{float(r['BLEU']):.2f} | {pct(r['TERM_ACC'])} | {pct(r['REAL_TERM_ADOPT'])} | "
        f"{pct(r['TERM_FCR'])} | {float(r['StreamLAAL']):.0f} |"
    )
lines += ["", f"Summary TSV: `{tsv}`", f"Output base: `{output_base}`"]
md = summary_dir / "summary_ja_raw_lm1to4.md"
md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(md)
print(tsv)
PY

echo "[ALL DONE] ${OUTPUT_BASE}/__summary__/summary_ja_raw_lm1to4.md"
