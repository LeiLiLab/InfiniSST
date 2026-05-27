#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

RUN_STAMP="${RUN_STAMP:-20260526T000000_origin_norag_de_lm4_batch_serialaligned_aries}"
OUT_ROOT="${OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_origin_norag_de_lm4_batch_serialaligned_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_origin_norag_de_lm4_batch_serialaligned_${RUN_STAMP}}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/gemini/data1/jiaxuanluo/cache/tagged_acl_origin_norag_de_lm4_batch_serialaligned_${RUN_STAMP}}"
INPUT_WORK_DIR="${OUT_ROOT}/__inputs__"
SRC_LIST_PORTABLE="${INPUT_WORK_DIR}/source.portable.list"

BATCH_LAUNCHER="${BATCH_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__batched_vllm_rag_eval.sh}"
MODEL_NAME="${MODEL_NAME:-/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_origin-bsz4}"
INPUT_DIR="${INPUT_DIR:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_de_lm1to4_20260524T075533_tagacl_newv9_hn1024_tau078_raw_de_retry1/__inputs__/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/full_all/de/all}"
RAW_GLOSSARY="${RAW_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260526__tagged_acl_origin_norag_de_lm4_batch_serialaligned_aries.md}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python}"

GPU_PAIR="${GPU_PAIR:-2,3}"
EVAL_TMPDIR="${EVAL_TMPDIR:-/tmp/jxnoilm4ba}"
DENSITY_TAG="${DENSITY_TAG:-tagacl_origin_norag_batch_serialaligned_lm4}"
GLOSSARY_TAG="${GLOSSARY_TAG:-acl6060_tagged_gt_raw_min_norm2}"
OUTPUT_BASE="${OUT_ROOT}/origin_norag_de_lm4_batch_serialaligned"

fail() {
  echo "[ERROR] $*" >&2
  exit 3
}

require_file() {
  local path="$1"
  [[ -s "${path}" ]] || fail "Missing/empty required file: ${path}"
}

main() {
  [[ "$(hostname -s)" == aries* ]] || fail "This launcher is Aries-only; current host=$(hostname -s)"
  mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" "${CACHE_ROOT}" "${INPUT_WORK_DIR}" "${EVAL_TMPDIR}"

  for f in \
    "${BATCH_LAUNCHER}" \
    "${MODEL_NAME}/config.json" \
    "${RAW_GLOSSARY}" \
    "${NOTES_FILE}" \
    "${INPUT_DIR}/source.list" \
    "${INPUT_DIR}/target.list" \
    "${INPUT_DIR}/source_text.txt" \
    "${INPUT_DIR}/ref.txt" \
    "${INPUT_DIR}/audio.yaml"; do
    require_file "${f}"
  done

  sed \
    -e 's#^/mnt/data/siqiouyang/#/mnt/taurus/data/siqiouyang/#' \
    -e 's#^/mnt/data1/siqiouyang/#/mnt/taurus/data1/siqiouyang/#' \
    "${INPUT_DIR}/source.list" > "${SRC_LIST_PORTABLE}"

  while IFS= read -r wav_path; do
    [[ -n "${wav_path}" ]] || continue
    require_file "${wav_path}"
  done < "${SRC_LIST_PORTABLE}"

  {
    echo "run_stamp=${RUN_STAMP}"
    echo "host=$(hostname -s)"
    echo "gpu_pair=${GPU_PAIR}"
    echo "model=${MODEL_NAME}"
    echo "input_dir=${INPUT_DIR}"
    echo "source_list_portable=${SRC_LIST_PORTABLE}"
    echo "glossary=${RAW_GLOSSARY}"
    echo "output_base=${OUTPUT_BASE}"
    echo "lang=de"
    echo "lm=4"
    echo "method=InfiniSST/no-RAG"
    echo "mode=batch_vllm"
    echo "alignment_target=serial_20260524T160830_lm4"
    echo "max_new_tokens=40_fixed"
    echo "max_cache_seconds=80"
    echo "keep_cache_seconds=60"
    echo "expected_cache_chunks=20/15"
    echo "vllm_limit_audio=20"
    echo "norag_prompt_policy=serial_compat"
    echo "empty_term_map_policy=omit"
  } | tee "${OUT_ROOT}/run_meta.txt"

  export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
  export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
  export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
  export CUDA_CACHE_PATH="${CACHE_ROOT}/cuda"
  export HF_HOME="${CACHE_ROOT}/hf"
  export HF_HUB_CACHE="${CACHE_ROOT}/hf/hub"
  export TRANSFORMERS_CACHE="${CACHE_ROOT}/hf/transformers"
  export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
  export NUMBA_CACHE_DIR="${CACHE_ROOT}/numba"
  export WANDB_CACHE_DIR="${CACHE_ROOT}/wandb_cache"
  mkdir -p "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
    "${CUDA_CACHE_PATH}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${VLLM_CACHE_ROOT}" \
    "${NUMBA_CACHE_DIR}" "${WANDB_CACHE_DIR}"

  ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
  RUN_TAG_OVERRIDE="${RUN_STAMP}" \
  LANG_CODE_OVERRIDE="de" \
  LMS_OVERRIDE="4" \
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${GPU_PAIR}" \
  VLLM_TP_SIZE_OVERRIDE=2 \
  MAX_NUM_SEQS_OVERRIDE=5 \
  SCHEDULER_BATCH_SIZE_OVERRIDE=5 \
  SCHEDULE_MODE_OVERRIDE=round_robin \
  DISABLE_RAG_OVERRIDE=1 \
  RAG_TOP_K_OVERRIDE=0 \
  RAG_SCORE_THRESHOLD_OVERRIDE=0 \
  RAG_BATCH_RETRIEVAL_OVERRIDE=0 \
  MAX_NEW_TOKENS_OVERRIDE=40 \
  MAX_NEW_TOKENS_POLICY_OVERRIDE=fixed \
  TEMPERATURE_OVERRIDE=0.6 \
  TOP_P_OVERRIDE=0.95 \
  TOP_K_DECODE_OVERRIDE=20 \
  MAX_CACHE_SECONDS_OVERRIDE=80 \
  KEEP_CACHE_SECONDS_OVERRIDE=60 \
  MIN_CACHE_CHUNKS_OVERRIDE=1 \
  VLLM_LIMIT_AUDIO_OVERRIDE=20 \
  VLLM_ENFORCE_EAGER_OVERRIDE=1 \
  VLLM_ENABLE_PREFIX_CACHING=1 \
  VLLM_MAX_MODEL_LEN_OVERRIDE=16384 \
  MAX_MODEL_LEN_OVERRIDE=16384 \
  VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
  VLLM_MOE_USE_DEEP_GEMM=0 \
  VLLM_USE_FUSED_MOE_GROUPED_TOPK=0 \
  GPU_MEMORY_UTILIZATION_OVERRIDE=0.78 \
  MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
  SRC_LIST_OVERRIDE="${SRC_LIST_PORTABLE}" \
  TGT_LIST_OVERRIDE="${INPUT_DIR}/target.list" \
  SOURCE_TEXT_FILE_OVERRIDE="${INPUT_DIR}/source_text.txt" \
  REF_FILE_OVERRIDE="${INPUT_DIR}/ref.txt" \
  AUDIO_YAML_OVERRIDE="${INPUT_DIR}/audio.yaml" \
  GLOSSARY_PATH_OVERRIDE="${RAW_GLOSSARY}" \
  EVAL_GLOSSARY_PATH_OVERRIDE="${RAW_GLOSSARY}" \
  OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
  DENSITY_TAG_OVERRIDE="${DENSITY_TAG}" \
  GLOSSARY_TAG_OVERRIDE="${GLOSSARY_TAG}" \
  EMPTY_TERM_MAP_POLICY_OVERRIDE=omit \
  NORAG_PROMPT_POLICY_OVERRIDE=serial_compat \
  TERM_FCR_POLICY_OVERRIDE=source_ref_negative_sentence \
  STRIP_OUTPUT_TAGS_OVERRIDE=term \
  WANDB_LOG_OVERRIDE=0 \
  NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
  LOG_ROOT_OVERRIDE="${LOG_ROOT}/batch" \
  EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR}" \
  bash "${BATCH_LAUNCHER}" 2>&1 | tee "${LOG_ROOT}/batch_launcher.stdout"

  "${PYTHON_BIN}" - "${OUTPUT_BASE}" <<'PY'
import csv
from pathlib import Path
import sys

output_base = Path(sys.argv[1])
paths = sorted(output_base.glob("de/*_lm4_*/eval_results.tsv"))
if len(paths) != 1:
    raise SystemExit(f"expected one lm4 eval_results.tsv, found {len(paths)}: {paths}")
rows = list(csv.DictReader(paths[0].open("r", encoding="utf-8"), delimiter="\t"))
if len(rows) != 1:
    raise SystemExit(f"expected one row in {paths[0]}, got {len(rows)}")
inst = paths[0].parent / "instances.log"
strip = paths[0].parent / "instances.strip_term.log"
inst_rows = sum(1 for _ in inst.open("r", encoding="utf-8"))
strip_rows = sum(1 for _ in strip.open("r", encoding="utf-8"))
if inst_rows != 5 or strip_rows != 5:
    raise SystemExit(f"expected 5 rows, got instances={inst_rows} strip={strip_rows}")
summary = output_base / "__summary__" / "summary_de_lm4_serialaligned.tsv"
summary.parent.mkdir(parents=True, exist_ok=True)
fields = [
    "method", "lang", "lm", "mode", "max_new_tokens", "cache_seconds",
    "vllm_limit_audio", "BLEU", "StreamLAAL", "StreamLAAL_CA",
    "TERM_ACC", "TERM_CORRECT", "TERM_TOTAL", "source_path",
]
row = rows[0]
record = {
    "method": "InfiniSST/no-RAG",
    "lang": "de",
    "lm": "4",
    "mode": "batch_vllm_serial_aligned",
    "max_new_tokens": "40",
    "cache_seconds": "80/60",
    "vllm_limit_audio": "20",
    "BLEU": row.get("BLEU", ""),
    "StreamLAAL": row.get("StreamLAAL", ""),
    "StreamLAAL_CA": row.get("StreamLAAL_CA", ""),
    "TERM_ACC": row.get("TERM_ACC", ""),
    "TERM_CORRECT": row.get("TERM_CORRECT", ""),
    "TERM_TOTAL": row.get("TERM_TOTAL", ""),
    "source_path": str(paths[0]),
}
with summary.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    w.writeheader()
    w.writerow(record)
print(
    f"RESULT lm4 BLEU={float(record['BLEU']):.4f} "
    f"StreamLAAL={float(record['StreamLAAL']):.4f} "
    f"StreamLAAL_CA={float(record['StreamLAAL_CA']):.4f} "
    f"TERM_ACC={float(record['TERM_ACC']):.4f} "
    f"TERM={record['TERM_CORRECT']}/{record['TERM_TOTAL']} "
    f"eval={paths[0]}"
)
print(summary)
PY
  date -u +%Y-%m-%dT%H:%M:%SZ > "${LOG_ROOT}/done.txt"
}

main "$@"
