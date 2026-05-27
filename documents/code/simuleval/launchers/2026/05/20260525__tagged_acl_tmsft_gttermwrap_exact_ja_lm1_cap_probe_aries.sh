#!/usr/bin/env bash
set -euo pipefail

# JA tagged-ACL lm=1 cap probe for the exact GT-term-wrapped TM-SFT SLM.
# One invocation runs one max_new_tokens setting on one GPU pair.

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

BATCH_LAUNCHER="${BATCH_LAUNCHER_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__batched_vllm_rag_eval.sh}"
RUN_STAMP="${RUN_STAMP_OVERRIDE:-$(date -u +%Y%m%dT%H%M%S)_ja_lm1_cap_probe}"
GPU_PAIR="${GPU_PAIR_OVERRIDE:-0,1}"
MAX_NEW_TOKENS_VALUE="${MAX_NEW_TOKENS_VALUE_OVERRIDE:-48}"
RAG_TOP_K_VALUE="${RAG_TOP_K_VALUE_OVERRIDE:-3}"
RAG_SCORE_THRESHOLD_VALUE="${RAG_SCORE_THRESHOLD_VALUE_OVERRIDE:-0.79}"
EMPTY_TERM_MAP_POLICY_VALUE="${EMPTY_TERM_MAP_POLICY_VALUE_OVERRIDE:-omit}"

MODEL_NAME="${MODEL_NAME_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_tmsft_gttermwrap_exact_ja_r32a32_ep4_taurus8/keep1.0_r32/v0-20260525-104902-hf}"
INPUT_DIR="${INPUT_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_ja_lm1to4_20260524T0933_tagacl_newv9_hn1024_tau078_raw_ja_taurus8/__inputs__/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/full_all/ja/all}"
RAW_GLOSSARY="${RAW_GLOSSARY_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
HN1024_CKPT="${HN1024_CKPT_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
NOTES_FILE="${NOTES_FILE_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260525__tagged_acl_tmsft_gttermwrap_exact_ja_lm1_cap_probe_aries.md}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/batched_vllm}"
PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python}"

MODEL_LABEL="${MODEL_LABEL_OVERRIDE:-tmsft_gttermwrap_exact_ja_lm1_cap${MAX_NEW_TOKENS_VALUE}_topk${RAG_TOP_K_VALUE}_tau${RAG_SCORE_THRESHOLD_VALUE/./}_omit}"
OUT_ROOT="${OUT_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_tmsft_gttermwrap_exact_ja_lm1_cap_probe_${RUN_STAMP}}"
OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-${OUT_ROOT}/${MODEL_LABEL}}"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_tmsft_gttermwrap_exact_ja_lm1_cap_probe_${RUN_STAMP}}"
EVAL_TMPDIR_ROOT="${EVAL_TMPDIR_ROOT_OVERRIDE:-/tmp/jx_jalp_${MAX_NEW_TOKENS_VALUE}}"
INPUT_WORK_DIR="${OUT_ROOT}/__inputs__"
SRC_LIST_PORTABLE="${INPUT_WORK_DIR}/source.portable.list"

fail() {
  echo "[ERROR] $*" >&2
  exit 3
}

require_file() {
  local path="$1"
  [[ -s "${path}" ]] || fail "Missing/empty required file: ${path}"
}

gpu_is_idle() {
  local gpu="$1" csv line mem util
  csv="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"
  line="$(awk -F, -v g="${gpu}" '$1 + 0 == g {print $0}' <<< "${csv}")"
  [[ -n "${line}" ]] || return 1
  mem="$(awk -F, '{gsub(/[[:space:]]/, "", $2); print $2}' <<< "${line}")"
  util="$(awk -F, '{gsub(/[[:space:]]/, "", $3); print $3}' <<< "${line}")"
  (( mem <= 2048 && util <= 25 ))
}

wait_pair_idle() {
  local g0 g1
  IFS=',' read -r g0 g1 <<< "${GPU_PAIR}"
  while true; do
    if gpu_is_idle "${g0}" && gpu_is_idle "${g1}"; then
      return 0
    fi
    echo "[WAIT] gpu_pair=${GPU_PAIR} not idle; retry in 30s" >&2
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,name --format=csv,noheader,nounits >&2 || true
    sleep 30
  done
}

summarize_results() {
  "${PYTHON_BIN}" - "${OUTPUT_BASE}" "${MODEL_LABEL}" "${MAX_NEW_TOKENS_VALUE}" "${RAG_TOP_K_VALUE}" "${RAG_SCORE_THRESHOLD_VALUE}" <<'PY'
import csv
import json
import sys
from pathlib import Path

output_base = Path(sys.argv[1])
model_label = sys.argv[2]
cap = sys.argv[3]
top_k = sys.argv[4]
tau = sys.argv[5]

paths = sorted(output_base.glob("ja/**_lm1_*/eval_results.tsv"))
if len(paths) != 1:
    raise SystemExit(f"expected one eval_results.tsv, found {len(paths)}: {paths}")
eval_path = paths[0]
with eval_path.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
if len(rows) != 1:
    raise SystemExit(f"expected one eval row in {eval_path}, got {len(rows)}")

out_dir = eval_path.parent
inst = out_dir / "instances.log"
strip = out_dir / "instances.strip_term.log"
if not inst.is_file() or not strip.is_file():
    raise SystemExit("missing instances.log or instances.strip_term.log")

length_rows = []
for line in strip.open("r", encoding="utf-8"):
    row = json.loads(line)
    hyp = row.get("prediction", "")
    ref = row.get("reference", "")
    length_rows.append(
        {
            "index": row.get("index"),
            "hyp_chars": len(hyp),
            "ref_chars": len(ref),
            "ratio": len(hyp) / max(1, len(ref)),
        }
    )
if len(length_rows) != 5:
    raise SystemExit(f"expected 5 strip rows, got {len(length_rows)}")

metric = rows[0]
summary_dir = output_base / "__summary__"
summary_dir.mkdir(parents=True, exist_ok=True)
summary_tsv = summary_dir / "summary_ja_lm1.tsv"
fields = [
    "method_key", "lang", "lm", "max_new_tokens", "rag_top_k", "tau",
    "BLEU", "StreamLAAL", "StreamLAAL_CA", "TERM_ACC", "TERM_CORRECT",
    "TERM_TOTAL", "max_hyp_ref_char_ratio", "eval_results",
    "instances_log", "instances_strip_term_log",
]
with summary_tsv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerow(
        {
            "method_key": model_label,
            "lang": metric.get("lang_code", "ja"),
            "lm": "1",
            "max_new_tokens": cap,
            "rag_top_k": top_k,
            "tau": tau,
            "BLEU": metric.get("BLEU", ""),
            "StreamLAAL": metric.get("StreamLAAL", ""),
            "StreamLAAL_CA": metric.get("StreamLAAL_CA", ""),
            "TERM_ACC": metric.get("TERM_ACC", ""),
            "TERM_CORRECT": metric.get("TERM_CORRECT", ""),
            "TERM_TOTAL": metric.get("TERM_TOTAL", ""),
            "max_hyp_ref_char_ratio": f"{max(x['ratio'] for x in length_rows):.6f}",
            "eval_results": str(eval_path),
            "instances_log": str(inst),
            "instances_strip_term_log": str(strip),
        }
    )
(summary_dir / "length_ratios_ja_lm1.json").write_text(
    json.dumps(length_rows, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(summary_tsv)
PY
}

for p in \
  "${BATCH_LAUNCHER}" \
  "${MODEL_NAME}/config.json" \
  "${MODEL_NAME}/generation_config.json" \
  "${MODEL_NAME}/model.safetensors.index.json" \
  "${RAW_GLOSSARY}" \
  "${HN1024_CKPT}" \
  "${NOTES_FILE}" \
  "${INPUT_DIR}/source.list" \
  "${INPUT_DIR}/target.list" \
  "${INPUT_DIR}/source_text.txt" \
  "${INPUT_DIR}/ref.txt" \
  "${INPUT_DIR}/audio.yaml"; do
  require_file "${p}"
done

mkdir -p "${OUT_ROOT}" "${OUTPUT_BASE}/__summary__" "${LOG_ROOT}" "${EVAL_TMPDIR_ROOT}" "${INPUT_WORK_DIR}"
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
  echo "model_label=${MODEL_LABEL}"
  echo "input_dir=${INPUT_DIR}"
  echo "source_list_portable=${SRC_LIST_PORTABLE}"
  echo "raw_glossary=${RAW_GLOSSARY}"
  echo "retriever=${HN1024_CKPT}"
  echo "output_base=${OUTPUT_BASE}"
  echo "lms=1"
  echo "max_new_tokens=${MAX_NEW_TOKENS_VALUE}"
  echo "rag_top_k=${RAG_TOP_K_VALUE}"
  echo "tau=${RAG_SCORE_THRESHOLD_VALUE}"
  echo "empty_term_map_policy=${EMPTY_TERM_MAP_POLICY_VALUE}"
  echo "lookback_sec=1.92"
  echo "vllm_limit_audio=128"
  echo "vllm_max_model_len=12288"
  echo "strip_output_tags=term"
} | tee "${OUT_ROOT}/run_meta.txt"

wait_pair_idle

RUN_TAG_OVERRIDE="${RUN_STAMP}_ja_lm1_cap${MAX_NEW_TOKENS_VALUE}" \
ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
LANG_CODE_OVERRIDE="ja" \
LMS_OVERRIDE="1" \
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${GPU_PAIR}" \
VLLM_TP_SIZE_OVERRIDE=2 \
MAX_NUM_SEQS_OVERRIDE=5 \
SCHEDULER_BATCH_SIZE_OVERRIDE=5 \
SCHEDULE_MODE_OVERRIDE=round_robin \
VLLM_ENFORCE_EAGER_OVERRIDE=1 \
VLLM_ENABLE_PREFIX_CACHING=1 \
VLLM_LIMIT_AUDIO_OVERRIDE=128 \
SAFETENSORS_LOAD_STRATEGY_OVERRIDE=eager \
MAX_MODEL_LEN_OVERRIDE=12288 \
VLLM_MAX_MODEL_LEN_OVERRIDE=12288 \
MAX_CACHE_SECONDS_OVERRIDE=80 \
KEEP_CACHE_SECONDS_OVERRIDE=60 \
MIN_CACHE_CHUNKS_OVERRIDE=1 \
MAX_NEW_TOKENS_OVERRIDE="${MAX_NEW_TOKENS_VALUE}" \
MAX_NEW_TOKENS_POLICY_OVERRIDE=fixed \
TEMPERATURE_OVERRIDE=0.6 \
TOP_P_OVERRIDE=0.95 \
TOP_K_DECODE_OVERRIDE=20 \
GPU_MEMORY_UTILIZATION_OVERRIDE=0.72 \
VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
SRC_LIST_OVERRIDE="${SRC_LIST_PORTABLE}" \
TGT_LIST_OVERRIDE="${INPUT_DIR}/target.list" \
SOURCE_TEXT_FILE_OVERRIDE="${INPUT_DIR}/source_text.txt" \
REF_FILE_OVERRIDE="${INPUT_DIR}/ref.txt" \
AUDIO_YAML_OVERRIDE="${INPUT_DIR}/audio.yaml" \
GLOSSARY_PATH_OVERRIDE="${RAW_GLOSSARY}" \
EVAL_GLOSSARY_PATH_OVERRIDE="${RAW_GLOSSARY}" \
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
DENSITY_TAG_OVERRIDE="tagacl_bv1_tmsft_gtwrap_lm1cap${MAX_NEW_TOKENS_VALUE}_topk${RAG_TOP_K_VALUE}_tau${RAG_SCORE_THRESHOLD_VALUE/./}_omit" \
GLOSSARY_TAG_OVERRIDE="acl6060_tagged_gt_raw_min_norm2" \
RAG_MODEL_PATH_OVERRIDE="${HN1024_CKPT}" \
RAG_SCORE_THRESHOLD_OVERRIDE="${RAG_SCORE_THRESHOLD_VALUE}" \
RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE=1.92 \
RAG_TOP_K_OVERRIDE="${RAG_TOP_K_VALUE}" \
RAG_DEVICE_OVERRIDE="cuda:0" \
RAG_BATCH_RETRIEVAL_OVERRIDE=1 \
INDEX_BUILD_DEVICE_OVERRIDE="cuda:0" \
INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}" \
TERM_MAP_FORMAT_OVERRIDE=plain \
EMPTY_TERM_MAP_POLICY_OVERRIDE="${EMPTY_TERM_MAP_POLICY_VALUE}" \
TERM_FCR_POLICY_OVERRIDE=term_map_source_ref_negative_sentence \
STRIP_OUTPUT_TAGS_OVERRIDE=term \
WANDB_LOG_OVERRIDE=0 \
NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
LOG_ROOT_OVERRIDE="${LOG_ROOT}/batch" \
EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR_ROOT}/cap${MAX_NEW_TOKENS_VALUE}" \
bash "${BATCH_LAUNCHER}" \
  > "${LOG_ROOT}/launcher.out" \
  2> "${LOG_ROOT}/launcher.err"

summarize_results | tee "${OUT_ROOT}/summary_path.txt"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tee "${LOG_ROOT}/gpu_postrun.csv"
echo "[ALL DONE] output_base=${OUTPUT_BASE}"
