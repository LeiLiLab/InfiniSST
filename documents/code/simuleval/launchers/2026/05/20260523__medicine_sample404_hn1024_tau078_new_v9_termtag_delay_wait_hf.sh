#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
EVAL_SCRIPT="${ROOT_DIR}/documents/code/simuleval/eval_density_unified.sh"
PREP_SCRIPT="${ROOT_DIR}/documents/code/simuleval/prepare_medicine_one_talk_inputs.py"
WANDB_LOGGER="${ROOT_DIR}/documents/code/offline_evaluation/wandb_eval_logger.py"
WANDB_PYTHON="${WANDB_PYTHON:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python}"
if [[ ! -x "${WANDB_PYTHON}" ]]; then
  WANDB_PYTHON="python3"
fi

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)}"
LANG_CODE="${LANG_CODE_OVERRIDE:-zh}"
TARGET_SAMPLE="${TARGET_SAMPLE_OVERRIDE:-404}"
TARGET_LM="${TARGET_LM_OVERRIDE:-2}"
GPU_PAIR="${GPU_PAIR:-4,5}"
RAG_GPU_OVERRIDE="${RAG_GPU_OVERRIDE:-cuda:1}"
RAG_TOP_K="${RAG_TOP_K_OVERRIDE:-10}"
RAG_SCORE_THRESHOLD="${RAG_SCORE_THRESHOLD_OVERRIDE:-0.78}"
RAG_LORA_R="${RAG_LORA_R_OVERRIDE:-128}"
RAG_TEXT_LORA_R="${RAG_TEXT_LORA_R_OVERRIDE:-128}"
RAG_TIMELINE_LOOKBACK_SEC="${RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE:-1.92}"
MODEL_LABEL="${MODEL_LABEL:-new_v9_termtag_delay_oldnewv3_r32a64}"
WAIT_FOR_HF_SECS="${WAIT_FOR_HF_SECS:-28800}"

MODEL_ROOT="${MODEL_ROOT_OVERRIDE:-/mnt/aries/data7/jiaxuanluo/slm/speech_llm_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_zh_r32a64_tp2_taurus8/keep1.0_r32}"
MODEL_NAME_OVERRIDE="${MODEL_NAME_OVERRIDE:-}"
HN1024_CKPT="${HN1024_CKPT_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
RUNTIME_GLOSSARY_GS10K="${RUNTIME_GLOSSARY_GS10K_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_glossary_gt_plus_medicine_wiki_gs10000_translated.json}"
STRICT_JSONL="${STRICT_MEDICINE_JSONL_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_dev_dataset.jsonl}"
STRICT_GLOSSARY="${STRICT_MEDICINE_GLOSSARY_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_glossary_gt_plus_medicine_wiki_gs10000_translated.json}"
ESO_TEST_ROOT="${ESO_TEST_ROOT_OVERRIDE:-/home/jiaxingxu/rag-sst/eso-dataset/outputs_v2/test}"
NOTES_FILE="${NOTES_FILE_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260523__medicine_sample404_hn1024_tau078_new_v9_termtag_delay.md}"

OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_sample404_hn1024_tau078_new_v9_termtag_delay_${RUN_STAMP}}"
MEDICINE_INPUTS="${OUTPUT_BASE}/${LANG_CODE}/__medicine_inputs__/lists"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs/medicine_sample404_hn1024_tau078_new_v9_${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/medicine_gs10k_pr_sweep}"
EVAL_TMPDIR="${EVAL_TMPDIR_OVERRIDE:-/tmp/jx_med_v9_tau078_${RUN_STAMP}}"
SUMMARY_TSV="${OUTPUT_BASE}/summary_medicine_sample404_hn1024_tau078_new_v9_metrics.tsv"
SUMMARY_MD="${OUTPUT_BASE}/summary_medicine_sample404_hn1024_tau078_new_v9_metrics.md"

latest_hf_dir() {
  local root="$1"
  local waited=0
  local found=""
  while true; do
    found="$(find "${root}" -maxdepth 1 -type d -name '*-hf' 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -n "${found}" && -f "${found}/config.json" ]]; then
      local shard_count
      shard_count="$(find "${found}" -maxdepth 1 -name 'model-*.safetensors' 2>/dev/null | wc -l | tr -d ' ')"
      if [[ "${shard_count}" == "15" ]]; then
        printf '%s\n' "${found}"
        return 0
      fi
      echo "[WAIT] HF checkpoint incomplete: ${found} has ${shard_count}/15 safetensor shards" >&2
    else
      echo "[WAIT] No complete HF checkpoint found under ${root}" >&2
    fi
    if (( WAIT_FOR_HF_SECS <= 0 || waited >= WAIT_FOR_HF_SECS )); then
      break
    fi
    sleep 60
    waited=$((waited + 60))
  done
  echo "[ERROR] No complete HF checkpoint found under ${root} after ${waited}s" >&2
  return 2
}

if [[ -n "${MODEL_NAME_OVERRIDE}" ]]; then
  MODEL_NAME="${MODEL_NAME_OVERRIDE}"
else
  MODEL_NAME="$(latest_hf_dir "${MODEL_ROOT}")"
fi

for p in "${EVAL_SCRIPT}" "${PREP_SCRIPT}" "${WANDB_LOGGER}" "${MODEL_NAME}" \
         "${RUNTIME_GLOSSARY_GS10K}" "${STRICT_JSONL}" "${STRICT_GLOSSARY}" \
         "${HN1024_CKPT}" "${NOTES_FILE}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

mkdir -p "${MEDICINE_INPUTS}" "${LOG_ROOT}" "${INDEX_CACHE_DIR}" "${EVAL_TMPDIR}"

echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] Preparing medicine inputs: sample=${TARGET_SAMPLE} lang=${LANG_CODE}"
python3 "${PREP_SCRIPT}" \
  --sample-id "${TARGET_SAMPLE}" \
  --lang-code "${LANG_CODE}" \
  --eso-test-root "${ESO_TEST_ROOT}" \
  --strict-jsonl "${STRICT_JSONL}" \
  --strict-glossary "${STRICT_GLOSSARY}" \
  --output-dir "${MEDICINE_INPUTS}" \
  --max-sentences "${MAX_SENTENCES_OVERRIDE:-0}"

PREFIX="medicine_${TARGET_SAMPLE}"
EVAL_GLOSSARY_PATH="${MEDICINE_INPUTS}/medicine_gt_strict_translated__${PREFIX}.json"
SRC_LIST="${MEDICINE_INPUTS}/medicine.source__${PREFIX}.txt"
TGT_LIST="${MEDICINE_INPUTS}/medicine.target.${LANG_CODE}__${PREFIX}.txt"
SOURCE_TEXT_FILE="${MEDICINE_INPUTS}/medicine.source_text.en__${PREFIX}.txt"
REF_FILE="${MEDICINE_INPUTS}/medicine.ref.${LANG_CODE}__${PREFIX}.txt"
AUDIO_YAML="${MEDICINE_INPUTS}/medicine.audio__${PREFIX}.yaml"
RUNTIME_GLOSSARY_TAG="$(basename "${RUNTIME_GLOSSARY_GS10K}" .json)"
TAU_TAG="${RAG_SCORE_THRESHOLD/./p}"
DENSITY="med1_${MODEL_LABEL}_hn1024_tau${TAU_TAG}_gs10k"
OUT_DIR="${OUTPUT_BASE}/${LANG_CODE}/d${DENSITY}_lm${TARGET_LM}_k${RAG_TOP_K}_th${RAG_SCORE_THRESHOLD}_g${RUNTIME_GLOSSARY_TAG}_pp${PREFIX}"
EVAL_TSV="${OUT_DIR}/eval_results.tsv"

for p in "${EVAL_GLOSSARY_PATH}" "${SRC_LIST}" "${TGT_LIST}" "${SOURCE_TEXT_FILE}" "${REF_FILE}" "${AUDIO_YAML}"; do
  if [[ ! -s "${p}" ]]; then
    echo "[ERROR] Prepared input missing or empty: ${p}" >&2
    exit 3
  fi
done

clean_shm() {
  local me
  me="$(id -un)"
  find /dev/shm -maxdepth 1 -user "${me}" -mmin +1 \
    \( -name 'psm_*' -o -name 'loky-*' -o -name 'torch_*' -o -name 'vllm_*' \) \
    -delete 2>/dev/null || true
}

if [[ -f "${EVAL_TSV}" && -s "${EVAL_TSV}" && "${FORCE_RERUN_OVERRIDE:-0}" != "1" ]]; then
  echo "[SKIP] existing eval: ${EVAL_TSV}"
else
  clean_shm
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${GPU_PAIR}" \
  MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
  RAG_MODEL_PATH_OVERRIDE="${HN1024_CKPT}" \
  RAG_LORA_R_OVERRIDE="${RAG_LORA_R}" \
  RAG_TEXT_LORA_R_OVERRIDE="${RAG_TEXT_LORA_R}" \
  OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
  EVAL_MODE_OVERRIDE="acl6060" \
  LANG_CODE_OVERRIDE="${LANG_CODE}" \
  GLOSSARY_PATH_OVERRIDE="${RUNTIME_GLOSSARY_GS10K}" \
  EVAL_GLOSSARY_PATH_OVERRIDE="${EVAL_GLOSSARY_PATH}" \
  SRC_LIST_OVERRIDE="${SRC_LIST}" \
  TGT_LIST_OVERRIDE="${TGT_LIST}" \
  REF_FILE_OVERRIDE="${REF_FILE}" \
  SOURCE_TEXT_FILE_OVERRIDE="${SOURCE_TEXT_FILE}" \
  AUDIO_YAML_OVERRIDE="${AUDIO_YAML}" \
  LATENCY_MULTIPLIER_OVERRIDE="${TARGET_LM}" \
  RAG_TOP_K_OVERRIDE="${RAG_TOP_K}" \
  RAG_SCORE_THRESHOLD_OVERRIDE="${RAG_SCORE_THRESHOLD}" \
  RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE="${RAG_TIMELINE_LOOKBACK_SEC}" \
  RAG_STREAMING_MODE_OVERRIDE="timeline" \
  TERM_MAP_FORMAT_OVERRIDE="plain" \
  STRIP_OUTPUT_TAGS_OVERRIDE="term" \
  TERM_FCR_POLICY="term_map_source_ref_negative_sentence" \
  DENSITY_TAG="${DENSITY}" \
  PAPER_ID_TAG="${PREFIX}" \
  INDEX_CACHE_DIR="${INDEX_CACHE_DIR}" \
  EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR}/hn1024_tau${TAU_TAG}" \
  RAG_GPU_OVERRIDE="${RAG_GPU_OVERRIDE}" \
  GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}" \
  CLEAN_OUTPUT_DIR_OVERRIDE="${CLEAN_OUTPUT_DIR_OVERRIDE:-0}" \
  bash "${EVAL_SCRIPT}" > "${LOG_ROOT}/hn1024_tau${TAU_TAG}.out" 2> "${LOG_ROOT}/hn1024_tau${TAU_TAG}.err"
  clean_shm
fi

if [[ ! -s "${EVAL_TSV}" ]]; then
  echo "[ERROR] Missing eval TSV after run: ${EVAL_TSV}" >&2
  exit 5
fi

HOME="${WANDB_HOME:-/mnt/taurus/home/jiaxuanluo}" \
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_HOME:-/mnt/taurus/home/jiaxuanluo}/.config/wandb}" \
"${WANDB_PYTHON}" "${WANDB_LOGGER}" \
  --project simuleval_eval \
  --run-name "${MODEL_LABEL}__medicine_${TARGET_SAMPLE}__gs10k__hn1024_tau${TAU_TAG}__lm${TARGET_LM}" \
  --experiment-family medicine_sample404_speech_llm_quick \
  --data-tag "medicine_sample${TARGET_SAMPLE}_gs10k_fixedraw_${LANG_CODE}" \
  --task-tag eval \
  --notes-file "${NOTES_FILE}" \
  --trained-from-run "2g6kan5y" \
  --extra-tags "variant:new_v9_hn1024_tau${TAU_TAG}" "retriever:lh1b88kw" "tau:${TAU_TAG}" "glossary:medicine_gs10k" "lang:${LANG_CODE}" "sample:${PREFIX}" "strip:term" "compute:taurus_direct" \
  --density "${DENSITY}" \
  --rag-top-k "${RAG_TOP_K}" \
  --rag-score-threshold "${RAG_SCORE_THRESHOLD}" \
  --output-base "${OUTPUT_BASE}" \
  --lang-code "${LANG_CODE}" \
  --latency-multipliers "${TARGET_LM}" \
  --glossary-tag "${RUNTIME_GLOSSARY_TAG}" \
  --paper-id "${PREFIX}" \
  --model-name "${MODEL_NAME}" \
  --rag-model-path "${HN1024_CKPT}" \
  --verdict "Medicine sample404 gs10k New V9 term-tag eval, HN1024 tau=${RAG_SCORE_THRESHOLD}; hypothesis <term> tags stripped before scoring."

"${WANDB_PYTHON}" - <<'PY' "${EVAL_TSV}" "${SUMMARY_TSV}" "${SUMMARY_MD}" "${MODEL_LABEL}" "${RAG_SCORE_THRESHOLD}" "${MODEL_NAME}"
import csv, sys
from pathlib import Path
eval_tsv, summary_tsv, summary_md, model_label, tau, model_name = sys.argv[1:]
with open(eval_tsv, newline='') as f:
    rec = next(csv.DictReader(f, delimiter='\t'))
row = {
    'model_label': model_label,
    'retriever': 'hn1024/lh1b88kw',
    'tau': tau,
    'BLEU': float(rec.get('BLEU', 'nan')),
    'TERM_ACC_pct': float(rec.get('TERM_ACC', 'nan')) * 100,
    'REAL_ADOPT_pct': float(rec.get('REAL_TERM_ADOPT', 'nan')) * 100,
    'TERM_FCR_pct': float(rec.get('TERM_FCR', 'nan')) * 100,
    'SOURCE_TERM_SENT_FCR_pct': float(rec.get('SOURCE_TERM_SENT_FCR', 'nan')) * 100,
    'StreamLAAL': float(rec.get('StreamLAAL', 'nan')),
    'model_name': model_name,
    'eval_tsv': eval_tsv,
}
Path(summary_tsv).parent.mkdir(parents=True, exist_ok=True)
with open(summary_tsv, 'w', newline='') as f:
    w = csv.DictWriter(f, delimiter='\t', fieldnames=list(row.keys()))
    w.writeheader(); w.writerow(row)
with open(summary_md, 'w') as f:
    f.write(f"# Medicine sample404 HN1024 tau{tau} New V9 metrics\n\n")
    f.write("Output-side `<term>` tags are stripped before scoring.\n\n")
    f.write("| model | retriever | tau | BLEU | TERM_ACC | REAL_ADOPT | TERM_FCR | SOURCE_SENT_FCR | StreamLAAL |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    f.write(f"| {model_label} | hn1024/lh1b88kw | {tau} | {row['BLEU']:.2f} | {row['TERM_ACC_pct']:.2f} | {row['REAL_ADOPT_pct']:.2f} | {row['TERM_FCR_pct']:.2f} | {row['SOURCE_TERM_SENT_FCR_pct']:.2f} | {row['StreamLAAL']:.1f} |\n")
    f.write(f"\nModel: `{model_name}`\n\nEval TSV: `{eval_tsv}`\n")
print(Path(summary_md).read_text())
PY

"${WANDB_PYTHON}" "${ROOT_DIR}/documents/code/general/wandb_tool.py" --project simuleval_eval db-sync \
  --family medicine_sample404_speech_llm_quick --best-bundles --limit 20 || true

echo "[ALL DONE] Medicine sample404 HN1024 tau=${RAG_SCORE_THRESHOLD} New V9 eval complete: ${OUTPUT_BASE}"
echo "[SUMMARY] ${SUMMARY_MD}"
