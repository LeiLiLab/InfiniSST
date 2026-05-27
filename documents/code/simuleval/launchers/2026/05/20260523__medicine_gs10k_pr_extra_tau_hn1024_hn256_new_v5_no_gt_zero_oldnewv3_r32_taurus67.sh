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

RUN_STAMP="${RUN_STAMP:-20260523T1810}"
LANG_CODE="${LANG_CODE_OVERRIDE:-zh}"
TARGET_SAMPLE="${TARGET_SAMPLE_OVERRIDE:-404}"
TARGET_LM="${TARGET_LM_OVERRIDE:-2}"
GPU_PAIR="${GPU_PAIR:-6,7}"
RAG_GPU_OVERRIDE="${RAG_GPU_OVERRIDE:-cuda:1}"
RAG_TOP_K="${RAG_TOP_K_OVERRIDE:-10}"
RAG_LORA_R="${RAG_LORA_R_OVERRIDE:-128}"
RAG_TEXT_LORA_R="${RAG_TEXT_LORA_R_OVERRIDE:-128}"
RAG_TIMELINE_LOOKBACK_SEC="${RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE:-1.92}"
MODEL_LABEL="${MODEL_LABEL:-new_v5_no_gt_zero_oldnewv3_r32a64}"

MODEL_NAME="${MODEL_NAME_OVERRIDE:-/mnt/aries/data6/jiaxuanluo/slm/speech_llm_new_v5_no_gt_zero_llm_variant_aug_oldnewv3_zh_r32a64_tp2_aries2/keep1.0_r32/v0-20260523-050346-hf}"
RUNTIME_GLOSSARY_GS10K="${RUNTIME_GLOSSARY_GS10K_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_glossary_gt_plus_medicine_wiki_gs10000_translated.json}"
STRICT_JSONL="${STRICT_MEDICINE_JSONL_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_dev_dataset.jsonl}"
STRICT_GLOSSARY="${STRICT_MEDICINE_GLOSSARY_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_glossary_gt_plus_medicine_wiki_gs10000_translated.json}"
ESO_TEST_ROOT="${ESO_TEST_ROOT_OVERRIDE:-/home/jiaxingxu/rag-sst/eso-dataset/outputs_v2/test}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260523__medicine_gs10k_pr_extra_tau_hn1024_hn256_new_v5_no_gt_zero_oldnewv3_r32.md}"

OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_gs10k_pr_extra_tau_new_v5_no_gt_zero_oldnewv3_r32_${RUN_STAMP}}"
MEDICINE_INPUTS="${OUTPUT_BASE}/${LANG_CODE}/__medicine_inputs__/lists"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs/medicine_gs10k_pr_extra_tau_new_v5_${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/medicine_gs10k_pr_sweep}"
EVAL_TMPDIR="${EVAL_TMPDIR_OVERRIDE:-/tmp/jx_mpr_extra_${RUN_STAMP}}"
SUMMARY_TSV="${OUTPUT_BASE}/summary_medicine_gs10k_pr_extra_tau_metrics.tsv"
SUMMARY_MD="${OUTPUT_BASE}/summary_medicine_gs10k_pr_extra_tau_metrics.md"
SETTING_FILTER="${SETTING_FILTER_OVERRIDE:-}"

HN1024_CKPT="${HN1024_CKPT_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
HN256_CKPT="${HN256_CKPT_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8190_gc256_wr1000k_m0.0_maxsim_mfa_variantE_hn256_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu012367_taurus_step1200_aclmetric_reset_latest.pt}"

for p in "${EVAL_SCRIPT}" "${PREP_SCRIPT}" "${WANDB_LOGGER}" "${MODEL_NAME}" \
         "${RUNTIME_GLOSSARY_GS10K}" "${STRICT_JSONL}" "${STRICT_GLOSSARY}" \
         "${HN1024_CKPT}" "${HN256_CKPT}" "${NOTES_FILE}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

mkdir -p "${MEDICINE_INPUTS}" "${LOG_ROOT}" "${INDEX_CACHE_DIR}" "${EVAL_TMPDIR}"

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

should_run_setting() {
  local key="$1"
  if [[ -z "${SETTING_FILTER}" ]]; then
    return 0
  fi
  [[ ",${SETTING_FILTER}," == *",${key},"* ]]
}

run_setting() {
  local setting="$1" retriever_id="$2" ckpt="$3" tau="$4" pr="$5"
  local tau_tag="${tau/./p}"
  local setting_key="${setting}_tau${tau_tag}"
  local density="med1_${MODEL_LABEL}_${setting}_tau${tau_tag}_gs10k"
  local log_prefix="${LOG_ROOT}/${setting}_tau${tau_tag}"
  local out_dir="${OUTPUT_BASE}/${LANG_CODE}/d${density}_lm${TARGET_LM}_k${RAG_TOP_K}_th${tau}_g${RUNTIME_GLOSSARY_TAG}_pp${PREFIX}"
  local eval_tsv="${out_dir}/eval_results.tsv"
  local run_name="${MODEL_LABEL}__medicine_${TARGET_SAMPLE}__gs10k__${setting}_tau${tau_tag}__lm${TARGET_LM}"

  if ! should_run_setting "${setting_key}"; then
    echo "[FILTER-SKIP] setting=${setting_key} SETTING_FILTER=${SETTING_FILTER}"
    return 0
  fi

  if [[ -f "${eval_tsv}" ]] && [[ -s "${eval_tsv}" ]] && [[ "${FORCE_RERUN_OVERRIDE:-0}" != "1" ]]; then
    echo "[SKIP] existing eval: ${eval_tsv}"
  else
    echo "[RUN] setting=${setting} retriever=${retriever_id} tau=${tau} P/R=${pr}"
    clean_shm
    CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${GPU_PAIR}" \
    MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
    RAG_MODEL_PATH_OVERRIDE="${ckpt}" \
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
    RAG_SCORE_THRESHOLD_OVERRIDE="${tau}" \
    RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE="${RAG_TIMELINE_LOOKBACK_SEC}" \
    RAG_STREAMING_MODE_OVERRIDE="timeline" \
    TERM_MAP_FORMAT_OVERRIDE="plain" \
    TERM_FCR_POLICY="term_map_source_ref_negative_sentence" \
    DENSITY_TAG="${density}" \
    PAPER_ID_TAG="${PREFIX}" \
    INDEX_CACHE_DIR="${INDEX_CACHE_DIR}" \
    EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR}/${setting}_tau${tau_tag}" \
    RAG_GPU_OVERRIDE="${RAG_GPU_OVERRIDE}" \
    GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}" \
    CLEAN_OUTPUT_DIR_OVERRIDE="${CLEAN_OUTPUT_DIR_OVERRIDE:-0}" \
    bash "${EVAL_SCRIPT}" > "${log_prefix}.out" 2> "${log_prefix}.err"
    clean_shm
  fi

  if [[ ! -s "${eval_tsv}" ]]; then
    echo "[ERROR] Missing eval TSV after run: ${eval_tsv}" >&2
    return 5
  fi

  HOME="${WANDB_HOME:-/mnt/taurus/home/jiaxuanluo}" \
  WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_HOME:-/mnt/taurus/home/jiaxuanluo}/.config/wandb}" \
  "${WANDB_PYTHON}" "${WANDB_LOGGER}" \
    --project simuleval_eval \
    --run-name "${run_name}" \
    --experiment-family medicine_gs10k_pr_sweep \
    --data-tag "medicine_sample${TARGET_SAMPLE}_gs10k_fixedraw_${LANG_CODE}" \
    --task-tag eval \
    --notes-file "${NOTES_FILE}" \
    --trained-from-run "cg5qisu9" \
    --extra-tags "variant:${setting}_tau${tau_tag}" "retriever:${retriever_id}" "tau:${tau_tag}" "glossary:medicine_gs10k" "lang:${LANG_CODE}" "sample:${PREFIX}" "compute:taurus_direct" "sweep:extra_tau" \
    --density "${density}" \
    --rag-top-k "${RAG_TOP_K}" \
    --rag-score-threshold "${tau}" \
    --output-base "${OUTPUT_BASE}" \
    --lang-code "${LANG_CODE}" \
    --latency-multipliers "${TARGET_LM}" \
    --glossary-tag "${RUNTIME_GLOSSARY_TAG}" \
    --paper-id "${PREFIX}" \
    --model-name "${MODEL_NAME}" \
    --rag-model-path "${ckpt}" \
    --verdict "Medicine gs10k extra tau sweep: ${MODEL_LABEL}, sample=${TARGET_SAMPLE}, lm=${TARGET_LM}, retriever=${retriever_id}, tau=${tau}, report P/R=${pr}, fixed raw per-talk denominator."
}

run_setting "hn1024" "lh1b88kw" "${HN1024_CKPT}" "0.80" "20.00/86.88"
run_setting "hn256" "gsjheh6r" "${HN256_CKPT}" "0.70" "11.22/93.81"
run_setting "hn256" "gsjheh6r" "${HN256_CKPT}" "0.76" "15.56/90.61"

"${WANDB_PYTHON}" - <<'PY' "${OUTPUT_BASE}" "${SUMMARY_TSV}" "${SUMMARY_MD}" "${MODEL_LABEL}"
import csv
import glob
import os
import sys

output_base, summary_tsv, summary_md, model_label = sys.argv[1:5]
rows = []
for path in sorted(glob.glob(os.path.join(output_base, "zh", "dmed1_*", "eval_results.tsv"))):
    parent = os.path.basename(os.path.dirname(path))
    parts = parent.split("_")
    model = "unknown"
    tau = "unknown"
    for i, part in enumerate(parts):
        if part in {"hn1024", "hn256"}:
            model = part
        if part.startswith("tau") and len(part) > 3:
            tau = part[3:].replace("p", ".")
    report_pr = {
        ("hn1024", "0.80"): "20.00/86.88",
        ("hn256", "0.70"): "11.22/93.81",
        ("hn256", "0.76"): "15.56/90.61",
    }.get((model, tau), "")
    with open(path, newline="") as f:
        rec = next(csv.DictReader(f, delimiter="\t"))
    rows.append({
        "model": model,
        "tau": tau,
        "report_P_R": report_pr,
        "BLEU": float(rec.get("BLEU", "nan")),
        "TERM_ACC_pct": float(rec.get("TERM_ACC", "nan")) * 100.0,
        "REAL_ADOPT_pct": float(rec.get("REAL_TERM_ADOPT", "nan")) * 100.0,
        "TERM_FCR_pct": float(rec.get("TERM_FCR", "nan")) * 100.0,
        "SOURCE_TERM_SENT_FCR_pct": float(rec.get("SOURCE_TERM_SENT_FCR", "nan")) * 100.0,
        "StreamLAAL": float(rec.get("StreamLAAL", "nan")),
        "eval_tsv": path,
    })

os.makedirs(os.path.dirname(summary_tsv), exist_ok=True)
fieldnames = ["model", "tau", "report_P_R", "BLEU", "TERM_ACC_pct", "REAL_ADOPT_pct", "TERM_FCR_pct", "SOURCE_TERM_SENT_FCR_pct", "StreamLAAL", "eval_tsv"]
with open(summary_tsv, "w", newline="") as f:
    w = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
    w.writeheader()
    for row in rows:
        w.writerow(row)

with open(summary_md, "w") as f:
    f.write(f"# Medicine gs10k extra tau metrics ({model_label})\n\n")
    f.write("| model | tau | report P/R | BLEU | TERM_ACC | REAL_ADOPT | TERM_FCR | SOURCE_SENT_FCR | StreamLAAL |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in rows:
        f.write(
            f"| {row['model']} | {row['tau']} | {row['report_P_R']} | "
            f"{row['BLEU']:.2f} | {row['TERM_ACC_pct']:.2f} | "
            f"{row['REAL_ADOPT_pct']:.2f} | {row['TERM_FCR_pct']:.2f} | "
            f"{row['SOURCE_TERM_SENT_FCR_pct']:.2f} | {row['StreamLAAL']:.1f} |\n"
        )
PY

"${WANDB_PYTHON}" "${ROOT_DIR}/documents/code/general/wandb_tool.py" --project simuleval_eval db-sync \
  --family medicine_gs10k_pr_sweep --best-bundles --limit 30 || true

echo "[ALL DONE] Medicine gs10k extra tau sweep complete: ${OUTPUT_BASE}"
echo "[SUMMARY] ${SUMMARY_MD}"
