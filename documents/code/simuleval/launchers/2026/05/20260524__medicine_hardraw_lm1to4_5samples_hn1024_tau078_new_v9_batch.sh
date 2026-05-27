#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
EVAL_SCRIPT="${ROOT_DIR}/documents/code/simuleval/eval_density_unified.sh"
PREP_SCRIPT="${ROOT_DIR}/documents/code/simuleval/prepare_medicine_one_talk_inputs.py"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)}"
LANG_CODE="${LANG_CODE_OVERRIDE:-zh}"
TARGET_SAMPLES="${TARGET_SAMPLES_OVERRIDE:-404 545006 596001 605000 606}"
TARGET_LMS="${TARGET_LMS_OVERRIDE:-1 2 3 4}"
GPU_PAIR="${GPU_PAIR:-0,1}"
RAG_GPU_OVERRIDE="${RAG_GPU_OVERRIDE:-cuda:1}"
RAG_TOP_K="${RAG_TOP_K_OVERRIDE:-10}"
RAG_SCORE_THRESHOLD="${RAG_SCORE_THRESHOLD_OVERRIDE:-0.78}"
RAG_LORA_R="${RAG_LORA_R_OVERRIDE:-128}"
RAG_TEXT_LORA_R="${RAG_TEXT_LORA_R_OVERRIDE:-128}"
RAG_TIMELINE_LOOKBACK_SEC="${RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE:-1.92}"
MODEL_LABEL="${MODEL_LABEL:-new_v9_termtag_delay_oldnewv3_r32a64}"

MODEL_NAME="${MODEL_NAME_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/slm_exports/speech_llm_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_zh_r32a64_tp2_taurus8/keep1.0_r32/v0-20260524-062743-hf}"
HN1024_CKPT="${HN1024_CKPT_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
HARD_RAW_GLOSSARY="${HARD_RAW_GLOSSARY_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/medicine_eval_hard_terms_llm_judge_manual_20260524/hard_medicine_glossary_raw_llm_judge_manual_zh215_unique212.json}"
RUNTIME_GLOSSARY="${RUNTIME_GLOSSARY_OVERRIDE:-${HARD_RAW_GLOSSARY}}"
FIXED_RAW_GLOSSARY="${FIXED_RAW_GLOSSARY_OVERRIDE:-${HARD_RAW_GLOSSARY}}"
ESO_TEST_ROOT="${ESO_TEST_ROOT_OVERRIDE:-/home/jiaxingxu/rag-sst/eso-dataset/outputs_v2/test}"

OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_hardraw_hn1024_tau078_new_v9_batch_${RUN_STAMP}}"
MEDICINE_INPUTS="${OUTPUT_BASE}/${LANG_CODE}/__medicine_inputs__/lists"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs/medicine_hardraw_hn1024_tau078_new_v9_batch_${RUN_STAMP}}"
INDEX_CACHE_DIR_BASE="${INDEX_CACHE_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/medicine_hardraw_batch_${RUN_STAMP}}"
EVAL_TMPDIR="${EVAL_TMPDIR_OVERRIDE:-/tmp/jx_med_hardraw_batch_${RUN_STAMP}}"

RUNTIME_GLOSSARY_TAG="$(basename "${RUNTIME_GLOSSARY}" .json)"
TAU_TAG="${RAG_SCORE_THRESHOLD/./p}"
DENSITY_TAG="${DENSITY_TAG_OVERRIDE:-medhard5_${MODEL_LABEL}_hn1024_tau${TAU_TAG}_raw}"
DEFAULT_GLOSSARY_TAG_PATTERN='hard_medicine_raw__medicine_{sample}'
DEFAULT_ORACLE_TERM_MAP_TAG_PATTERN='hard_medicine.oracle_term_map__medicine_{sample}'
GLOSSARY_TAG_PATTERN="${GLOSSARY_TAG_PATTERN_OVERRIDE:-${DEFAULT_GLOSSARY_TAG_PATTERN}}"
ORACLE_TERM_MAP_TAG_PATTERN="${ORACLE_TERM_MAP_TAG_PATTERN_OVERRIDE:-${DEFAULT_ORACLE_TERM_MAP_TAG_PATTERN}}"
COMBINED_PREFIX="${COMBINED_PREFIX_OVERRIDE:-medicine5_hardraw}"
COMBINED_GLOSSARY_TAG="${COMBINED_GLOSSARY_TAG_OVERRIDE:-hard_medicine_raw__medicine5}"

for p in "${EVAL_SCRIPT}" "${PREP_SCRIPT}" "${MODEL_NAME}" "${HN1024_CKPT}" "${RUNTIME_GLOSSARY}" "${FIXED_RAW_GLOSSARY}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

mkdir -p "${MEDICINE_INPUTS}" "${LOG_ROOT}" "${INDEX_CACHE_DIR_BASE}" "${EVAL_TMPDIR}"

render_sample_pattern() {
  local pattern="$1"
  local sample="$2"
  printf '%s' "${pattern}" | sed "s/{sample}/${sample}/g"
}

clean_shm() {
  local me
  me="$(id -un)"
  find /dev/shm -maxdepth 1 -user "${me}" -mmin +1 \
    \( -name 'psm_*' -o -name 'loky-*' -o -name 'torch_*' -o -name 'vllm_*' \) \
    -delete 2>/dev/null || true
}

prepare_sample() {
  local sample="$1"
  local glossary_tag oracle_tag
  glossary_tag="$(render_sample_pattern "${GLOSSARY_TAG_PATTERN}" "${sample}")"
  oracle_tag="$(render_sample_pattern "${ORACLE_TERM_MAP_TAG_PATTERN}" "${sample}")"
  local manifest="${MEDICINE_INPUTS}/medicine_inputs_manifest__medicine_${sample}.json"
  if [[ -s "${manifest}" && "${FORCE_PREPARE_OVERRIDE:-0}" != "1" ]]; then
    echo "[SKIP] prepared sample=${sample}: ${manifest}"
    return 0
  fi
  echo "[PREP] sample=${sample}"
  python3 "${PREP_SCRIPT}" \
    --sample-id "${sample}" \
    --lang-code "${LANG_CODE}" \
    --eso-test-root "${ESO_TEST_ROOT}" \
    --term-source glossary_match \
    --oracle-glossary "${FIXED_RAW_GLOSSARY}" \
    --eval-glossary "${FIXED_RAW_GLOSSARY}" \
    --strict-glossary "${FIXED_RAW_GLOSSARY}" \
    --glossary-tag "${glossary_tag}" \
    --oracle-term-map-tag "${oracle_tag}" \
    --output-dir "${MEDICINE_INPUTS}" \
    --max-sentences "${MAX_SENTENCES_OVERRIDE:-0}"
}

build_combined_inputs() {
  local src_list="${MEDICINE_INPUTS}/medicine.source__${COMBINED_PREFIX}.txt"
  local tgt_list="${MEDICINE_INPUTS}/medicine.target.${LANG_CODE}__${COMBINED_PREFIX}.txt"
  local source_text="${MEDICINE_INPUTS}/medicine.source_text.en__${COMBINED_PREFIX}.txt"
  local ref_file="${MEDICINE_INPUTS}/medicine.ref.${LANG_CODE}__${COMBINED_PREFIX}.txt"
  local audio_yaml="${MEDICINE_INPUTS}/medicine.audio__${COMBINED_PREFIX}.yaml"
  local manifest="${MEDICINE_INPUTS}/medicine_inputs_manifest__${COMBINED_PREFIX}.json"
  local glossary_out="${MEDICINE_INPUTS}/${COMBINED_GLOSSARY_TAG}.json"

  if [[ -s "${manifest}" && "${FORCE_PREPARE_OVERRIDE:-0}" != "1" ]]; then
    echo "[SKIP] combined inputs: ${manifest}"
    return 0
  fi

  : > "${src_list}"
  : > "${tgt_list}"
  : > "${source_text}"
  : > "${ref_file}"
  : > "${audio_yaml}"

  for sample in ${TARGET_SAMPLES}; do
    local prefix="medicine_${sample}"
    for p in \
      "${MEDICINE_INPUTS}/medicine.source__${prefix}.txt" \
      "${MEDICINE_INPUTS}/medicine.target.${LANG_CODE}__${prefix}.txt" \
      "${MEDICINE_INPUTS}/medicine.source_text.en__${prefix}.txt" \
      "${MEDICINE_INPUTS}/medicine.ref.${LANG_CODE}__${prefix}.txt" \
      "${MEDICINE_INPUTS}/medicine.audio__${prefix}.yaml"; do
      if [[ ! -s "${p}" ]]; then
        echo "[ERROR] Prepared sample input missing or empty: ${p}" >&2
        exit 3
      fi
    done
    cat "${MEDICINE_INPUTS}/medicine.source__${prefix}.txt" >> "${src_list}"
    cat "${MEDICINE_INPUTS}/medicine.target.${LANG_CODE}__${prefix}.txt" >> "${tgt_list}"
    cat "${MEDICINE_INPUTS}/medicine.source_text.en__${prefix}.txt" >> "${source_text}"
    cat "${MEDICINE_INPUTS}/medicine.ref.${LANG_CODE}__${prefix}.txt" >> "${ref_file}"
    cat "${MEDICINE_INPUTS}/medicine.audio__${prefix}.yaml" >> "${audio_yaml}"
  done

  # Use the complete hard raw glossary for fixed-denominator TERM metrics;
  # runtime retrieval may use a larger glossary via RUNTIME_GLOSSARY.
  cp "${FIXED_RAW_GLOSSARY}" "${glossary_out}"
  python3 - <<PY
import json
from pathlib import Path
manifest = {
  "lang_code": "${LANG_CODE}",
  "samples": "${TARGET_SAMPLES}".split(),
  "combined_prefix": "${COMBINED_PREFIX}",
  "runtime_glossary": "${RUNTIME_GLOSSARY}",
  "eval_glossary": "${glossary_out}",
  "files": {
    "source_list": "${src_list}",
    "target_list": "${tgt_list}",
    "source_text": "${source_text}",
    "reference_text": "${ref_file}",
    "audio_yaml": "${audio_yaml}",
    "glossary": "${glossary_out}",
  },
}
Path("${manifest}").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
PY
  echo "[PREP] combined inputs: ${manifest}"
}

run_lm_batch() {
  local lm="$1"
  local eval_glossary="${MEDICINE_INPUTS}/${COMBINED_GLOSSARY_TAG}.json"
  local src_list="${MEDICINE_INPUTS}/medicine.source__${COMBINED_PREFIX}.txt"
  local tgt_list="${MEDICINE_INPUTS}/medicine.target.${LANG_CODE}__${COMBINED_PREFIX}.txt"
  local source_text="${MEDICINE_INPUTS}/medicine.source_text.en__${COMBINED_PREFIX}.txt"
  local ref_file="${MEDICINE_INPUTS}/medicine.ref.${LANG_CODE}__${COMBINED_PREFIX}.txt"
  local audio_yaml="${MEDICINE_INPUTS}/medicine.audio__${COMBINED_PREFIX}.yaml"
  local out_dir="${OUTPUT_BASE}/${LANG_CODE}/d${DENSITY_TAG}_lm${lm}_k${RAG_TOP_K}_th${RAG_SCORE_THRESHOLD}_g${RUNTIME_GLOSSARY_TAG}_pp${COMBINED_PREFIX}"
  local eval_tsv="${out_dir}/eval_results.tsv"
  local log_prefix="${LOG_ROOT}/batch_${COMBINED_PREFIX}_lm${lm}"

  for p in "${eval_glossary}" "${src_list}" "${tgt_list}" "${source_text}" "${ref_file}" "${audio_yaml}"; do
    if [[ ! -s "${p}" ]]; then
      echo "[ERROR] Combined input missing or empty: ${p}" >&2
      exit 3
    fi
  done

  if [[ -s "${eval_tsv}" && "${FORCE_RERUN_OVERRIDE:-0}" != "1" ]]; then
    echo "[SKIP] lm=${lm}: ${eval_tsv}"
    return 0
  fi

  echo "[RUN] combined_samples=${TARGET_SAMPLES} lm=${lm} gpu=${GPU_PAIR}"
  clean_shm
  mkdir -p "${EVAL_TMPDIR}/lm${lm}/triton" \
    "${EVAL_TMPDIR}/lm${lm}/torchinductor" \
    "${EVAL_TMPDIR}/lm${lm}/xdg"
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${GPU_PAIR}" \
  TRITON_CACHE_DIR="${EVAL_TMPDIR}/lm${lm}/triton" \
  TORCHINDUCTOR_CACHE_DIR="${EVAL_TMPDIR}/lm${lm}/torchinductor" \
  XDG_CACHE_HOME="${EVAL_TMPDIR}/lm${lm}/xdg" \
  MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
  RAG_MODEL_PATH_OVERRIDE="${HN1024_CKPT}" \
  RAG_LORA_R_OVERRIDE="${RAG_LORA_R}" \
  RAG_TEXT_LORA_R_OVERRIDE="${RAG_TEXT_LORA_R}" \
  OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
  EVAL_MODE_OVERRIDE="acl6060" \
  LANG_CODE_OVERRIDE="${LANG_CODE}" \
  GLOSSARY_PATH_OVERRIDE="${RUNTIME_GLOSSARY}" \
  EVAL_GLOSSARY_PATH_OVERRIDE="${eval_glossary}" \
  SRC_LIST_OVERRIDE="${src_list}" \
  TGT_LIST_OVERRIDE="${tgt_list}" \
  REF_FILE_OVERRIDE="${ref_file}" \
  SOURCE_TEXT_FILE_OVERRIDE="${source_text}" \
  AUDIO_YAML_OVERRIDE="${audio_yaml}" \
  LATENCY_MULTIPLIER_OVERRIDE="${lm}" \
  RAG_TOP_K_OVERRIDE="${RAG_TOP_K}" \
  RAG_SCORE_THRESHOLD_OVERRIDE="${RAG_SCORE_THRESHOLD}" \
  RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE="${RAG_TIMELINE_LOOKBACK_SEC}" \
  RAG_STREAMING_MODE_OVERRIDE="timeline" \
  TERM_MAP_FORMAT_OVERRIDE="plain" \
  STRIP_OUTPUT_TAGS_OVERRIDE="term" \
  TERM_FCR_POLICY="term_map_source_ref_negative_sentence" \
  DENSITY_TAG="${DENSITY_TAG}" \
  PAPER_ID_TAG="${COMBINED_PREFIX}" \
  INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR_BASE}/lm${lm}" \
  EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR}/lm${lm}" \
  RAG_GPU_OVERRIDE="${RAG_GPU_OVERRIDE}" \
  GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}" \
  MAX_CACHE_SECONDS_OVERRIDE="${MAX_CACHE_SECONDS_OVERRIDE:-80.0}" \
  KEEP_CACHE_SECONDS_OVERRIDE="${KEEP_CACHE_SECONDS_OVERRIDE:-60.0}" \
  VLLM_LIMIT_AUDIO_OVERRIDE="${VLLM_LIMIT_AUDIO_OVERRIDE:-8}" \
  MAX_NEW_TOKENS_OVERRIDE="${MAX_NEW_TOKENS_OVERRIDE:-40}" \
  VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN_OVERRIDE:-8192}" \
  VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}" \
  VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}" \
  VLLM_USE_FUSED_MOE_GROUPED_TOPK="${VLLM_USE_FUSED_MOE_GROUPED_TOPK:-0}" \
  CLEAN_OUTPUT_DIR_OVERRIDE="${CLEAN_OUTPUT_DIR_OVERRIDE:-0}" \
  VLLM_TP_SIZE_OVERRIDE="${VLLM_TP_SIZE_OVERRIDE:-2}" \
  bash "${EVAL_SCRIPT}" > "${log_prefix}.out" 2> "${log_prefix}.err"
  clean_shm
}

echo "[INFO] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] RUNTIME_GLOSSARY=${RUNTIME_GLOSSARY}"
echo "[INFO] FIXED_RAW_GLOSSARY=${FIXED_RAW_GLOSSARY}"
echo "[INFO] TARGET_SAMPLES=${TARGET_SAMPLES}"
echo "[INFO] TARGET_LMS=${TARGET_LMS}"
echo "[INFO] GPU_PAIR=${GPU_PAIR}"

for sample in ${TARGET_SAMPLES}; do
  prepare_sample "${sample}"
done
build_combined_inputs

if [[ "${PREP_ONLY_OVERRIDE:-0}" == "1" ]]; then
  echo "[PREP ONLY] medicine hard raw combined inputs ready: ${MEDICINE_INPUTS}"
  exit 0
fi

for lm in ${TARGET_LMS}; do
  run_lm_batch "${lm}"
done

echo "[ALL DONE] medicine hard raw batch eval complete: ${OUTPUT_BASE}"
