#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
EVAL_SCRIPT="${ROOT_DIR}/documents/code/simuleval/eval_density_unified.sh"
PREP_SCRIPT="${ROOT_DIR}/documents/code/simuleval/prepare_medicine_one_talk_inputs.py"
AGG_SCRIPT="${ROOT_DIR}/documents/code/simuleval/aggregate_medicine_retriever_lm_sweep.py"

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

MODEL_NAME="${MODEL_NAME_OVERRIDE:-/mnt/aries/data7/jiaxuanluo/slm/speech_llm_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_zh_r32a64_tp2_taurus8/keep1.0_r32/v0-20260524-062743-hf}"
HN1024_CKPT="${HN1024_CKPT_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
HARD_RAW_GLOSSARY="${HARD_RAW_GLOSSARY_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/medicine_eval_hard_terms_llm_judge_manual_20260524/hard_medicine_glossary_raw_llm_judge_manual_zh215_unique212.json}"
ESO_TEST_ROOT="${ESO_TEST_ROOT_OVERRIDE:-/home/jiaxingxu/rag-sst/eso-dataset/outputs_v2/test}"

OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_hardraw_hn1024_tau078_new_v9_${RUN_STAMP}}"
MEDICINE_INPUTS="${OUTPUT_BASE}/${LANG_CODE}/__medicine_inputs__/lists"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs/medicine_hardraw_hn1024_tau078_new_v9_${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/medicine_hardraw}"
EVAL_TMPDIR="${EVAL_TMPDIR_OVERRIDE:-/tmp/jx_med_hardraw_${RUN_STAMP}}"

RUNTIME_GLOSSARY_TAG="$(basename "${HARD_RAW_GLOSSARY}" .json)"
TAU_TAG="${RAG_SCORE_THRESHOLD/./p}"
DENSITY_TAG="${DENSITY_TAG_OVERRIDE:-medhard_${MODEL_LABEL}_hn1024_tau${TAU_TAG}_raw}"
AGG_DENSITY_TAG="${AGG_DENSITY_TAG_OVERRIDE:-medhard5_${MODEL_LABEL}_hn1024_tau${TAU_TAG}_raw}"
GLOSSARY_TAG_PATTERN="${GLOSSARY_TAG_PATTERN_OVERRIDE:-hard_medicine_raw__medicine_{sample}}"
ORACLE_TERM_MAP_TAG_PATTERN="${ORACLE_TERM_MAP_TAG_PATTERN_OVERRIDE:-hard_medicine.oracle_term_map__medicine_{sample}}"
COMBINED_GLOSSARY_TAG="${COMBINED_GLOSSARY_TAG_OVERRIDE:-hard_medicine_raw_five_samples}"
AGGREGATE_ONLY="${AGGREGATE_ONLY:-0}"

for p in "${EVAL_SCRIPT}" "${PREP_SCRIPT}" "${AGG_SCRIPT}" "${MODEL_NAME}" "${HN1024_CKPT}" "${HARD_RAW_GLOSSARY}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

mkdir -p "${MEDICINE_INPUTS}" "${LOG_ROOT}" "${INDEX_CACHE_DIR}" "${EVAL_TMPDIR}"

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
    --oracle-glossary "${HARD_RAW_GLOSSARY}" \
    --eval-glossary "${HARD_RAW_GLOSSARY}" \
    --strict-glossary "${HARD_RAW_GLOSSARY}" \
    --glossary-tag "${glossary_tag}" \
    --oracle-term-map-tag "${oracle_tag}" \
    --output-dir "${MEDICINE_INPUTS}" \
    --max-sentences "${MAX_SENTENCES_OVERRIDE:-0}"
}

run_one() {
  local sample="$1"
  local lm="$2"
  local prefix="medicine_${sample}"
  local glossary_tag
  glossary_tag="$(render_sample_pattern "${GLOSSARY_TAG_PATTERN}" "${sample}")"
  local eval_glossary="${MEDICINE_INPUTS}/${glossary_tag}.json"
  local src_list="${MEDICINE_INPUTS}/medicine.source__${prefix}.txt"
  local tgt_list="${MEDICINE_INPUTS}/medicine.target.${LANG_CODE}__${prefix}.txt"
  local source_text="${MEDICINE_INPUTS}/medicine.source_text.en__${prefix}.txt"
  local ref_file="${MEDICINE_INPUTS}/medicine.ref.${LANG_CODE}__${prefix}.txt"
  local audio_yaml="${MEDICINE_INPUTS}/medicine.audio__${prefix}.yaml"
  local out_dir="${OUTPUT_BASE}/${LANG_CODE}/d${DENSITY_TAG}_lm${lm}_k${RAG_TOP_K}_th${RAG_SCORE_THRESHOLD}_g${RUNTIME_GLOSSARY_TAG}_pp${prefix}"
  local eval_tsv="${out_dir}/eval_results.tsv"
  local log_prefix="${LOG_ROOT}/sample${sample}_lm${lm}"

  for p in "${eval_glossary}" "${src_list}" "${tgt_list}" "${source_text}" "${ref_file}" "${audio_yaml}"; do
    if [[ ! -s "${p}" ]]; then
      echo "[ERROR] Prepared input missing or empty: ${p}" >&2
      exit 3
    fi
  done

  if [[ -s "${eval_tsv}" && "${FORCE_RERUN_OVERRIDE:-0}" != "1" ]]; then
    echo "[SKIP] sample=${sample} lm=${lm}: ${eval_tsv}"
    return 0
  fi

  echo "[RUN] sample=${sample} lm=${lm} gpu=${GPU_PAIR}"
  clean_shm
  mkdir -p "${EVAL_TMPDIR}/sample${sample}_lm${lm}/triton" \
    "${EVAL_TMPDIR}/sample${sample}_lm${lm}/torchinductor" \
    "${EVAL_TMPDIR}/sample${sample}_lm${lm}/xdg"
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${GPU_PAIR}" \
  TRITON_CACHE_DIR="${EVAL_TMPDIR}/sample${sample}_lm${lm}/triton" \
  TORCHINDUCTOR_CACHE_DIR="${EVAL_TMPDIR}/sample${sample}_lm${lm}/torchinductor" \
  XDG_CACHE_HOME="${EVAL_TMPDIR}/sample${sample}_lm${lm}/xdg" \
  MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
  RAG_MODEL_PATH_OVERRIDE="${HN1024_CKPT}" \
  RAG_LORA_R_OVERRIDE="${RAG_LORA_R}" \
  RAG_TEXT_LORA_R_OVERRIDE="${RAG_TEXT_LORA_R}" \
  OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
  EVAL_MODE_OVERRIDE="acl6060" \
  LANG_CODE_OVERRIDE="${LANG_CODE}" \
  GLOSSARY_PATH_OVERRIDE="${HARD_RAW_GLOSSARY}" \
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
  PAPER_ID_TAG="${prefix}" \
  INDEX_CACHE_DIR="${INDEX_CACHE_DIR}" \
  EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR}/sample${sample}_lm${lm}" \
  RAG_GPU_OVERRIDE="${RAG_GPU_OVERRIDE}" \
  GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}" \
  CLEAN_OUTPUT_DIR_OVERRIDE="${CLEAN_OUTPUT_DIR_OVERRIDE:-0}" \
  VLLM_TP_SIZE_OVERRIDE="${VLLM_TP_SIZE_OVERRIDE:-2}" \
  bash "${EVAL_SCRIPT}" > "${log_prefix}.out" 2> "${log_prefix}.err"
  clean_shm
}

aggregate_results() {
  python3 "${AGG_SCRIPT}" \
    --output-base "${OUTPUT_BASE}" \
    --samples 404 545006 596001 605000 606 \
    --lms 1 2 3 4 \
    --lang-code "${LANG_CODE}" \
    --density-tag "${DENSITY_TAG}" \
    --aggregate-density-tag "${AGG_DENSITY_TAG}" \
    --runtime-glossary-tag "${RUNTIME_GLOSSARY_TAG}" \
    --combined-glossary-tag "${COMBINED_GLOSSARY_TAG}" \
    --glossary-tag-pattern "${GLOSSARY_TAG_PATTERN}" \
    --rag-top-k "${RAG_TOP_K}" \
    --rag-score-threshold "${RAG_SCORE_THRESHOLD}" \
    --term-fcr-policy "term_map_source_ref_negative_sentence" \
    --strip-output-tags "term"
}

echo "[INFO] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] HARD_RAW_GLOSSARY=${HARD_RAW_GLOSSARY}"
echo "[INFO] TARGET_SAMPLES=${TARGET_SAMPLES}"
echo "[INFO] TARGET_LMS=${TARGET_LMS}"
echo "[INFO] GPU_PAIR=${GPU_PAIR}"

if [[ "${AGGREGATE_ONLY}" == "1" ]]; then
  aggregate_results
  exit 0
fi

for sample in ${TARGET_SAMPLES}; do
  prepare_sample "${sample}"
done

for lm in ${TARGET_LMS}; do
  for sample in ${TARGET_SAMPLES}; do
    run_one "${sample}" "${lm}"
  done
done

if [[ "${SKIP_AGGREGATE_OVERRIDE:-0}" != "1" ]]; then
  aggregate_results || echo "[WARN] aggregate failed; another split may still be running."
fi

echo "[ALL DONE] medicine hard raw eval split complete: ${OUTPUT_BASE}"
