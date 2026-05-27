#!/usr/bin/env bash
set -euo pipefail

# Minimal single-paper evaluation for the 8h audit-rank-neg experiment.
#
# Runs SimulEval + offline_streamlaal_eval on paper 2022.acl-long.110 only,
# lm=1, per-paper extracted glossary. Supports two modes via MODEL_TYPE:
#   - speech_llm : HF-exported Qwen3-Omni checkpoint, uses run_phase5 pipeline
#   - old_slm    : the owaski baseline checkpoint, uses eval_density_unified
#                  directly with per-paper extracted glossary index.
#
# No silent fallbacks: every required env var has a :? assertion.
#
# Outputs land in OUTPUT_BASE/zh/d${DENSITY_TAG}_lm1_k10_g*_pp2022.acl-long.110
# and the combined per-paper dir. A summary row is appended to the shared
# PAPER110_SUMMARY_TSV so aggregate_rank_ablation_report.py can join.

# ======Configuration=====
ROOT_DIR="/home/jiaxuanluo/InfiniSST"
EVAL_SCRIPT="${ROOT_DIR}/documents/code/simuleval/eval_density_unified.sh"
PHASE5_SCRIPT="${ROOT_DIR}/documents/code/simuleval/run_phase5_model_eval.sh"
PAPER_ID="2022.acl-long.110"
PER_PAPER_GLOSSARY="${ROOT_DIR}/documents/data/data_pre/extracted_glossaries_by_paper/extracted_glossary__2022.acl-long.110.json"

OUTPUT_BASE_DEFAULT="/mnt/gemini/data2/jiaxuanluo/density_eval_rank_ablation"
RAG_TOP_K_DEFAULT="10"
GPUS_DEFAULT="7,5,6"
LATENCY_MULTIPLIER_DEFAULT="1"
RAG_RETRIEVE_STRIDE_SEC_DEFAULT="1.92"
VLLM_DISABLE_CUSTOM_ALL_REDUCE_DEFAULT="1"

# Per-paper extracted glossary maxsim index (must pre-exist; see
# retriever/gigaspeech/modal/build_index_multi_gpu.py).
PER_PAPER_INDEX_DEFAULT="/mnt/gemini/data2/jiaxuanluo/maxsim_index_cache/q3rag_scale_lora-r128-tr128_bs10752_t=0.03_3var_clean_gc_wr1000k_m0.1_maxsim_sp07_best_acl6060_gs10000__extracted_glossary__2022.acl-long.110__maxsim.pt"

RAG_MODEL_PATH_DEFAULT="/mnt/taurus/data/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs10752_t=0.03_3var_clean_gc_wr1000k_m0.1_maxsim_sp07_best_acl6060_gs10000.pt"

# spaCyEnv bin path for offline eval (activated only for speech_llm path;
# old_slm path uses eval_density_unified directly which should already set
# its own python env).
SPACY_ENV_BIN="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin"
# ======Configuration=====

MODEL_TYPE="${MODEL_TYPE:?MODEL_TYPE required (speech_llm|old_slm)}"
DENSITY_TAG="${DENSITY_TAG:?DENSITY_TAG required (unique run identifier)}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME required (HF dir for speech_llm, checkpoint dir for old_slm)}"

OUTPUT_BASE="${OUTPUT_BASE:-${OUTPUT_BASE_DEFAULT}}"
RAG_TOP_K="${RAG_TOP_K:-${RAG_TOP_K_DEFAULT}}"
GPUS="${GPUS:-${GPUS_DEFAULT}}"
LATENCY_MULTIPLIER="${LATENCY_MULTIPLIER:-${LATENCY_MULTIPLIER_DEFAULT}}"
RAG_RETRIEVE_STRIDE_SEC="${RAG_RETRIEVE_STRIDE_SEC:-${RAG_RETRIEVE_STRIDE_SEC_DEFAULT}}"
VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-${VLLM_DISABLE_CUSTOM_ALL_REDUCE_DEFAULT}}"
RAG_MODEL_PATH="${RAG_MODEL_PATH:-${RAG_MODEL_PATH_DEFAULT}}"
PER_PAPER_INDEX="${PER_PAPER_INDEX:-${PER_PAPER_INDEX_DEFAULT}}"

if [[ ! -d "${MODEL_NAME}" ]]; then
  echo "[ERROR] MODEL_NAME not a directory: ${MODEL_NAME}" >&2
  exit 2
fi
if [[ ! -f "${PER_PAPER_GLOSSARY}" ]]; then
  echo "[ERROR] per-paper glossary missing: ${PER_PAPER_GLOSSARY}" >&2
  exit 2
fi

echo "[paper110-eval] ============================================"
echo "[paper110-eval] MODEL_TYPE=${MODEL_TYPE}"
echo "[paper110-eval] DENSITY_TAG=${DENSITY_TAG}"
echo "[paper110-eval] MODEL_NAME=${MODEL_NAME}"
echo "[paper110-eval] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[paper110-eval] GPUS=${GPUS} LM=${LATENCY_MULTIPLIER}"
echo "[paper110-eval] PAPER_ID=${PAPER_ID}"
echo "[paper110-eval] PER_PAPER_GLOSSARY=${PER_PAPER_GLOSSARY}"
echo "[paper110-eval] ============================================"

case "${MODEL_TYPE}" in
  speech_llm)
    # Reuse the phase5 wrapper. We leave it to scan the paper_inputs map and
    # iterate only paper-110 via RUN_PAPERS_OVERRIDE (honored inside
    # run_one_density_eval.sh).
    export RUN_PAPERS_OVERRIDE="${PAPER_ID}"
    DENSITY_TAG="${DENSITY_TAG}" \
    MODEL_NAME="${MODEL_NAME}" \
    OUTPUT_BASE="${OUTPUT_BASE}" \
    GPUS="${GPUS}" \
    RAG_TOP_K="${RAG_TOP_K}" \
    LATENCY_MULTIPLIER="${LATENCY_MULTIPLIER}" \
    RAG_RETRIEVE_STRIDE_SEC="${RAG_RETRIEVE_STRIDE_SEC}" \
    VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE}" \
    bash "${PHASE5_SCRIPT}"
    ;;
  old_slm)
    # Old SLM path: call eval_density_unified directly with per-paper
    # extracted glossary + per-paper maxsim index + per-paper dev list.
    if [[ ! -f "${PER_PAPER_INDEX}" ]]; then
      echo "[ERROR] per-paper maxsim index missing: ${PER_PAPER_INDEX}" >&2
      exit 2
    fi
    PAPER_INPUTS_DIR="${OUTPUT_BASE}/zh/__paper_inputs__/lists"
    # If the per-paper dev lists for this OUTPUT_BASE don't exist yet, copy
    # them from the shared density_eval_maxsim_fixed location so we don't
    # have to rerun prepare_extracted_glossary_by_paper_inputs.py.
    if [[ ! -f "${PAPER_INPUTS_DIR}/dev.source__${PAPER_ID}.txt" ]]; then
      SHARED_PAPER_INPUTS="/mnt/gemini/data2/jiaxuanluo/density_eval_maxsim_fixed/zh/__paper_inputs__/lists"
      if [[ ! -f "${SHARED_PAPER_INPUTS}/dev.source__${PAPER_ID}.txt" ]]; then
        echo "[ERROR] shared paper_inputs missing: ${SHARED_PAPER_INPUTS}" >&2
        exit 2
      fi
      mkdir -p "${PAPER_INPUTS_DIR}"
      cp "${SHARED_PAPER_INPUTS}/dev.source__${PAPER_ID}.txt" "${PAPER_INPUTS_DIR}/"
      cp "${SHARED_PAPER_INPUTS}/dev.target.zh__${PAPER_ID}.txt" "${PAPER_INPUTS_DIR}/"
    fi
    SRC_LIST="${PAPER_INPUTS_DIR}/dev.source__${PAPER_ID}.txt"
    TGT_LIST="${PAPER_INPUTS_DIR}/dev.target.zh__${PAPER_ID}.txt"

    # Invoke eval_density_unified directly (produces instances.log only;
    # offline eval runs below).
    MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
    EVAL_MODE_OVERRIDE="extracted_by_paper" \
    OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
    CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${GPUS}" \
    LATENCY_MULTIPLIER_OVERRIDE="${LATENCY_MULTIPLIER}" \
    DENSITY_TAG="${DENSITY_TAG}" \
    SKIP_OFFLINE_EVAL="1" \
    GLOSSARY_PATH_OVERRIDE="${PER_PAPER_GLOSSARY}" \
    INDEX_PATH_OVERRIDE="${PER_PAPER_INDEX}" \
    SRC_LIST_OVERRIDE="${SRC_LIST}" \
    TGT_LIST_OVERRIDE="${TGT_LIST}" \
    PAPER_ID_TAG="${PAPER_ID}" \
    RAG_RETRIEVE_STRIDE_SEC_OVERRIDE="${RAG_RETRIEVE_STRIDE_SEC}" \
    RAG_TOP_K_OVERRIDE="${RAG_TOP_K}" \
    RAG_MODEL_PATH_OVERRIDE="${RAG_MODEL_PATH}" \
    bash "${EVAL_SCRIPT}"

    # Offline eval (extracted_by_paper mode is the same as the speech_llm
    # path, but only a single paper's instances were produced so the combined
    # dir equals the per-paper dir).
    GLOSSARY_TAG="$(basename "${PER_PAPER_GLOSSARY}" .json)"
    PP_DIR="${OUTPUT_BASE}/zh/d${DENSITY_TAG}_lm${LATENCY_MULTIPLIER}_k${RAG_TOP_K}_g${GLOSSARY_TAG}_pp${PAPER_ID}"
    PP_INST="${PP_DIR}/instances.log"
    if [[ ! -f "${PP_INST}" ]] || [[ ! -s "${PP_INST}" ]]; then
      echo "[ERROR] old_slm instances.log missing or empty: ${PP_INST}" >&2
      exit 2
    fi

    if [[ -x "${SPACY_ENV_BIN}/python3" ]]; then
      export PATH="${SPACY_ENV_BIN}:${PATH}"
    fi
    GLOSSARY_ACL6060="${ROOT_DIR}/retriever/gigaspeech/data_pre/glossary_acl6060.json"
    EXTRACTED_GLOSSARY_MANIFEST="${ROOT_DIR}/documents/data/data_pre/extracted_glossary_by_paper_manifest.json"
    OFFLINE_EVAL_SCRIPT="${ROOT_DIR}/documents/code/offline_sst_eval/offline_streamlaal_eval.py"
    python3 "${OFFLINE_EVAL_SCRIPT}" \
      --mode extracted_by_paper \
      --instances-log "${PP_INST}" \
      --lang-code zh \
      --glossary-acl6060 "${GLOSSARY_ACL6060}" \
      --extracted-glossary-manifest "${EXTRACTED_GLOSSARY_MANIFEST}" \
      --output-tsv "${PP_DIR}/eval_results_by_paper.tsv" \
      --output-log "${PP_DIR}/eval_results_by_paper.log"
    ;;
  *)
    echo "[ERROR] Unknown MODEL_TYPE: ${MODEL_TYPE}" >&2
    exit 2
    ;;
esac

echo "[paper110-eval] DONE for ${DENSITY_TAG}."
