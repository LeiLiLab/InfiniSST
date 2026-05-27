#!/usr/bin/env bash
#SBATCH --job-name=med1_lh1b88kw
#SBATCH --partition=aries
#SBATCH --exclusive
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_med1_lh1b88kw.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_med1_lh1b88kw.err

set -euo pipefail

# One-talk medicine SimulEval readout using strict MFA-only medicine terms.
# The strict medicine source glossary currently has no target translations for
# expansion/filler terms, so this launcher evaluates translated strict GT terms
# for the selected ESO talk.  It is meant as the fast medicine smoke/readout
# before broader translated-filler glossary work.

ROOT_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST"
EVAL_SCRIPT="${ROOT_DIR}/documents/code/simuleval/eval_density_unified.sh"
PREP_SCRIPT="${ROOT_DIR}/documents/code/simuleval/prepare_medicine_one_talk_inputs.py"

MODEL_NAME="${MODEL_NAME_OVERRIDE:?Set MODEL_NAME_OVERRIDE to the HF speech LLM path}"
TARGET_SAMPLE="${TARGET_SAMPLE_OVERRIDE:-404}"
TARGET_LM="${TARGET_LM:?Set TARGET_LM, e.g. 2}"
LANG_CODE="${LANG_CODE_OVERRIDE:-zh}"
MAX_SENTENCES_OVERRIDE="${MAX_SENTENCES_OVERRIDE:-0}"
OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/medicine_onetalk_lh1b88kw_tau073}"
MEDICINE_INPUTS="${OUTPUT_BASE}/${LANG_CODE}/__medicine_inputs__/lists"

if [[ -n "${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV:-}" ]]; then
    CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV//:/,}"
fi

RAG_MODEL_PATH="${RAG_MODEL_PATH_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
RAG_TOP_K="${RAG_TOP_K_OVERRIDE:-10}"
RAG_LORA_R="${RAG_LORA_R_OVERRIDE:-128}"
RAG_TEXT_LORA_R="${RAG_TEXT_LORA_R_OVERRIDE:-128}"
RAG_SCORE_THRESHOLD="${RAG_SCORE_THRESHOLD_OVERRIDE:-0.73}"
DENSITY_TAG="${DENSITY_TAG_OVERRIDE:-medicine1_lh1b88kw}"

STRICT_JSONL="${STRICT_MEDICINE_JSONL_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_dev_dataset.jsonl}"
STRICT_GLOSSARY="${STRICT_MEDICINE_GLOSSARY_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_glossary_gt_plus_medicine_wiki_gs10000_translated.json}"
ESO_TEST_ROOT="${ESO_TEST_ROOT_OVERRIDE:-/home/jiaxingxu/rag-sst/eso-dataset/outputs_v2/test}"

clean_shm() {
    local me
    me="$(id -un)"
    find /dev/shm -maxdepth 1 -user "${me}" -mmin +1 \
        \( -name 'psm_*' -o -name 'loky-*' -o -name 'torch_*' -o -name 'vllm_*' \) \
        -delete 2>/dev/null || true
    echo "[INFO] /dev/shm: $(df -h /dev/shm | tail -n1) ; user-owned entries: $(find /dev/shm -maxdepth 1 -user "${me}" 2>/dev/null | wc -l)"
}

prepare_medicine_inputs() {
    mkdir -p "${MEDICINE_INPUTS}"
    python3 "${PREP_SCRIPT}" \
        --sample-id "${TARGET_SAMPLE}" \
        --lang-code "${LANG_CODE}" \
        --eso-test-root "${ESO_TEST_ROOT}" \
        --strict-jsonl "${STRICT_JSONL}" \
        --strict-glossary "${STRICT_GLOSSARY}" \
        --output-dir "${MEDICINE_INPUTS}" \
        --max-sentences "${MAX_SENTENCES_OVERRIDE}"
}

run_medicine() {
    local prefix="medicine_${TARGET_SAMPLE}"
    local glossary_path="${MEDICINE_INPUTS}/medicine_gt_strict_translated__${prefix}.json"
    local glossary_tag="medicine_gt_strict_translated__${prefix}"
    local src_list="${MEDICINE_INPUTS}/medicine.source__${prefix}.txt"
    local tgt_list="${MEDICINE_INPUTS}/medicine.target.${LANG_CODE}__${prefix}.txt"
    local source_text_file="${MEDICINE_INPUTS}/medicine.source_text.en__${prefix}.txt"
    local ref_file="${MEDICINE_INPUTS}/medicine.ref.${LANG_CODE}__${prefix}.txt"
    local audio_yaml="${MEDICINE_INPUTS}/medicine.audio__${prefix}.yaml"
    local output_dir="${OUTPUT_BASE}/${LANG_CODE}/d${DENSITY_TAG}_lm${TARGET_LM}_k${RAG_TOP_K}_th${RAG_SCORE_THRESHOLD}_g${glossary_tag}_sample${TARGET_SAMPLE}"
    local instances_log="${output_dir}/instances.log"
    local eval_tsv="${output_dir}/eval_results.tsv"

    if [[ -f "${instances_log}" ]] && [[ -s "${instances_log}" ]] \
       && [[ -f "${eval_tsv}" ]] && [[ -s "${eval_tsv}" ]] \
       && ! grep -q $'\tN/A\tN/A\tN/A' "${eval_tsv}"; then
        echo "[SKIP] medicine sample=${TARGET_SAMPLE} lm=${TARGET_LM} tau=${RAG_SCORE_THRESHOLD} already complete"
        return 0
    fi

    echo "[RUN] medicine sample=${TARGET_SAMPLE} lm=${TARGET_LM} tau=${RAG_SCORE_THRESHOLD}"
    clean_shm

    INDEX_PATH_OVERRIDE="" \
    SKIP_OFFLINE_EVAL="0" \
    EVAL_MODE_OVERRIDE="acl6060" \
    GLOSSARY_PATH_OVERRIDE="${glossary_path}" \
    LANG_CODE_OVERRIDE="${LANG_CODE}" \
    SRC_LIST_OVERRIDE="${src_list}" \
    TGT_LIST_OVERRIDE="${tgt_list}" \
    REF_FILE_OVERRIDE="${ref_file}" \
    SOURCE_TEXT_FILE_OVERRIDE="${source_text_file}" \
    AUDIO_YAML_OVERRIDE="${audio_yaml}" \
    MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
    RAG_MODEL_PATH_OVERRIDE="${RAG_MODEL_PATH}" \
    RAG_LORA_R_OVERRIDE="${RAG_LORA_R}" \
    RAG_TEXT_LORA_R_OVERRIDE="${RAG_TEXT_LORA_R}" \
    RAG_SCORE_THRESHOLD_OVERRIDE="${RAG_SCORE_THRESHOLD}" \
    OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
    CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE:-${CUDA_VISIBLE_DEVICES:-0,1,2}}" \
    LATENCY_MULTIPLIER_OVERRIDE="${TARGET_LM}" \
    RAG_TOP_K_OVERRIDE="${RAG_TOP_K}" \
    GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}" \
    DENSITY_TAG="${DENSITY_TAG}" \
    PAPER_ID_TAG="medicine_${TARGET_SAMPLE}" \
    env -u MAX_CACHE_SECONDS_OVERRIDE \
        -u KEEP_CACHE_SECONDS_OVERRIDE \
        -u VLLM_MAX_MODEL_LEN_OVERRIDE \
        -u VLLM_LIMIT_AUDIO_OVERRIDE \
        -u VLLM_ENABLE_PREFIX_CACHING \
        bash "${EVAL_SCRIPT}"

    echo "[DONE] ${output_dir}"
    clean_shm
}

prepare_medicine_inputs
run_medicine

echo "[ALL DONE] One-talk medicine evaluation complete (sample=${TARGET_SAMPLE}, lm=${TARGET_LM})."
