#!/usr/bin/env bash
#SBATCH --job-name=aclpp_lm_sner_2w
#SBATCH --partition=aries
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_aclpp_lm_sner_2w.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_aclpp_lm_sner_2w.err

set -euo pipefail

# Aries ACL per-paper LM=1..4 eval. Runs two 3-GPU workers in parallel:
# worker 0 uses GPUs 0,1,2; worker 1 uses GPUs 3,4,5. GPUs 6,7 remain unused.

ROOT_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST"
EVAL_SCRIPT="${ROOT_DIR}/documents/code/simuleval/eval_density_unified.sh"
AGG_SCRIPT="${ROOT_DIR}/documents/code/simuleval/aggregate_acl_perpaper_lm2_results.py"

MODEL_NAME="${MODEL_NAME_OVERRIDE:-/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4}"
RAG_MODEL_PATH="${RAG_MODEL_PATH_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_ntcm_final_v3_n64_p1n4_s2000_aries-8gpu_smallest_dense_smoke4650_best.pt}"
OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/acl_perpaper_lm1to4_raw1k10k_sner_tcmrag_aries2w}"
PAPER_INPUTS="${OUTPUT_BASE}/zh/__paper_inputs__/lists"

RAG_TOP_K="10"
RAG_LORA_R="${RAG_LORA_R_OVERRIDE:-128}"
RAG_TEXT_LORA_R="${RAG_TEXT_LORA_R_OVERRIDE:-128}"
DENSITY_TAG="aclpp_sner_tcmrag"

DATA_ROOT="/mnt/taurus/data/siqiouyang/datasets/acl6060"
DEV_SOURCE="${DATA_ROOT}/dev.source"
DEV_TARGET="${DATA_ROOT}/dev.target.zh"
DEV_SOURCE_TEXT="${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.en.txt"
DEV_REF="${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.zh.txt"
DEV_AUDIO_YAML="${DATA_ROOT}/dev.yaml"

LATENCY_MULTIPLIERS=(1 2 3 4)
PAPERS=("2022.acl-long.110" "2022.acl-long.117" "2022.acl-long.268" "2022.acl-long.367" "2022.acl-long.590")

GLOSSARY_DEFAULT_DIR="${ROOT_DIR}/documents/data/data_pre/extracted_glossaries_by_paper"
GLOSSARY_GS1K="${ROOT_DIR}/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs1000.json"
GLOSSARY_GS10K="${ROOT_DIR}/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs10000.json"

prepare_paper_inputs() {
    mkdir -p "${PAPER_INPUTS}"

    PAPERS_CSV="$(IFS=,; echo "${PAPERS[*]}")" \
    PAPER_INPUTS="${PAPER_INPUTS}" \
    DEV_SOURCE="${DEV_SOURCE}" \
    DEV_TARGET="${DEV_TARGET}" \
    DEV_SOURCE_TEXT="${DEV_SOURCE_TEXT}" \
    DEV_REF="${DEV_REF}" \
    DEV_AUDIO_YAML="${DEV_AUDIO_YAML}" \
    python3 - <<'PY'
import os
from pathlib import Path

import yaml

papers = [p for p in os.environ["PAPERS_CSV"].split(",") if p]
out_dir = Path(os.environ["PAPER_INPUTS"])
dev_source = Path(os.environ["DEV_SOURCE"])
dev_target = Path(os.environ["DEV_TARGET"])
dev_source_text = Path(os.environ["DEV_SOURCE_TEXT"])
dev_ref = Path(os.environ["DEV_REF"])
dev_audio_yaml = Path(os.environ["DEV_AUDIO_YAML"])

source_lines = dev_source.read_text(encoding="utf-8").splitlines()
target_lines = dev_target.read_text(encoding="utf-8").splitlines()
if len(source_lines) != len(target_lines):
    raise SystemExit(f"dev.source lines {len(source_lines)} != dev.target lines {len(target_lines)}")

source_by_paper = {}
target_by_paper = {}
for src, tgt in zip(source_lines, target_lines):
    paper_id = Path(src.strip()).stem
    source_by_paper[paper_id] = src.replace("/mnt/data/siqiouyang", "/mnt/taurus/data/siqiouyang")
    target_by_paper[paper_id] = tgt

audio_entries = yaml.safe_load(dev_audio_yaml.read_text(encoding="utf-8"))
if not isinstance(audio_entries, list):
    raise SystemExit(f"Invalid audio yaml format: {dev_audio_yaml}")
source_text_lines = dev_source_text.read_text(encoding="utf-8").splitlines()
ref_lines = dev_ref.read_text(encoding="utf-8").splitlines()
if len(audio_entries) != len(source_text_lines) or len(audio_entries) != len(ref_lines):
    raise SystemExit(
        f"alignment mismatch: yaml={len(audio_entries)} source_text={len(source_text_lines)} ref={len(ref_lines)}"
    )

for paper_id in papers:
    if paper_id not in source_by_paper:
        raise SystemExit(f"Missing paper in dev.source/dev.target: {paper_id}")

    indices = [
        i for i, item in enumerate(audio_entries)
        if isinstance(item, dict) and Path(str(item.get("wav", ""))).stem == paper_id
    ]
    if not indices:
        raise SystemExit(f"No dev.yaml entries for paper: {paper_id}")

    (out_dir / f"dev.source__{paper_id}.txt").write_text(source_by_paper[paper_id] + "\n", encoding="utf-8")
    (out_dir / f"dev.target.zh__{paper_id}.txt").write_text(target_by_paper[paper_id] + "\n", encoding="utf-8")
    (out_dir / f"dev.source_text.en__{paper_id}.txt").write_text(
        "\n".join(source_text_lines[i] for i in indices) + "\n", encoding="utf-8"
    )
    (out_dir / f"dev.ref.zh__{paper_id}.txt").write_text(
        "\n".join(ref_lines[i] for i in indices) + "\n", encoding="utf-8"
    )
    (out_dir / f"audio__{paper_id}.yaml").write_text(
        yaml.safe_dump([audio_entries[i] for i in indices], allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"[INPUT] {paper_id}: {len(indices)} sentence-level rows")
PY
}

run_one() {
    local LM="$1" GLOSSARY_PATH="$2" GLOSSARY_TAG="$3" PAPER_ID="$4" GPUS="$5"

    local SRC_LIST="${PAPER_INPUTS}/dev.source__${PAPER_ID}.txt"
    local TGT_LIST="${PAPER_INPUTS}/dev.target.zh__${PAPER_ID}.txt"
    local SOURCE_TEXT_FILE="${PAPER_INPUTS}/dev.source_text.en__${PAPER_ID}.txt"
    local REF_FILE="${PAPER_INPUTS}/dev.ref.zh__${PAPER_ID}.txt"
    local AUDIO_YAML="${PAPER_INPUTS}/audio__${PAPER_ID}.yaml"
    local OUTPUT_DIR="${OUTPUT_BASE}/zh/d${DENSITY_TAG}_lm${LM}_k${RAG_TOP_K}_th0.0_g${GLOSSARY_TAG}_pp${PAPER_ID}"
    local INSTANCES_LOG="${OUTPUT_DIR}/instances.log"
    local EVAL_TSV="${OUTPUT_DIR}/eval_results.tsv"

    if [[ -f "${INSTANCES_LOG}" ]] && [[ -s "${INSTANCES_LOG}" ]] && [[ -f "${EVAL_TSV}" ]] && [[ -s "${EVAL_TSV}" ]] && ! grep -q $'\tN/A\tN/A\tN/A' "${EVAL_TSV}"; then
        echo "[SKIP][gpus=${GPUS}] lm=${LM} gs=${GLOSSARY_TAG} paper=${PAPER_ID}"
        return 0
    fi

    echo "[RUN][gpus=${GPUS}] lm=${LM} gs=${GLOSSARY_TAG} paper=${PAPER_ID}"

    INDEX_PATH_OVERRIDE="" \
    SKIP_OFFLINE_EVAL="0" \
    EVAL_MODE_OVERRIDE="acl6060" \
    GLOSSARY_PATH_OVERRIDE="${GLOSSARY_PATH}" \
    SRC_LIST_OVERRIDE="${SRC_LIST}" \
    TGT_LIST_OVERRIDE="${TGT_LIST}" \
    REF_FILE_OVERRIDE="${REF_FILE}" \
    SOURCE_TEXT_FILE_OVERRIDE="${SOURCE_TEXT_FILE}" \
    AUDIO_YAML_OVERRIDE="${AUDIO_YAML}" \
    MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
    RAG_MODEL_PATH_OVERRIDE="${RAG_MODEL_PATH}" \
    RAG_LORA_R_OVERRIDE="${RAG_LORA_R}" \
    RAG_TEXT_LORA_R_OVERRIDE="${RAG_TEXT_LORA_R}" \
    GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}" \
    OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
    CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${GPUS}" \
    LATENCY_MULTIPLIER_OVERRIDE="${LM}" \
    RAG_TOP_K_OVERRIDE="${RAG_TOP_K}" \
    DENSITY_TAG="${DENSITY_TAG}" \
    PAPER_ID_TAG="${PAPER_ID}" \
    bash "${EVAL_SCRIPT}"

    echo "[DONE][gpus=${GPUS}] lm=${LM} gs=${GLOSSARY_TAG} paper=${PAPER_ID}"
}

write_tasks() {
    local task_file="$1"
    : > "${task_file}"
    for LM in "${LATENCY_MULTIPLIERS[@]}"; do
        for PAPER_ID in "${PAPERS[@]}"; do
            printf "%s\t%s\t%s\t%s\n" "${LM}" "${GLOSSARY_DEFAULT_DIR}/extracted_glossary__${PAPER_ID}.json" "extracted_glossary__${PAPER_ID}" "${PAPER_ID}" >> "${task_file}"
            printf "%s\t%s\t%s\t%s\n" "${LM}" "${GLOSSARY_GS1K}" "glossary_acl6060_gt_union_gs1000" "${PAPER_ID}" >> "${task_file}"
            printf "%s\t%s\t%s\t%s\n" "${LM}" "${GLOSSARY_GS10K}" "glossary_acl6060_gt_union_gs10000" "${PAPER_ID}" >> "${task_file}"
        done
    done
}

run_worker() {
    local worker_id="$1" worker_count="$2" gpus="$3" task_file="$4"
    local line_no=0
    local lm glossary_path glossary_tag paper_id

    while IFS=$'\t' read -r lm glossary_path glossary_tag paper_id; do
        if (( line_no % worker_count == worker_id )); then
            run_one "${lm}" "${glossary_path}" "${glossary_tag}" "${paper_id}" "${gpus}"
        fi
        line_no=$((line_no + 1))
    done < "${task_file}"
}

prepare_paper_inputs

TASK_FILE="${OUTPUT_BASE}/zh/__paper_inputs__/lm1to4_tasks.tsv"
mkdir -p "$(dirname "${TASK_FILE}")"
write_tasks "${TASK_FILE}"
echo "[INFO] Task file: ${TASK_FILE} ($(wc -l < "${TASK_FILE}") tasks)"

run_worker 0 2 "0,1,2" "${TASK_FILE}" > "${OUTPUT_BASE}/zh/worker0_gpus012.log" 2>&1 &
pid0=$!
run_worker 1 2 "3,4,5" "${TASK_FILE}" > "${OUTPUT_BASE}/zh/worker1_gpus345.log" 2>&1 &
pid1=$!

cleanup_workers() {
    kill "${pid0:-}" "${pid1:-}" 2>/dev/null || true
}
trap cleanup_workers INT TERM

set +e
wait "${pid0}"
rc0=$?
wait "${pid1}"
rc1=$?
set -e

if [[ "${rc0}" -ne 0 || "${rc1}" -ne 0 ]]; then
    echo "[ERROR] Worker failure: worker0=${rc0} worker1=${rc1}" >&2
    exit 1
fi

trap - INT TERM

python3 "${AGG_SCRIPT}" --base-dir "${OUTPUT_BASE}"

echo "[ALL DONE] Aries ACL per-paper LM=1..4 evaluation complete."
