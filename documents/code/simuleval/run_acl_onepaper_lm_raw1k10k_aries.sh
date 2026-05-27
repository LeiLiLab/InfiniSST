#!/usr/bin/env bash
#SBATCH --job-name=acl_onepp_aries
#SBATCH --partition=aries
#SBATCH --exclusive
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_acl_onepp_aries.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_acl_onepp_aries.err

set -euo pipefail

# Aries variant of run_acl_onepaper_lm_raw1k10k_taurus.sh.
#
# Why this exists:
#   On aries the partition is one node with 8 A6000 GPUs and 504 GB /dev/shm,
#   but Slurm allowed up to two of these jobs (gpu=3, mem=256G) to run on the
#   same node simultaneously. Two concurrently-starting vLLM v1 engines
#   clashed on `/dev/shm` MessageQueue allocation and crashed during
#   `get_kv_cache_specs` with
#     AttributeError: 'ShmRingBuffer' object has no attribute 'shared_memory'
#   Only the single concurrent job that started while the partner was still
#   in init succeeded (44323 on 2026-05-06).
#
# Mitigations applied here:
#   * --exclusive : one job per node at a time (prevents concurrent shm fight)
#   * pre-launch /dev/shm cleanup of stale psm_*/loky-* from prior crashed runs
#   * unchanged vLLM args (TP=2, gpu_mem_util=0.72, prefix-cache=1, max_len=32768)
#     so resulting eval_results.tsv stays comparable to the 33 already-completed
#     entries in the same OUTPUT_BASE.
#
# Required env (set on submit):
#   MODEL_NAME_OVERRIDE  - HF speech LLM dir (must exist on aries-readable mount)
#   TARGET_PAPER         - e.g. 2022.acl-long.117
#   TARGET_LM            - e.g. 1
# Optional:
#   OUTPUT_BASE_OVERRIDE - default points at the aries2w sweep root; use a
#                          tau-aware path when running with RAG filtering so
#                          tau=0.0 and tau=0.75 results don't co-mingle.
#   RAG_MODEL_PATH_OVERRIDE / RAG_LORA_R_OVERRIDE / RAG_TEXT_LORA_R_OVERRIDE
#   RAG_SCORE_THRESHOLD_OVERRIDE - default 0.0 (no filtering); set e.g. 0.75 for
#                                  inference-time score filtering. The chosen
#                                  value is embedded in the per-run output dir
#                                  as `th${tau}` via eval_density_unified.sh.
#   GPU_MEMORY_UTILIZATION_OVERRIDE
#   DENSITY_TAG_OVERRIDE - default aclpp_sner_tcmrag (matches existing TSV rows)
#   MAX_SENTENCES_OVERRIDE

ROOT_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST"
EVAL_SCRIPT="${ROOT_DIR}/documents/code/simuleval/eval_density_unified.sh"

MODEL_NAME="${MODEL_NAME_OVERRIDE:?Set MODEL_NAME_OVERRIDE to the HF speech LLM path}"
TARGET_PAPER="${TARGET_PAPER:?Set TARGET_PAPER, e.g. 2022.acl-long.110}"
TARGET_LM="${TARGET_LM:?Set TARGET_LM, e.g. 2}"
MAX_SENTENCES_OVERRIDE="${MAX_SENTENCES_OVERRIDE:-0}"
OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/acl_perpaper_lm1to4_raw1k10k_sner_tcmrag_aries2w}"
PAPER_INPUTS="${OUTPUT_BASE}/zh/__paper_inputs__/lists"

if [[ -n "${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV:-}" ]]; then
    CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV//:/,}"
fi

RAG_MODEL_PATH="${RAG_MODEL_PATH_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_ntcm_final_v3_n64_p1n4_s2000_aries-8gpu_smallest_dense_smoke4650_best.pt}"
RAG_TOP_K="${RAG_TOP_K_OVERRIDE:-10}"
RAG_LORA_R="${RAG_LORA_R_OVERRIDE:-128}"
RAG_TEXT_LORA_R="${RAG_TEXT_LORA_R_OVERRIDE:-128}"
RAG_SCORE_THRESHOLD="${RAG_SCORE_THRESHOLD_OVERRIDE:-0.0}"
DENSITY_TAG="${DENSITY_TAG_OVERRIDE:-aclpp_sner_tcmrag}"

DATA_ROOT="/mnt/taurus/data/siqiouyang/datasets/acl6060"
DEV_SOURCE="${DATA_ROOT}/dev.source"
DEV_TARGET="${DATA_ROOT}/dev.target.zh"
DEV_SOURCE_TEXT="${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.en.txt"
DEV_REF="${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.zh.txt"
DEV_AUDIO_YAML="${DATA_ROOT}/dev.yaml"

GLOSSARY_DEFAULT_DIR="${ROOT_DIR}/documents/data/data_pre/extracted_glossaries_by_paper"
GLOSSARY_GS1K="${ROOT_DIR}/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs1000.json"
GLOSSARY_GS10K="${ROOT_DIR}/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs10000.json"

# ---- /dev/shm hygiene -------------------------------------------------------
# Remove stale psm_*/loky-* entries owned by this user that are >1 minute old.
# These are leftovers from prior crashed vLLM/loky workers and have been
# observed to make fresh ShmRingBuffer creation fail with
# `AttributeError: 'ShmRingBuffer' object has no attribute 'shared_memory'`.
clean_shm() {
    local me
    me="$(id -un)"
    find /dev/shm -maxdepth 1 -user "${me}" -mmin +1 \
        \( -name 'psm_*' -o -name 'loky-*' -o -name 'torch_*' -o -name 'vllm_*' \) \
        -delete 2>/dev/null || true
    echo "[INFO] /dev/shm: $(df -h /dev/shm | tail -n1) ; user-owned entries: $(find /dev/shm -maxdepth 1 -user "${me}" 2>/dev/null | wc -l)"
}

prepare_paper_inputs() {
    mkdir -p "${PAPER_INPUTS}"

    TARGET_PAPER="${TARGET_PAPER}" \
    MAX_SENTENCES_OVERRIDE="${MAX_SENTENCES_OVERRIDE}" \
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

paper_id = os.environ["TARGET_PAPER"]
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
    pid = Path(src.strip()).stem
    source_by_paper[pid] = src.replace("/mnt/data/siqiouyang", "/mnt/taurus/data/siqiouyang")
    target_by_paper[pid] = tgt
if paper_id not in source_by_paper:
    raise SystemExit(f"Missing paper in dev.source/dev.target: {paper_id}")

audio_entries = yaml.safe_load(dev_audio_yaml.read_text(encoding="utf-8"))
if not isinstance(audio_entries, list):
    raise SystemExit(f"Invalid audio yaml format: {dev_audio_yaml}")
source_text_lines = dev_source_text.read_text(encoding="utf-8").splitlines()
ref_lines = dev_ref.read_text(encoding="utf-8").splitlines()
if len(audio_entries) != len(source_text_lines) or len(audio_entries) != len(ref_lines):
    raise SystemExit(
        f"alignment mismatch: yaml={len(audio_entries)} source_text={len(source_text_lines)} ref={len(ref_lines)}"
    )

indices = [
    i for i, item in enumerate(audio_entries)
    if isinstance(item, dict) and Path(str(item.get("wav", ""))).stem == paper_id
]
if not indices:
    raise SystemExit(f"No dev.yaml entries for paper: {paper_id}")
max_sentences = int(os.environ.get("MAX_SENTENCES_OVERRIDE", "0") or "0")
if max_sentences > 0:
    indices = indices[:max_sentences]

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
    local glossary_path="$1" glossary_tag="$2"

    local src_list="${PAPER_INPUTS}/dev.source__${TARGET_PAPER}.txt"
    local tgt_list="${PAPER_INPUTS}/dev.target.zh__${TARGET_PAPER}.txt"
    local source_text_file="${PAPER_INPUTS}/dev.source_text.en__${TARGET_PAPER}.txt"
    local ref_file="${PAPER_INPUTS}/dev.ref.zh__${TARGET_PAPER}.txt"
    local audio_yaml="${PAPER_INPUTS}/audio__${TARGET_PAPER}.yaml"
    local output_dir="${OUTPUT_BASE}/zh/d${DENSITY_TAG}_lm${TARGET_LM}_k${RAG_TOP_K}_th${RAG_SCORE_THRESHOLD}_g${glossary_tag}_pp${TARGET_PAPER}"
    local instances_log="${output_dir}/instances.log"
    local eval_tsv="${output_dir}/eval_results.tsv"

    if [[ -f "${instances_log}" ]] && [[ -s "${instances_log}" ]] \
       && [[ -f "${eval_tsv}" ]] && [[ -s "${eval_tsv}" ]] \
       && ! grep -q $'\tN/A\tN/A\tN/A' "${eval_tsv}"; then
        echo "[SKIP] lm=${TARGET_LM} gs=${glossary_tag} paper=${TARGET_PAPER} tau=${RAG_SCORE_THRESHOLD} (eval_results.tsv already complete)"
        return 0
    fi

    echo "[RUN] lm=${TARGET_LM} gs=${glossary_tag} paper=${TARGET_PAPER} tau=${RAG_SCORE_THRESHOLD}"
    clean_shm

    INDEX_PATH_OVERRIDE="" \
    SKIP_OFFLINE_EVAL="0" \
    EVAL_MODE_OVERRIDE="acl6060" \
    GLOSSARY_PATH_OVERRIDE="${glossary_path}" \
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
    PAPER_ID_TAG="${TARGET_PAPER}" \
    env -u MAX_CACHE_SECONDS_OVERRIDE \
        -u KEEP_CACHE_SECONDS_OVERRIDE \
        -u VLLM_MAX_MODEL_LEN_OVERRIDE \
        -u VLLM_LIMIT_AUDIO_OVERRIDE \
        -u VLLM_ENABLE_PREFIX_CACHING \
        bash "${EVAL_SCRIPT}"

    echo "[DONE] ${output_dir}"
    clean_shm
}

prepare_paper_inputs

run_one "${GLOSSARY_DEFAULT_DIR}/extracted_glossary__${TARGET_PAPER}.json" "extracted_glossary__${TARGET_PAPER}"
run_one "${GLOSSARY_GS1K}" "glossary_acl6060_gt_union_gs1000"
run_one "${GLOSSARY_GS10K}" "glossary_acl6060_gt_union_gs10000"

python3 "${ROOT_DIR}/documents/code/simuleval/aggregate_acl_perpaper_lm2_results.py" --base-dir "${OUTPUT_BASE}" || true

echo "[ALL DONE] One-paper ACL evaluation complete (paper=${TARGET_PAPER}, lm=${TARGET_LM})."
