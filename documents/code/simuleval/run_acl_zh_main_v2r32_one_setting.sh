#!/usr/bin/env bash
#SBATCH --job-name=aclmain_v2r32
#SBATCH --partition=aries
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_aclmain_v2r32.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_aclmain_v2r32.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
EVAL_SCRIPT="${ROOT_DIR}/documents/code/simuleval/eval_density_unified.sh"
AGG_SCRIPT="${ROOT_DIR}/documents/code/simuleval/aggregate_acl_main_sweep_results.py"

MODEL_NAME="${MODEL_NAME_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/speech_llm_tcm_filtered_wiki100kgt_tau075_v2_sourcefinal_gtzh/keep1.0_r32/v0-20260507-103419-hf}"
RAG_MODEL_PATH="${RAG_MODEL_PATH_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_ntcm_final_v3_n64_p1n4_s2000_aries-8gpu_smallest_dense_smoke4650_best.pt}"
OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/acl_main_zh_v2_r32_srcgated_no_utterance}"
PAPER_INPUTS="${OUTPUT_BASE}/zh/__paper_inputs__/lists"

TARGET_PAPER="${TARGET_PAPER:?Set TARGET_PAPER, e.g. 2022.acl-long.110}"
TARGET_LM="${TARGET_LM:?Set TARGET_LM, e.g. 1}"
GLOSSARY_KIND="${GLOSSARY_KIND:?Set GLOSSARY_KIND to raw|gs1k|gs10k}"
RAG_SCORE_THRESHOLD="${RAG_SCORE_THRESHOLD_OVERRIDE:?Set RAG_SCORE_THRESHOLD_OVERRIDE to 0.0 or 0.75}"
MAX_SENTENCES_OVERRIDE="${MAX_SENTENCES_OVERRIDE:-0}"

RAG_TOP_K="${RAG_TOP_K_OVERRIDE:-10}"
RAG_LORA_R="${RAG_LORA_R_OVERRIDE:-128}"
RAG_TEXT_LORA_R="${RAG_TEXT_LORA_R_OVERRIDE:-128}"
RAG_STREAMING_MODE="${RAG_STREAMING_MODE_OVERRIDE:-timeline}"
RAG_MAXSIM_WINDOWS="${RAG_MAXSIM_WINDOWS_OVERRIDE:-2 3 4 5 6 7 8 10 12 16 20 24}"
RAG_MAXSIM_STRIDE="${RAG_MAXSIM_STRIDE_OVERRIDE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}"
DENSITY_TAG="${DENSITY_TAG_OVERRIDE:-aclmain_v2r32}"
CLEAN_SHM_OVERRIDE="${CLEAN_SHM_OVERRIDE:-0}"
CLEAN_OUTPUT_DIR_OVERRIDE="${CLEAN_OUTPUT_DIR_OVERRIDE:-0}"

if [[ -n "${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV:-}" ]]; then
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV//:/,}"
fi
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE:-0,1,2}"

DATA_ROOT="/mnt/taurus/data/siqiouyang/datasets/acl6060"
DEV_SOURCE="${DATA_ROOT}/dev.source"
DEV_TARGET="${DATA_ROOT}/dev.target.zh"
DEV_SOURCE_TEXT="${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.en.txt"
DEV_REF="${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.zh.txt"
DEV_AUDIO_YAML="${DATA_ROOT}/dev.yaml"

GLOSSARY_DEFAULT_DIR="${ROOT_DIR}/documents/data/data_pre/extracted_glossaries_by_paper"
GLOSSARY_GS1K="${ROOT_DIR}/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs1000.json"
GLOSSARY_GS10K="${ROOT_DIR}/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs10000.json"

case "${GLOSSARY_KIND}" in
  raw)
    GLOSSARY_PATH="${GLOSSARY_DEFAULT_DIR}/extracted_glossary__${TARGET_PAPER}.json"
    GLOSSARY_TAG="extracted_glossary__${TARGET_PAPER}"
    ;;
  gs1k)
    GLOSSARY_PATH="${GLOSSARY_GS1K}"
    GLOSSARY_TAG="glossary_acl6060_gt_union_gs1000"
    ;;
  gs10k)
    GLOSSARY_PATH="${GLOSSARY_GS10K}"
    GLOSSARY_TAG="glossary_acl6060_gt_union_gs10000"
    ;;
  *)
    echo "[ERROR] Unsupported GLOSSARY_KIND=${GLOSSARY_KIND}; expected raw|gs1k|gs10k" >&2
    exit 2
    ;;
esac

for p in "${EVAL_SCRIPT}" "${MODEL_NAME}" "${RAG_MODEL_PATH}" "${GLOSSARY_PATH}" "${DEV_SOURCE}" "${DEV_TARGET}" "${DEV_SOURCE_TEXT}" "${DEV_REF}" "${DEV_AUDIO_YAML}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

clean_shm() {
  if [[ "${CLEAN_SHM_OVERRIDE}" != "1" ]]; then
    echo "[INFO] Skipping /dev/shm cleanup (CLEAN_SHM_OVERRIDE=${CLEAN_SHM_OVERRIDE})."
    return 0
  fi
  python3 - <<'PY' || true
import os
import time

prefixes = ("psm_", "loky-", "torch_", "vllm_")
cutoff = time.time() - 60
try:
    with os.scandir("/dev/shm") as it:
        for entry in it:
            if not entry.name.startswith(prefixes):
                continue
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if st.st_uid == os.getuid() and st.st_mtime < cutoff:
                try:
                    os.unlink(entry.path)
                except OSError:
                    pass
except OSError:
    pass
PY
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

prepare_paper_inputs

SRC_LIST="${PAPER_INPUTS}/dev.source__${TARGET_PAPER}.txt"
TGT_LIST="${PAPER_INPUTS}/dev.target.zh__${TARGET_PAPER}.txt"
SOURCE_TEXT_FILE="${PAPER_INPUTS}/dev.source_text.en__${TARGET_PAPER}.txt"
REF_FILE="${PAPER_INPUTS}/dev.ref.zh__${TARGET_PAPER}.txt"
AUDIO_YAML="${PAPER_INPUTS}/audio__${TARGET_PAPER}.yaml"
OUTPUT_DIR="${OUTPUT_BASE}/zh/d${DENSITY_TAG}_lm${TARGET_LM}_k${RAG_TOP_K}_th${RAG_SCORE_THRESHOLD}_g${GLOSSARY_TAG}_pp${TARGET_PAPER}"
EVAL_TSV="${OUTPUT_DIR}/eval_results.tsv"

echo "[INFO] TASK paper=${TARGET_PAPER} lm=${TARGET_LM} glossary=${GLOSSARY_KIND}/${GLOSSARY_TAG} tau=${RAG_SCORE_THRESHOLD}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] CUDA_VISIBLE_DEVICES_PHYSICAL=${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE}"

if [[ -f "${EVAL_TSV}" ]] && [[ -s "${EVAL_TSV}" ]] && ! grep -q $'\tN/A\tN/A\tN/A' "${EVAL_TSV}"; then
  echo "[SKIP] eval_results.tsv already complete: ${EVAL_TSV}"
else
  clean_shm
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
  OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE}" \
  LATENCY_MULTIPLIER_OVERRIDE="${TARGET_LM}" \
  RAG_TOP_K_OVERRIDE="${RAG_TOP_K}" \
  RAG_SCORE_THRESHOLD_OVERRIDE="${RAG_SCORE_THRESHOLD}" \
  RAG_STREAMING_MODE_OVERRIDE="${RAG_STREAMING_MODE}" \
  RAG_MAXSIM_WINDOWS_OVERRIDE="${RAG_MAXSIM_WINDOWS}" \
  RAG_MAXSIM_STRIDE_OVERRIDE="${RAG_MAXSIM_STRIDE}" \
  GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION}" \
  DENSITY_TAG="${DENSITY_TAG}" \
  PAPER_ID_TAG="${TARGET_PAPER}" \
  CLEAN_OUTPUT_DIR_OVERRIDE="${CLEAN_OUTPUT_DIR_OVERRIDE}" \
  VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
  env -u MAX_CACHE_SECONDS_OVERRIDE \
      -u KEEP_CACHE_SECONDS_OVERRIDE \
      -u VLLM_MAX_MODEL_LEN_OVERRIDE \
      -u VLLM_LIMIT_AUDIO_OVERRIDE \
      -u VLLM_ENABLE_PREFIX_CACHING \
      bash "${EVAL_SCRIPT}"
  clean_shm
fi

if [[ -f "${AGG_SCRIPT}" ]]; then
  python3 "${AGG_SCRIPT}" --base-dir "${OUTPUT_BASE}" || true
fi

echo "[ALL DONE] paper=${TARGET_PAPER} lm=${TARGET_LM} glossary=${GLOSSARY_KIND} tau=${RAG_SCORE_THRESHOLD}"
