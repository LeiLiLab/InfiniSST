#!/usr/bin/env bash
#SBATCH --job-name=acl_v2_rank
#SBATCH --partition=taurus
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_acl_v2_rank.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_acl_v2_rank.err

set -euo pipefail

ROOT_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST"
ONEPAPER="${ROOT_DIR}/documents/code/simuleval/run_acl_onepaper_lm_raw1k10k_taurus.sh"
RANK="${LORA_RANK_OVERRIDE:?Set LORA_RANK_OVERRIDE to 64 or 128}"
TAU="${RAG_SCORE_THRESHOLD_OVERRIDE:?Set RAG_SCORE_THRESHOLD_OVERRIDE to 0.0 or 0.75}"

if [[ -n "${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV:-}" ]]; then
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV//:/,}"
fi
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE:-0,1,2}"

case "${TAU}" in
  0|0.0) TAU_TAG="tau0"; TAU_VALUE="0.0" ;;
  0.75) TAU_TAG="tau075"; TAU_VALUE="0.75" ;;
  *) echo "[ERROR] Unsupported tau: ${TAU}" >&2; exit 2 ;;
esac

SAVE_BASE="/mnt/gemini/data2/jiaxuanluo/speech_llm_tcm_filtered_wiki100kgt_tau075_v2_sourcefinal_gtzh_rank_sweep"
HF_MODEL="$(python3 - "${SAVE_BASE}" "${RANK}" <<'PY'
import sys
from pathlib import Path

base = Path(sys.argv[1])
rank = sys.argv[2]
roots = [base / f"keep1.0_r{rank}", base / f"keepdirect_r{rank}"]
candidates = sorted(
    [p for root in roots for p in root.glob("*-hf") if p.is_dir()],
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
for p in candidates:
    weights = list(p.glob("*.safetensors")) + list(p.glob("pytorch_model*.bin"))
    if (p / "config.json").exists() and weights:
        print(p)
        break
else:
    raise SystemExit(f"No complete HF export under {roots}")
PY
)"

OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/acl_onepaper_lm1_raw1k10k_${TAU_TAG}_tcmwiki100kgt_v2_r${RANK}_slm"
DENSITY_TAG="aclone_tcmw100kgt_v2_r${RANK}_tcmrag_${TAU_TAG}"

echo "[INFO] RANK=${RANK}"
echo "[INFO] TAU=${TAU_VALUE}"
echo "[INFO] HF_MODEL=${HF_MODEL}"
echo "[INFO] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[INFO] CUDA_VISIBLE_DEVICES_PHYSICAL=${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE}"

MODEL_NAME_OVERRIDE="${HF_MODEL}" \
TARGET_PAPER="2022.acl-long.110" \
TARGET_LM="1" \
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
DENSITY_TAG_OVERRIDE="${DENSITY_TAG}" \
RAG_SCORE_THRESHOLD_OVERRIDE="${TAU_VALUE}" \
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE}" \
VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
bash "${ONEPAPER}"
