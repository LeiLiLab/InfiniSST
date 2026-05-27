#!/usr/bin/env bash
set -euo pipefail

# Auto-select latest *-hf under each keep folder and submit simuleval (zh only).
#
# Default base dir:
#   /mnt/gemini/data/jiaxuanluo/Omni-30B-sampling-0107
#
# It finds:
#   keep0.3/ <latest>*-hf
#   keep0.5/ <latest>*-hf
#   keep0.8/ <latest>*-hf
#   keep1.0/ <latest>*-hf
#
# For each existing hf dir, it runs:
#   ONLY_LANG=zh MODEL_NAME_OVERRIDE=... OUTPUT_BASE_OVERRIDE=... sbatch documents/code/run_simuleval_rag_aries_v4_final_result_taurus.sh
#
# Logs and outputs:
# - Slurm logs: configured inside the simuleval script (#SBATCH --output/--error)
# - Simuleval outputs: under OUTPUT_BASE_OVERRIDE/zh/...

ROOT_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST"
SIM_SCRIPT="${ROOT_DIR}/documents/code/run_simuleval_rag_aries_v4_final_result_taurus.sh"

: "${SAVE_BASE:=/mnt/gemini/data/jiaxuanluo/Omni-30B-sampling-0107}"
: "${KEEP_DIRS:=keep0.5}"
: "${ONLY_LANG:=zh}"

# Put results into a dedicated folder to avoid mixing with other experiments.
: "${OUTPUT_BASE_OVERRIDE:=/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_sweep_v4_sampling_latest_hf_zh}"

echo "[INFO] SAVE_BASE=${SAVE_BASE}"
echo "[INFO] KEEP_DIRS=${KEEP_DIRS}"
echo "[INFO] ONLY_LANG=${ONLY_LANG}"
echo "[INFO] OUTPUT_BASE_OVERRIDE=${OUTPUT_BASE_OVERRIDE}"
echo "[INFO] SIM_SCRIPT=${SIM_SCRIPT}"

if [[ ! -f "${SIM_SCRIPT}" ]]; then
  echo "[ERROR] Simuleval script not found: ${SIM_SCRIPT}" >&2
  exit 2
fi

find_latest_hf() {
  local dir="$1"
  # newest by mtime
  ls -1dt "${dir}"/*-hf 2>/dev/null | head -n 1 || t212`rue
}

submitted=0
skipped=0

for keep in ${KEEP_DIRS}; do
  keep_dir="${SAVE_BASE}/${keep}"
  if [[ ! -d "${keep_dir}" ]]; then
    echo "[WARN] Missing keep dir: ${keep_dir} (skip)"
    skipped=$((skipped+1))
    continue
  fi

  hf_dir="$(find_latest_hf "${keep_dir}")"
  if [[ -z "${hf_dir}" ]]; then
    echo "[WARN] No *-hf found under: ${keep_dir} (skip)"
    skipped=$((skipped+1))
    continue
  fi

  echo "[INFO] keep=${keep} latest_hf=${hf_dir}"

  ONLY_LANG="${ONLY_LANG}" \
  MODEL_NAME_OVERRIDE="${hf_dir}" \
  OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE_OVERRIDE}/${keep}" \
  DISABLE_GPU_IDS="" \
  sbatch "${SIM_SCRIPT}"

  submitted=$((submitted+1))
done

echo "[INFO] Done. submitted=${submitted} skipped=${skipped}"


