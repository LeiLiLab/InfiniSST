#!/bin/bash
set -euo pipefail

# Submit the Phase 4 adversarial copy-faith training jobs.
#
# Two 2-GPU (EP=2) LoRA rank=16 jobs across taurus + aries partitions:
#   * control    -> d5_cap r16        taurus  (data: train_maxsim_varlen_d5_cap.jsonl)
#   * experiment -> d5_cap_adv r16    aries   (data: train_maxsim_varlen_d5_cap_adv.jsonl)
#
# Output goes to /mnt/taurus/data2/jiaxuanluo/speech_llm_density_ablation/. The docker
# bind-mount for this path is partition-aware:
#   * on taurus nodes the taurus disk is local at /mnt/data2, so we mount
#     /mnt/data2 -> /mnt/taurus/data2 (dereferences the local symlink).
#   * on aries nodes taurus is reachable over NFS at /mnt/taurus/data2, so we
#     mount that NFS path directly.
#
# Reuses the existing 2-GPU maxsim trainer inside docker via DATASET_PATH_OVERRIDE
# and SAVE_BASE_OVERRIDE so we do not fork the training script.
#
# Usage:
#   bash run_adversarial_train_sbatch.sh
#   bash run_adversarial_train_sbatch.sh control       # submit only one variant
#   bash run_adversarial_train_sbatch.sh experiment

# ======Configuration=====
DOCKER_IMAGE="modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.1"
TRAIN_SCRIPT="/workspace/InfiniSST/documents/code/train/sst_omni_train/run_speech_llm_4gpu_maxsim.sh"
LOG_DIR="/mnt/gemini/data1/jiaxuanluo/logs"

BASE_MODEL_HOST="/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct"
BASE_MODEL_DOCKER="/workspace/Qwen3-Omni-30B-A3B-Instruct"

# Variant definitions:
#   tag:density_arg:dataset_path:save_base:port_offset:partition
# density_arg is passed as $1 to the trainer. It is used to derive the wandb
# experiment name (omni-maxsim-varlen-d<density_arg>-r<rank>-2gpu) and thus must
# be distinct across variants so the two jobs do not share a wandb run.
VARIANTS=(
  "control:5_cap_control:/mnt/gemini/data1/jiaxuanluo/adversarial/train_maxsim_varlen_d5_cap.jsonl:/mnt/taurus/data2/jiaxuanluo/speech_llm_density_ablation/d5_cap:0:taurus"
  "experiment:5_cap_adv:/mnt/gemini/data1/jiaxuanluo/adversarial/train_maxsim_varlen_d5_cap_adv.jsonl:/mnt/taurus/data2/jiaxuanluo/speech_llm_density_ablation/d5_cap_adv:1:aries"
  # "half" variant (perturb-prob 0.25) was attempted on 2026-03-28 but aborted
  # at iter 150/585 because the aries node was 4.4x slower than the reference
  # 43715 run (67s/iter vs 15s/iter) due to shared-resource contention on that
  # partition, and the 3-way (d5/d5_cap/d5_cap_adv) analysis already showed the
  # adversarial track is a net loss vs the no-cap baseline. See
  # simuleval/dev_journal.md and train/sst_omni_train/dev_journal.md.
  # "half:5_cap_adv_half:/mnt/gemini/data1/jiaxuanluo/adversarial/train_maxsim_varlen_d5_cap_adv_half.jsonl:/mnt/taurus/data2/jiaxuanluo/speech_llm_density_ablation/d5_cap_adv_half:2:aries"
)

BASE_PORT="29551"
LORA_RANK="16"
# ======Configuration=====

ONLY_VARIANT="${1:-}"

mkdir -p "${LOG_DIR}"

echo "=============================================="
echo " Submitting Phase 4 adversarial training jobs"
echo " rank=${LORA_RANK}"
echo "=============================================="

submit_one() {
  local tag="$1"
  local density_arg="$2"
  local dataset="$3"
  local save_base="$4"
  local port_offset="$5"
  local partition="$6"

  if [[ ! -f "${dataset}" ]]; then
    echo "[ERROR] Missing dataset for variant=${tag}: ${dataset}" >&2
    exit 2
  fi

  # Partition-aware bind mount for the taurus output disk. On taurus the disk is
  # local at /mnt/data2; on aries it is NFS-mounted at /mnt/taurus/data2.
  local TAURUS_DATA2_HOST
  case "${partition}" in
    taurus) TAURUS_DATA2_HOST="/mnt/data2" ;;
    aries)  TAURUS_DATA2_HOST="/mnt/taurus/data2" ;;
    *)
      echo "[ERROR] Unsupported partition='${partition}' for variant=${tag}" >&2
      exit 2
      ;;
  esac

  local port=$((BASE_PORT + port_offset))

  local SBATCH_SCRIPT
  SBATCH_SCRIPT="$(mktemp "/tmp/train_${density_arg}_XXXXXX.sh")"
  cat > "${SBATCH_SCRIPT}" << INNER_EOF
#!/bin/bash
#SBATCH --job-name=train_${density_arg}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
set -euo pipefail

echo "[TRAIN ${tag}] Starting on \$(hostname), partition=${partition}"
echo "[TRAIN ${tag}] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-not set}"
echo "[TRAIN ${tag}] taurus_data2_host=${TAURUS_DATA2_HOST}"

ALLOCATED_GPUS="\${CUDA_VISIBLE_DEVICES:-0,1}"
# Real GPU isolation for shared nodes (aries): docker --gpus device=... only
# exposes the slurm-allocated physical GPUs to the container. Inside, those are
# re-indexed starting at 0, so CUDA_VISIBLE_DEVICES must be the container-local
# view (0,1), not the host-global allocation.
echo "[TRAIN ${tag}] docker --gpus device=\${ALLOCATED_GPUS} (container-local CVD=0,1)"

docker run --rm \\
    --gpus "\"device=\${ALLOCATED_GPUS}\"" \\
    --shm-size=32g \\
    --ipc=host \\
    -e CUDA_VISIBLE_DEVICES="0,1" \\
    -e NCCL_P2P_DISABLE=1 \\
    -e NCCL_IB_DISABLE=1 \\
    -e WANDB_MODE=offline \\
    -e DATASET_PATH_OVERRIDE="${dataset}" \\
    -e SAVE_BASE_OVERRIDE="${save_base}" \\
    -e LORA_RANK_OVERRIDE="${LORA_RANK}" \\
    -v /mnt/taurus/home/jiaxuanluo/InfiniSST:/workspace/InfiniSST \\
    -v ${BASE_MODEL_HOST}:${BASE_MODEL_DOCKER}:ro \\
    -v /mnt/gemini/data:/mnt/gemini/data \\
    -v /mnt/gemini/data1:/mnt/gemini/data1 \\
    -v /mnt/gemini/data2:/mnt/gemini/data2 \\
    -v ${TAURUS_DATA2_HOST}:/mnt/taurus/data2 \\
    "${DOCKER_IMAGE}" \\
    bash "${TRAIN_SCRIPT}" "${density_arg}" "${port}"

echo "[TRAIN ${tag}] Finished."
INNER_EOF

  local JOB_ID
  JOB_ID=$(sbatch --parsable \
      -p "${partition}" \
      -o "${LOG_DIR}/%j_train_${density_arg}.out" \
      -e "${LOG_DIR}/%j_train_${density_arg}.err" \
      "${SBATCH_SCRIPT}")

  echo "  variant=${tag}  density_arg=${density_arg}  partition=${partition}"
  echo "    dataset=${dataset}"
  echo "    save_base=${save_base}"
  echo "    port=${port}  job=${JOB_ID}"
}

for entry in "${VARIANTS[@]}"; do
  IFS=':' read -r tag density_arg dataset save_base port_offset partition <<< "${entry}"
  if [[ -n "${ONLY_VARIANT}" && "${ONLY_VARIANT}" != "${tag}" ]]; then
    continue
  fi
  submit_one "${tag}" "${density_arg}" "${dataset}" "${save_base}" "${port_offset}" "${partition}"
done

echo ""
echo "=============================================="
echo " Done. Monitor:  squeue -u \$(whoami)"
echo " Logs:           ${LOG_DIR}/"
echo "=============================================="
