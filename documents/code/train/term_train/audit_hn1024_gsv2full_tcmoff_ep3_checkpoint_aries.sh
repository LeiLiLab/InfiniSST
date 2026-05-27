#!/bin/bash
#SBATCH --job-name=q3_audit_off3
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_audit_off3_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_audit_off3_%x.err

set -euo pipefail

export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

CHECKPOINT="${CHECKPOINT:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep3_bs12k_smallest_dense_normAGGR_8gpu_aries_epoch_2.pt}"
OUTPUT="${OUTPUT:-/mnt/gemini/home/jiaxuanluo/train_outputs/audits/hn1024_gsv2full_tcmoff_ep3_epoch2_optimizer.json}"

python /mnt/taurus/home/jiaxuanluo/InfiniSST/scripts/tools/audit_qwen3_rag_checkpoint_optimizer.py \
    "${CHECKPOINT}" \
    --output "${OUTPUT}"

echo "[AUDIT] wrote ${OUTPUT}"
