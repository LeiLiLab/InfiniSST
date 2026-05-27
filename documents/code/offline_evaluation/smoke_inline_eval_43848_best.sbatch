#!/bin/bash
#SBATCH --job-name=smoke_inline_eval_43848
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_smoke_inline_eval.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_smoke_inline_eval.err

# Smoke test for inline_eval_retriever.sh.
#
# Evaluates 43848's *_best_acl6060_gs10000.pt snapshot — the numbers should
# match the training-time log at the step this ckpt was saved (step=480):
#   DEV   r@10_gs10000 ~= 0.9099  gs10000_sweep@0.80 R ~= 0.8680
#   ACL   r@10_gs10000 ~= 0.8233  gs10000_sweep@0.80 R ~= 0.7320
#
# If the reproduced numbers are within ~0.01 of the training-time values,
# the inline-eval code path is validated and we can launch the full sweep.

set -euo pipefail

CKPT="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hardneg_k64_tcm_ep3_cold_smallest_dense_normAGGR_best_acl6060_gs10000.pt"
OUT_LOG="/mnt/gemini/data1/jiaxuanluo/offline_eval/inline_sweep_tau0p80/smoke_43848_best_acl6060_gs10000.log"

mkdir -p "$(dirname "${OUT_LOG}")"

export CKPT OUT_LOG
# 43848 was trained with --term_id_normalize aggressive; must match or the
# inline eval will disagree with the training-time saved metric value.
export TERM_ID_NORMALIZE="aggressive"
bash /mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/offline_evaluation/inline_eval_retriever.sh

echo "[SMOKE] Extracting key metrics from ${OUT_LOG}:"
grep -E "EVAL_DEV|EVAL_ACL6060" "${OUT_LOG}" | head -4 || true
