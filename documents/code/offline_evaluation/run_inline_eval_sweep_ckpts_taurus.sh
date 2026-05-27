#!/bin/bash
#SBATCH --job-name=inline_eval_sweep
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=0-03:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_inline_eval_sweep.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_inline_eval_sweep.err

# Sweep the in-training eval code path over a curated list of historical
# retriever checkpoints so that we get SAME-METRIC gs10000_sweep@0.80 R
# numbers (+ OOD gap = acl - dev) across the 43827 (pre-aggrNorm) baseline,
# the pool-k hard-neg sweep, and the 43848/43849/43850 aggrNorm family.
#
# All evals run on a single taurus GPU with identical eval hyperparams and
# identical DEV+ACL jsonl. Any remaining small absolute drift vs the
# training-time wandb value (~0.02) is due to bf16 kernel differences
# (A100 train vs A6000 eval) and is the same for every ckpt here, so the
# RELATIVE comparison across ckpts is apples-to-apples.
#
# Output: one .log per ckpt under
#   /mnt/gemini/data1/jiaxuanluo/offline_eval/inline_sweep_tau0p80/
# plus a consolidated TSV produced by parse_inline_eval_results.py.

set -euo pipefail

SUBMIT_TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="/mnt/gemini/data1/jiaxuanluo/offline_eval/inline_sweep_tau0p80"
mkdir -p "${OUT_DIR}"

# ---------------------------------------------------------------------------
# Ckpt manifest: "tag|term_id_normalize|ckpt_path"
# - term_id_normalize MUST match the training recipe of each ckpt. Pre-
#   aggrNorm runs (43827 family, variantE pool-k sweep) used the default
#   'none'; 43848/43849/43850 explicitly set 'aggressive'. The value only
#   affects eval_loss (fn mask / pos mask); recall and sweep@tau numbers
#   are term_id_normalize-independent (dedup uses raw lowercased text).
# - Snapshots under train_outputs/snapshots/ are frozen copies taken while
#   the originating run was still live, so the ckpt file we evaluate does
#   not drift between repeated runs.
# ---------------------------------------------------------------------------
CKPTS=(
    # Pre-aggrNorm baselines (term_id_normalize=none):
    "43827_snap_step1320|none|/mnt/gemini/home/jiaxuanluo/train_outputs/snapshots/43827_best_acl6060_gs10000_step1320_0p8775.pt"
    "ps_k1024_cold_ep5_best|none|/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hardneg_per_sample_k1024_tcm_ep5_cold_best_acl6060_gs10000.pt"
    "variantE_k64_ep5_best|none|/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hardneg_tcm_ep5_best_acl6060_gs10000.pt"
    "variantE_k128_ep5_best|none|/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hardneg_k128_tcm_ep5_best_acl6060_gs10000.pt"
    "variantE_k256_ep5_best|none|/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hardneg_k256_tcm_ep5_best_acl6060_gs10000.pt"
    "variantE_k1024_ep5_best|none|/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hardneg_k1024_tcm_ep5_best_acl6060_gs10000.pt"
    # Post-aggrNorm (term_id_normalize=aggressive):
    "43849_ps_k1024_normAGGR_best|aggressive|/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hardneg_per_sample_k1024_tcm_ep5_cold_normAGGR_best_acl6060_gs10000.pt"
    "43848_smallest_k64_normAGGR_snap|aggressive|/mnt/gemini/home/jiaxuanluo/train_outputs/snapshots/20260422_q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hardneg_k64_tcm_ep3_cold_smallest_dense_normAGGR_best_acl6060_gs10000.pt"
    "43850_NOhardneg_normAGGR_snap|aggressive|/mnt/gemini/home/jiaxuanluo/train_outputs/snapshots/20260422_q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_NOhardneg_tcm_ep5_cold_normAGGR_best_acl6060_gs10000.pt"
)

RUNNER="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/offline_evaluation/inline_eval_retriever.sh"

echo "[SWEEP] submit_ts=${SUBMIT_TS}"
echo "[SWEEP] ckpt count=${#CKPTS[@]}"
echo "[SWEEP] OUT_DIR=${OUT_DIR}"

n_fail=0
for entry in "${CKPTS[@]}"; do
    tag="${entry%%|*}"
    rest="${entry#*|}"
    norm="${rest%%|*}"
    ckpt="${rest#*|}"
    out_log="${OUT_DIR}/${tag}.log"

    if [[ ! -f "${ckpt}" ]]; then
        echo "[SWEEP][FATAL] missing ckpt for tag=${tag}: ${ckpt}" >&2
        exit 3
    fi

    echo ""
    echo "====================================================================="
    echo "[SWEEP] tag=${tag}"
    echo "[SWEEP]  norm=${norm}"
    echo "[SWEEP]  ckpt=${ckpt}"
    echo "[SWEEP]  out=${out_log}"
    echo "[SWEEP]  started at $(date)"
    echo "====================================================================="

    # Sub-run should NOT abort the sweep on a single-ckpt failure; we run
    # each inside a subshell and collect exit codes so subsequent ckpts
    # still get evaluated.
    CKPT="${ckpt}" \
    OUT_LOG="${out_log}" \
    TERM_ID_NORMALIZE="${norm}" \
    bash "${RUNNER}" \
        && echo "[SWEEP] tag=${tag} OK" \
        || { echo "[SWEEP][FAIL] tag=${tag} exit=$?" >&2; n_fail=$((n_fail + 1)); }
done

echo ""
echo "[SWEEP] done at $(date). failures=${n_fail}/${#CKPTS[@]}"

# Aggregate immediately after sweep so the TSV lands in the same dir.
PARSER="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/offline_evaluation/parse_inline_eval_results.py"
TSV_OUT="${OUT_DIR}/summary_${SUBMIT_TS}.tsv"
python3 "${PARSER}" \
    --log_dir "${OUT_DIR}" \
    --out_tsv "${TSV_OUT}" \
    && echo "[SWEEP] summary -> ${TSV_OUT}" \
    || echo "[SWEEP][WARN] aggregation failed; re-run parse_inline_eval_results.py manually"

exit "${n_fail}"
