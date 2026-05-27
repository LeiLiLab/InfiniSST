#!/bin/bash
# Run the in-training eval code path (`qwen3_glossary_neg_train.py --eval_only`)
# on a SINGLE retriever checkpoint, so we get metrics in the SAME format as
# during training:
#   [EVAL_DEV] ... gs10000_sweep@0.80: R=... P_mic=...
#   [EVAL_ACL6060] ... gs10000_sweep@0.80: R=... P_mic=...
#
# This lets us retro-compute `topk10_filtered_recall@tau_0p80_gs10000` on
# older checkpoints (43827 pre-aggrNorm baseline, earlier variantE HN k
# sweeps, etc.) that were trained before this metric was logged inline.
#
# Usage:
#   CKPT=/abs/path/to/ckpt.pt OUT_LOG=/abs/path/log.txt bash inline_eval_retriever.sh
#
# Designed to be driven by run_inline_eval_sweep_ckpts_taurus.sh.
#
# Fail-loud: no defaults for CKPT or OUT_LOG; bail if either is missing.

set -euo pipefail

: "${CKPT:?CKPT env var is required (absolute path to retriever .pt)}"
: "${OUT_LOG:?OUT_LOG env var is required (absolute path to write stdout+stderr)}"

# term_id_normalize affects the bank-dedup + false-negative mask used in
# eval. It MUST match the training-time value of the ckpt, or the reported
# recall / sweep numbers will not reconcile with the training log. Caller
# must set TERM_ID_NORMALIZE explicitly (no default) — fail loud if missing
# so the common pitfall of silently using 'none' is caught at submit time.
: "${TERM_ID_NORMALIZE:?TERM_ID_NORMALIZE env var is required: one of {none, lower_strip, aggressive} matching the ckpt's training recipe}"
case "${TERM_ID_NORMALIZE}" in
    none|lower_strip|aggressive) ;;
    *)
        echo "[INLINE_EVAL][FATAL] TERM_ID_NORMALIZE='${TERM_ID_NORMALIZE}' invalid; expected none|lower_strip|aggressive" >&2
        exit 2
        ;;
esac

if [[ ! -f "${CKPT}" ]]; then
    echo "[INLINE_EVAL][FATAL] CKPT not found: ${CKPT}" >&2
    exit 2
fi
mkdir -p "$(dirname "${OUT_LOG}")"

# ---- Environment (match training recipe) -------------------------------
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

export HF_HOME="/mnt/taurus/data/jiaxuanluo/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TORCH_HOME="/mnt/taurus/data/jiaxuanluo/cache/torch"
export XDG_CACHE_HOME="/mnt/taurus/data/jiaxuanluo/cache"

export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Default to GPU 0 if caller didn't pin a device. Must be set explicitly per
# project rules — never rely on implicit device selection.
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

# ---- Fixed inputs shared with training ---------------------------------
SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/qwen3_glossary_neg_train.py"
# --train_jsonl is argparse-required even under --eval_only, but it is only
# READ (never opened) in the non-eval branch. We point it at the same merged
# jsonl used in training so the path is valid and the intent is obvious.
TRAIN_JSONL="/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_mfa.jsonl"
DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
ACL_DEV_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
EVAL_WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"

# Architecture flags (identical across the 43827 / 43848 / 43849 / 43850
# family — rank 128 LoRA, target_dim 1024, transformer pooling, max-sim).
LORA_RANK=128
LORA_ALPHA=256
TEXT_LORA_RANK=128
TEXT_LORA_ALPHA=256
TARGET_DIM=1024
POOLING_TYPE="transformer"
TEXT_POOLING="cls"
TEMPERATURE="0.07"
SPARSE_WEIGHT="0.0"

# At eval time max-sim uses unrestricted max across windows, so window-set
# choice does NOT affect the metric — but we pin the legacy 4-window grid
# so that re-runs stay reproducible regardless of the training recipe.
MAXSIM_WINDOWS="6 10 16 24"
MAXSIM_STRIDE=2

# ---- Eval knobs that must match the training-time emission -------------
EVAL_BATCH_SIZE=256
EVAL_TOPK=10
EVAL_GLOSSARY_SIZES="1000 10000"
TCM_POS_THRESHOLD="0.85"
TCM_NEG_THRESHOLD="0.25"
TCM_LOSS_FORM="squared_hinge"
TCM_REDUCTION="mean_viol"
TCM_SWEEP_THRESHOLDS="0.5 0.6 0.7 0.8"

echo "[INLINE_EVAL] CKPT=${CKPT}"
echo "[INLINE_EVAL] OUT_LOG=${OUT_LOG}"
echo "[INLINE_EVAL] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INLINE_EVAL] started at $(date)"

# torchrun is used so the training script's DDP init logic (world_size=1)
# works unchanged. nproc_per_node=1, single process, no multi-GPU rank
# contention — ideal for eval_only since the eval itself only runs on rank 0.
torchrun \
    --nproc_per_node=1 \
    --master_addr="127.0.0.1" \
    --master_port=$((29000 + RANDOM % 1000)) \
    "${SCRIPT_PATH}" \
    --eval_only \
    --resume "${CKPT}" \
    --train_jsonl "${TRAIN_JSONL}" \
    --dev_jsonl "${DEV_JSONL}" \
    --acl_dev_jsonl "${ACL_DEV_JSONL}" \
    --eval_wiki_glossary "${EVAL_WIKI_GLOSSARY}" \
    --eval_glossary_sizes ${EVAL_GLOSSARY_SIZES} \
    --eval_topk "${EVAL_TOPK}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --eval_minimal_metrics \
    --eval_top100_samples 0 \
    --tcm_pos_threshold "${TCM_POS_THRESHOLD}" \
    --tcm_neg_threshold "${TCM_NEG_THRESHOLD}" \
    --tcm_loss_form "${TCM_LOSS_FORM}" \
    --tcm_reduction "${TCM_REDUCTION}" \
    --tcm_sweep_thresholds ${TCM_SWEEP_THRESHOLDS} \
    --target_dim "${TARGET_DIM}" \
    --pooling_type "${POOLING_TYPE}" \
    --text_pooling "${TEXT_POOLING}" \
    --sparse_weight "${SPARSE_WEIGHT}" \
    --temperature "${TEMPERATURE}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --text_lora_rank "${TEXT_LORA_RANK}" \
    --text_lora_alpha "${TEXT_LORA_ALPHA}" \
    --lora_target_modules q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2 \
    --text_lora_target_modules query key value dense \
    --term_id_normalize "${TERM_ID_NORMALIZE}" \
    --use_lora \
    --use_maxsim \
    --maxsim_windows ${MAXSIM_WINDOWS} \
    --maxsim_stride "${MAXSIM_STRIDE}" \
    --hard_neg_k 0 \
    --hard_neg_k_per_sample 0 \
    --neg_bank_size 0 \
    --glossary_neg_path "" \
    --glossary_neg_refresh_steps 0 \
    --wiki_rank 0 \
    --batch_size 1 \
    --epochs 1 \
    --lr 0 \
    --text_lr 0 \
    --save_path "/tmp/_unused_${USER}_${RANDOM}.pt" \
    2>&1 | tee "${OUT_LOG}"

echo "[INLINE_EVAL] done at $(date) -> ${OUT_LOG}"
