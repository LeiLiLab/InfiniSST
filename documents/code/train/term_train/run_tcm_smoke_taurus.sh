#!/bin/bash
# Smoke test for TCM auxiliary loss integration.
#
# Single-GPU on taurus, ~8 optimizer steps (PER_GPU_BATCH=256, TRAIN_LIMIT=2000),
# no sample-eval, no WandB, no checkpoint save.  Verifies:
#   1) compute_masked_contrastive_loss still produces a usable total_loss
#      when TCM is enabled,
#   2) TCM components (loss_tcm_pos / loss_tcm_neg / viol rates) move in
#      the expected directions,
#   3) Backward pass propagates TCM gradients through the GradCache path.
#
# Launch:
#   env -u CUDA_VISIBLE_DEVICES \
#       bash documents/code/train/term_train/run_tcm_smoke_taurus.sh
#
# Log goes to stdout.

set -euo pipefail

# ======Configuration=====
SMOKE_GPU="${SMOKE_GPU:-5}"
SMOKE_TRAIN_LIMIT=2000
SMOKE_PER_GPU_BATCH=256
SMOKE_GRAD_CACHE_CHUNK=32
SMOKE_EPOCHS=6  # resume epoch is 3; need +3 more to actually run ~24 iters
SMOKE_LR="5e-5"
SMOKE_EVAL_STEPS_SAMPLE=999999  # disable periodic eval
SMOKE_SAVE_STEPS=999999          # disable step-save
SMOKE_LOG_DIR="/mnt/gemini/data2/jiaxuanluo/tcm_smoke_logs"

CONDA_PREFIX_DIR="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
TRAIN_JSONL="/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_mfa.jsonl"
DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
ACL_DEV_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
EVAL_WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"
SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/qwen3_glossary_neg_train.py"
RESUME_PATH="/mnt/aries/data4/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.1_maxsim_mfa_final_C_best_acl6060_gs10000.pt"
SMOKE_SAVE_PATH="/mnt/gemini/data2/jiaxuanluo/tcm_smoke_logs/smoke_ckpt.pt"

# TCM config under test.
TCM_LOSS_WEIGHT="${TCM_LOSS_WEIGHT:-0.1}"
TCM_POS_THRESHOLD="0.7"
TCM_NEG_THRESHOLD="0.4"
TCM_LOSS_FORM="squared_hinge"
TCM_REDUCTION="mean_viol"
HCL_BETA="${HCL_BETA:-0.0}"

# Model / training recipe, mirroring the Aries 8-GPU Config C launcher.
LORA_RANK=128
LORA_ALPHA=256
TARGET_DIM=1024
TARGET_MODULES="q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2"
POOLING_TYPE="transformer"
MAXSIM_WINDOWS="6 10 16 24"
MAXSIM_STRIDE=2
TEXT_LORA_RANK=128
TEXT_LORA_ALPHA=256
TEXT_TARGET_MODULES="query key value dense"
TEXT_POOLING="cls"
SPARSE_WEIGHT="0.0"
TEMPERATURE="0.07"
MARGIN=0.1
WIKI_RANK=1000000
# ======Configuration=====

mkdir -p "${SMOKE_LOG_DIR}"

export PATH="${CONDA_PREFIX_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX_DIR}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

LOCAL_TMP_DIR="/dev/shm/${USER}/pytorch_tmp"
mkdir -p "${LOCAL_TMP_DIR}"
export TMPDIR="${LOCAL_TMP_DIR}"
export TMP="${LOCAL_TMP_DIR}"
export TEMP="${LOCAL_TMP_DIR}"

export CUDA_VISIBLE_DEVICES="${SMOKE_GPU}"
export HF_HOME="/mnt/taurus/data/jiaxuanluo/cache/huggingface"
export TORCH_HOME="/mnt/taurus/data/jiaxuanluo/cache/torch"
export XDG_CACHE_HOME="/mnt/taurus/data/jiaxuanluo/cache"

MASTER_ADDR="127.0.0.1"
MASTER_PORT=29952

echo "[SMOKE] GPU=${SMOKE_GPU} train_limit=${SMOKE_TRAIN_LIMIT} batch=${SMOKE_PER_GPU_BATCH}"
echo "[SMOKE] TCM: lambda=${TCM_LOSS_WEIGHT} T_beta=${TCM_POS_THRESHOLD} T_alpha=${TCM_NEG_THRESHOLD} form=${TCM_LOSS_FORM} reduction=${TCM_REDUCTION}"
echo "[SMOKE] HCL: beta=${HCL_BETA}"

torchrun \
    --nproc_per_node=1 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_PATH}" \
    --train_jsonl "${TRAIN_JSONL}" \
    --dev_jsonl "${DEV_JSONL}" \
    --save_path "${SMOKE_SAVE_PATH}" \
    --resume "${RESUME_PATH}" \
    --train_limit "${SMOKE_TRAIN_LIMIT}" \
    --batch_size "${SMOKE_PER_GPU_BATCH}" \
    --grad_cache_chunk_size "${SMOKE_GRAD_CACHE_CHUNK}" \
    --epochs "${SMOKE_EPOCHS}" \
    --num_workers 2 \
    --lr "${SMOKE_LR}" \
    --text_lr 0 \
    --temperature "${TEMPERATURE}" \
    --target_dim "${TARGET_DIM}" \
    --pooling_type "${POOLING_TYPE}" \
    --use_lora \
    --use_maxsim \
    --mfa_supervised_maxsim \
    --maxsim_windows ${MAXSIM_WINDOWS} \
    --maxsim_stride "${MAXSIM_STRIDE}" \
    --text_pooling "${TEXT_POOLING}" \
    --sparse_weight "${SPARSE_WEIGHT}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --text_lora_rank "${TEXT_LORA_RANK}" \
    --text_lora_alpha "${TEXT_LORA_ALPHA}" \
    --lora_target_modules ${TARGET_MODULES} \
    --text_lora_target_modules ${TEXT_TARGET_MODULES} \
    --margin "${MARGIN}" \
    --wiki_rank "${WIKI_RANK}" \
    --save_steps "${SMOKE_SAVE_STEPS}" \
    --eval_steps_sample "${SMOKE_EVAL_STEPS_SAMPLE}" \
    --tcm_loss_weight "${TCM_LOSS_WEIGHT}" \
    --tcm_pos_threshold "${TCM_POS_THRESHOLD}" \
    --tcm_neg_threshold "${TCM_NEG_THRESHOLD}" \
    --tcm_loss_form "${TCM_LOSS_FORM}" \
    --tcm_reduction "${TCM_REDUCTION}" \
    --hcl_beta "${HCL_BETA}"

echo "[SMOKE] done at $(date)"
