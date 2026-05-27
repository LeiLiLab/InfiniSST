#!/bin/bash
#SBATCH --job-name=mt_hnmask_smoke
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=0-03:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_mt_hnmask_smoke.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_mt_hnmask_smoke.err

set -euo pipefail

export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_MODE=online

export HF_HOME="/mnt/taurus/data/jiaxuanluo/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TORCH_HOME="/mnt/taurus/data/jiaxuanluo/cache/torch"
export XDG_CACHE_HOME="/mnt/taurus/data/jiaxuanluo/cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/qwen3_glossary_neg_train.py"
TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_partial0_20_oldwiki_clean_mfa.jsonl"
DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
ACL_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
EVAL_WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"
NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_multiterm_hn_mask_smoke_taurus.md"
SAVE_PATH="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_multiterm_hn_mask_smoke_taurus2.pt"

echo "[SMOKE] started_at=$(date)"
echo "[SMOKE] script=${SCRIPT}"
echo "[SMOKE] notes=${NOTES_FILE}"

python3 "${SCRIPT}" \
  --train_jsonl "${TRAIN_JSONL}" \
  --dev_jsonl "${DEV_JSONL}" \
  --save_path "${SAVE_PATH}" \
  --use_lora \
  --use_maxsim \
  --mfa_supervised_maxsim \
  --lr 1.7e-4 \
  --text_lr 0 \
  --batch_size 64 \
  --epochs 1 \
  --train_limit 20000 \
  --num_workers 2 \
  --temperature 0.07 \
  --target_dim 1024 \
  --pooling_type transformer \
  --maxsim_windows 2 3 4 5 6 7 8 10 12 16 20 24 \
  --maxsim_stride 2 \
  --mfa_window_selection smallest \
  --mfa_positive_scope auto \
  --text_pooling cls \
  --sparse_weight 0.0 \
  --lora_rank 128 \
  --lora_alpha 256 \
  --text_lora_rank 128 \
  --text_lora_alpha 256 \
  --lora_target_modules q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2 \
  --text_lora_target_modules query key value dense \
  --glossary_neg_path "" \
  --glossary_neg_refresh_steps 0 \
  --neg_bank_size 0 \
  --neg_bank_refresh_steps 50 \
  --hard_neg_k 0 \
  --hard_neg_k_per_sample 32 \
  --noisy_ratio 0.0 \
  --margin 0.0 \
  --online_hard_neg_k 0 \
  --grad_cache_chunk_size 32 \
  --save_steps 999999 \
  --max_steps 20 \
  --eval_steps_sample 20 \
  --eval_batch_size 256 \
  --eval_topk 10 \
  --keep_checkpoints 1 \
  --acl_dev_jsonl "${ACL_JSONL}" \
  --eval_wiki_glossary "${EVAL_WIKI_GLOSSARY}" \
  --eval_glossary_sizes 1000 10000 \
  --best_metric "eval_acl6060/recall@10_gs1000" \
  --best_metric_secondary "eval_acl6060/recall@10_gs10000" \
  --eval_top100_samples 0 \
  --eval_minimal_metrics \
  --enable_wandb \
  --wandb_project qwen3_rag \
  --wandb_exp_name "multiterm_hnmask_smoke_taurus2" \
  --tcm_loss_weight 0.0 \
  --tcm_pos_loss_weight 0.10 \
  --tcm_neg_loss_weight 0.50 \
  --tcm_pos_threshold 0.85 \
  --tcm_neg_threshold 0.25 \
  --tcm_loss_form hinge \
  --tcm_reduction mean_viol \
  --tcm_neg_scope topk \
  --tcm_neg_topk 32 \
  --tcm_warmup_steps 2 \
  --hcl_beta 0.0 \
  --term_id_normalize aggressive \
  --max_train_seconds 7200 \
  --experiment_family sst_ood_hardneg \
  --data_tag 3variant_gsv2p020_oldwiki_mfa \
  --task_tag smoke \
  --extra_wandb_tags variant:mt_hnmask_smoke_taurus2 compute:taurus-1gpu \
  --baseline_run_ids fma3wmh2 yx52spnl tys70s0y \
  --notes_file "${NOTES_FILE}" \
  --run_verdict "Functional smoke for multi-term chunk HN masking and MFA term-scoped positives; metrics are not scientific evidence."

echo "[SMOKE] finished_at=$(date)"
