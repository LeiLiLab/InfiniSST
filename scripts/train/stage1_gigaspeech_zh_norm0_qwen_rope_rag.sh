#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=392GB
#SBATCH --gres=gpu:3
#SBATCH --partition=taurus
##SBATCH --array=1-5
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jaxanluo@gmail.com
#SBATCH -e train_infinisst_%j.err
#SBATCH -o train_infinisst_%j.out



set -euo pipefail

# Activate Conda environment
if [ -f "/home/jiaxuanluo/miniconda3/etc/profile.d/conda.sh" ]; then
    source /home/jiaxuanluo/miniconda3/etc/profile.d/conda.sh
    conda activate infinisst
else
    source /home/jaxanluo/anaconda3/bin/activate infinisst
fi

# ------------------------------
# Model and encoder paths (aligned with serve/inference_engine.py)
# ------------------------------
llm_path="/mnt/aries/data6/jiaxuanluo/Qwen2.5-7B-Instruct"
w2v2_path="/mnt/aries/data6/xixu/demo/wav2_vec_vox_960h_pl.pt"

w2v2_type=w2v2
ctc_finetuned=True

# ------------------------------
# Data configuration
# ------------------------------
preferred_root="/mnt/data/siqiouyang/datasets/gigaspeech/"
ROOT=${preferred_root}

lang_code=zh
lang=Chinese
data_path=${ROOT}

# ------------------------------
# Checkpoint/output configuration
# ------------------------------
model_export_dir="/mnt/aries/data6/jiaxuanluo/demo/en-${lang_code}"
run_root="${model_export_dir}/runs"

source_lang="English"
target_lang=${lang} # e.g. German
name="stage1_M=12_ls-cv-vp_norm0_qwen_rope_vv2_gigaspeech"
save_path=${run_root}/${name}
#rm -rf ${save_path} # comment this line if you want to resume training
mkdir -p ${save_path}
mkdir -p ${model_export_dir}

export PYTHONPATH=${PYTHONPATH:-}:$PWD
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="infinisst"
export WANDB_ENTITY="luojiaxuan1215-johns-hopkins-university"

# Specify which GPUs to use (single task with 2 GPUs)
export CUDA_VISIBLE_DEVICES=1,2

# disable P2P and InfiniBand for L40S 8-GPU nodes
# if your node supports P2P and InfiniBand, you need to remove these two lines
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO

if [ -n "${SLURM_GPUS_PER_NODE:-}" ]; then
    NUM_GPUS=${SLURM_GPUS_PER_NODE}
elif [ -n "${SLURM_GPUS:-}" ]; then
    if [[ ${SLURM_GPUS} =~ ^[0-9]+$ ]]; then
        NUM_GPUS=${SLURM_GPUS}
    else
        NUM_GPUS=$(echo "${SLURM_GPUS}" | awk -F',' '{print NF}')
    fi
else
    NUM_GPUS=2
fi

python train/main.py \
    \
    --model_type w2v2_qwen25 \
    \
    --w2v2_path ${w2v2_path} \
    --w2v2_type ${w2v2_type} \
    --ctc_finetuned ${ctc_finetuned} \
    --length_shrink_cfg "[(1024,2,2)] * 2" \
    --block_size 48 \
    --max_cache_size 576 \
    \
    --llm_path ${llm_path} \
    --llm_freeze True \
    --llm_emb_freeze True \
    --llm_head_freeze True \
    --use_flash_attn True \
    \
    --data_path ${data_path} \
    --data_split_train 'train_xl_case_ft-qwen2.5-32b-instruct_marked_mfa_punc_asr' \
    --data_split_eval 'dev_case_ft-qwen2.5-32b-instruct_marked_mfa_punc' \
    --source_lang "${source_lang}" \
    --target_lang "${target_lang}" \
    --trajectory 9 \
    --trajectory_max_multiplier 12 \
    --trajectory_prob_aug 0.0 \
    --audio_normalize False \
    \
    --seed 998244353 \
    --stage 1 \
    --train_bsz 1800 \
    --eval_bsz 1800 \
    --bsz_sent 2 \
    --learning_rate 2e-4 \
    --warmup_steps 1000 \
    --run_name $name \
    \
    --n_device ${NUM_GPUS} \
    --max_epochs 1 \
    --grad_acc_steps 4 \
    --clip_norm 1.0 \
    --save_dir ${save_path} \
    --save_step 1000 \
    --log_step 100 \
    --eval_step 1000

# Using DDPStrategy (no DeepSpeed), checkpoint is saved in Lightning format directly
# Extract model weights from Lightning checkpoint
if [ -f "${save_path}/last.ckpt/checkpoint" ]; then
    python -c "
import torch
ckpt = torch.load('${save_path}/last.ckpt/checkpoint', map_location='cpu')
torch.save(ckpt['state_dict'], '${save_path}/last.ckpt/pytorch_model.bin')
print('[INFO] Extracted model weights from Lightning checkpoint')
"
    # Export to the location expected by the inference service
    cp -f ${save_path}/last.ckpt/pytorch_model.bin ${model_export_dir}/pytorch_model.bin
    echo "[INFO] Exported Stage1 checkpoint to ${model_export_dir}/pytorch_model.bin"
else
    echo "[WARN] No checkpoint found at ${save_path}/last.ckpt/checkpoint"
fi
