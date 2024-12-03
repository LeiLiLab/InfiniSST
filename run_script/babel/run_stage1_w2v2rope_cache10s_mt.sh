#!/usr/bin/env bash

##SBATCH --nodelist=babel-4-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512GB
#SBATCH --gres=gpu:L40S:8
#SBATCH --exclude=babel-13-13,babel-3-9,babel-13-29
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
##SBATCH --dependency=afterok:job_id
##SBATCH --array=1-7
##SBATCH --account=siqiouya
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH --output=slurm_logs/slurm-%j.out

source /home/siqiouya/anaconda3/bin/activate sllama_lightning

# llm_model=meta-llama/Llama-2-7b-hf
llm_path=/compute/babel-4-1/siqiouya/llama-3.1-8b-hf
w2v2_path=/data/user_data/siqiouya/runs/pretrained/wav2_vec_vox_960h_pl.pt
# w2v2_path=/data/user_data/siqiouya/runs/pretrained/hubert_large_ll60k_finetune_ls960.pt
w2v2_type=w2v2
ctc_finetuned=True
# data_path=/scratch/xixu/dataset/must-c-v1.0/en-es
data_path=/compute/babel-6-17/xixu/datasets/must-c-v1.0/en-de
# data_path=/compute/babel-6-17/xixu/datasets/must-c-v1.0/en-fr
source_lang="English"
target_lang="German"
name="3.1-8B-s1-${source_lang,,}-${target_lang,,}-${w2v2_type}-rope-mt"
save_path=/compute/babel-5-23/siqiouya/runs/$name
rm -rf ${save_path}
mkdir -p ${save_path}

export PYTHONPATH=/home/siqiouya/work/sllama
export WANDB_PROJECT="mustc_1.0_de"
export WANDB_ENTITY="streamllama"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
SLURM_GPUS=8

cd /home/siqiouya/work/sllama/train
torchrun --nproc_per_node=$SLURM_GPUS --rdzv-endpoint=0.0.0.0:9105 \
    main.py \
    --stage 1 \
    \
    --w2v2_path ${w2v2_path} \
    --w2v2_type ${w2v2_type} \
    --ctc_finetuned ${ctc_finetuned} \
    --length_shrink_cfg "[(1024,2,2)] * 2" \
    --block_size 48 \
    --max_cache_size 500 \
    --text_weight 1.0 \
    \
    --llm_path ${llm_path} \
    --llm_freeze True \
    \
    --data_path ${data_path} \
    --data_split_train 'train' \
    --data_split_eval 'dev' \
    --source_lang "${source_lang}" \
    --target_lang "${target_lang}" \
    \
    --output_dir ${save_path} \
    --num_train_epochs  6 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --eval_steps 200 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 4 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.2 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --gradient_checkpointing True \
    --seed 998244353 \
    --report_to wandb \
    --run_name $name \
    --bf16 True \
    --deepspeed ../configs/deepspeed_config_bf16.json