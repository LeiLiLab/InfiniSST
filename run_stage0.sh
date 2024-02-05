#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --gpus=4
##SBATCH --constraint=xeon-4116 
#SBATCH --partition=gemini
#SBATCH --time=1-00:00:00
##SBATCH --dependency=afterok:job_id
##SBATCH --array=1-7
#SBATCH --account=siqiouyang
#SBATCH --mail-type=ALL
##SBATCH --mail-user=siqiouya@andrew.cmu.edu
##SBATCH --output=/home/xixu/slurm.txts

conda config --append envs_dirs /mnt/taurus/home/siqiouyang/anaconda3/envs/
source /mnt/taurus/home/siqiouyang/anaconda3/bin/activate /mnt/taurus/home/siqiouyang/anaconda3/envs/sllama

cd train

llm_model=/mnt/taurus/data/xixu/llm/llama-2-7b/hf
ssl_model=/mnt/taurus/data/xixu/models/wav2_vec_vox_960h_pl.pt
data_path=/mnt/taurus/data/xixu/datasets/must-c-v1.0/en-es
save_path=/mnt/taurus/data/siqiouyang/runs/sllama/en-es/7b/uni/stage0

# export WANDB_WATCH=all
export WANDB_PROJECT=en-es

export PYTHONPATH=/home/siqiouyang/work/projects/sllama
torchrun --nproc_per_node=$SLURM_GPUS --rdzv-endpoint=0.0.0.0:9101 \
    stage0.py \
    --model_name_or_path ${llm_model} \
    --speech_tower_path ${ssl_model} \
    --ssl_fintuned True \
    --data_path ${data_path} \
    --data_split_train 'train_mfa' \
    --data_split_eval 'dev_mfa' \
    --freeze_speech_foundation False \
    --freeze_backbone True \
    --only_tune_adapter False \
    --output_dir ${save_path} \
    --num_train_epochs 20 \
    --per_device_train_batch_size 25 \
    --per_device_eval_batch_size 25 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.2 \
    --lr_scheduler_type "cosine" \
    --logging_steps 100 \
    --gradient_checkpointing True \
    --seed 998244353 \
    --report_to wandb \
    --run_name 7b-uni-stage0 \
    --fp16 True \
    --deepspeed ../configs/deepspeed_config.json \
    --unidirectional True