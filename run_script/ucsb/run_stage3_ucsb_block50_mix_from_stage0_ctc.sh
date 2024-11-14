#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --gpus=4
##SBATCH --constraint=xeon-4116 
#SBATCH --partition=taurus
#SBATCH --time=1-00:00:00
##SBATCH --dependency=afterok:job_id
##SBATCH --array=1-7
#SBATCH --account=siqiouyang
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu


conda config --append envs_dirs /mnt/taurus/home/siqiouyang/anaconda3/envs/
source /mnt/taurus/home/siqiouyang/anaconda3/bin/activate /mnt/taurus/home/siqiouyang/anaconda3/envs/sllama_lightning


llm_model=/mnt/taurus/data/xixu/llm/llama-2-7b/hf
speech_model=/mnt/data1/siqiouyang/runs/sllama/en-es/7b/uni/stage0-block50-mix-ctc
data_path=/mnt/taurus/data/xixu/datasets/must-c-v1.0/en-es
name=stage3-uni-ctc-block50-mix-from-stage0
save_path=/mnt/taurus/data1/siqiouyang/runs/sllama/en-es/7b/uni/$name

mkdir -p ${save_path}
rm -rf ${save_path}/*

# export WANDB_WATCH=all
export WANDB_PROJECT=en-es

export PYTHONPATH=/home/siqiouyang/work/projects/sllama
torchrun  --nproc_per_node=$SLURM_GPUS_ON_NODE --rdzv-endpoint=0.0.0.0:9106 \
    /home/siqiouyang/work/projects/sllama/train/stage3_large_uni_word_from_stage0.py \
    --model_name_or_path ${llm_model} \
    --speech_tower_path ${speech_model} \
    --ssl_fintuned True \
    --data_path ${data_path} \
    --data_split_train 'train_mfa_30s_mix_filtered' \
    --data_split_eval 'dev_mfa_30s_mix_filtered' \
    --freeze_speech_foundation False \
    --freeze_backbone False \
    --only_tune_adapter False \
    --output_dir ${save_path} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 200 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.2 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --seed 998244353 \
    --report_to wandb \
    --run_name $name \
    --fp16 True \
    --deepspeed /home/siqiouyang/work/projects/sllama/configs/deepspeed_config_stage3.json \
    --unidirectional True \
    --blocksize 50