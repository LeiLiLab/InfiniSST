#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --gpus=4
##SBATCH --constraint=xeon-4116 
#SBATCH --partition=taurus
#SBATCH --time=2-00:00:00
##SBATCH --dependency=afterok:job_id
##SBATCH --array=1-7
#SBATCH --account=xixu
#SBATCH --mail-type=ALL
##SBATCH --mail-user=xixu@andrew.cmu.edu
#SBATCH --output=s1_ctc.txt



cd /mnt/gemini/home/xixu/sllama_ctc/train

llm_model=/mnt/taurus/data/xixu/llm/llama-2-7b/hf
ssl_model=/mnt/taurus/data/xixu/models/wav2_vec_vox_960h_pl.pt
data_path=/mnt/taurus/data/xixu/datasets/must-c-v1.0/en-es
save_path=/mnt/taurus/data1/xixu/runs/sllama/en-es/7b/uni/stage1_ctc
stage0_path=/mnt/taurus/data1/xixu/runs/sllama/en-es/7b/uni/stage0

export WANDB_PROJECT=en-es

export PYTHONPATH=/mnt/gemini/home/xixu/sllama_ctc

torchrun --nproc_per_node=$SLURM_GPUS --rdzv-endpoint=0.0.0.0:9105 \
    /mnt/gemini/home/xixu/sllama_ctc/train/stage1.py \
    --model_name_or_path ${llm_model} \
    --speech_tower_path ${ssl_model} \
    --stage0 True \
    --stage0_path ${stage0_path} \
    --ssl_fintuned True \
    --data_path ${data_path} \
    --data_split_train 'train' \
    --data_split_eval 'dev' \
    --freeze_speech_foundation False \
    --freeze_backbone True \
    --only_tune_adapter True \
    --output_dir ${save_path} \
    --num_train_epochs 6 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.2 \
    --lr_scheduler_type "cosine" \
    --logging_steps 20 \
    --gradient_checkpointing True \
    --seed 998244353 \
    --report_to wandb \
    --run_name 7b-uni-stage1_ctc \
    --fp16 True \
    --deepspeed ../configs/deepspeed_config.json \
    --unidirectional True