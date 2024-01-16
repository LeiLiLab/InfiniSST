#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --gpus=3
##SBATCH --constraint=xeon-4116 
#SBATCH --partition=taurus
#SBATCH --time=1-2:34:56 
##SBATCH --dependency=afterok:job_id
##SBATCH --array=1-7
#SBATCH --account=siqiouyang
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
##SBATCH --output=stdout_sevl.txt
##SBATCH --error=stderr_sevl.txt

cd train

llm_model=/mnt/taurus/data/xixu/runs/sllama/en-es/7b/run3/stage2/checkpoint-3600
ssl_model=/mnt/taurus/data/xixu/models/wav2_vec_vox_960h_pl.pt
data_path=/mnt/taurus/data/xixu/datasets/must-c-v1.0/en-es
save_path=/mnt/taurus/data/siqiouyang/runs/sllama/en-es/7b/run2/stage3-fix

# python ./zero_to_fp32.py ${llm_model}/checkpoint-12000 ${llm_model}/checkpoint-12000/pytorch_model.bin

# model_path=/mnt/data/xixu/runs/sllama/en-es/13b/stage1
# python ./extract_adapter.py \
#   --model_name_or_path ${model_path}/checkpoint-2100 \
#   --extracted_name 'mm_length_adapter' \
#   --output ${model_path}/length_adapter.bin 
# python ./extract_adapter.py \
#   --model_name_or_path ${model_path}/checkpoint-2100 \
#   --extracted_name 'mm_mlp_adapter' \
#   --output ${model_path}/mlp_adapter.bin 

export PYTHONPATH=/mnt/taurus/home/siqiouyang/work/projects/sllama

batch_size_multiplier=36

torchrun  --nproc_per_node=$SLURM_GPUS\
    stage3_large.py \
    --model_name_or_path ${llm_model} \
    --speech_tower_path ${ssl_model} \
    --ssl_fintuned True \
    --data_path ${data_path} \
    --data_split_train 'train' \
    --data_split_eval 'dev' \
    --freeze_speech_foundation True \
    --freeze_backbone False \
    --only_tune_adapter False \
    --output_dir ${save_path} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $((batch_size_multiplier / SLURM_GPUS)) \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --seed 1234 \
    --report_to none \
    --fp16 True \
    --deepspeed ../configs/deepspeed_config_stage3.json

