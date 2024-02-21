#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --gres=gpu:A6000:4
##SBATCH --constraint=xeon-4116 
##SBATCH --partition=taurus
#SBATCH --time=0-10:00:00
##SBATCH --dependency=afterok:job_id
##SBATCH --array=1-7
#SBATCH --account=siqiouya
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
##SBATCH --output=stdout_sevl.txt
##SBATCH --error=stderr_sevl.txt

cd train

llm_model=/data/user_data/siqiouya/runs/checkpoint-2000-uni
ssl_model=/data/user_data/siqiouya/runs/pretrained/wav2_vec_vox_960h_pl.pt
data_path=/data/user_data/siqiouya/dataset/must-c-v1.0/en-es
save_path=/scratch/siqiouya/stage3-bi

mkdir -p ${save_path}

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

export PYTHONPATH=/home/siqiouya/work/sllama

batch_size_multiplier=40
gpus=4

torchrun --rdzv-endpoint=0.0.0.0:9000 --nproc_per_node=$gpus \
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
    --gradient_accumulation_steps $((batch_size_multiplier / gpus)) \
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
    --deepspeed ../configs/deepspeed_config_stage3.json # \
    # --unidirectional True \

/home/siqiouya/aws-i/v2/2.15.10/bin/aws s3 sync /scratch/siqiouya/stage3-bi s3://must-c-v1.0/checkpoints/stage3-bi