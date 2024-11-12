#!/bin/bash
#SBATCH --job-name=llama3-8b
#SBATCH --output=./slurm-out/llama3-8b_de.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:6000Ada:8
#SBATCH --mem=256GB
#SBATCH --time 2-00:00:00
#SBATCH --mail-user=xixu@andrew.cmu.edu
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=general

cd train

source $HOME/sllama/bin/activate

# mkdir -p /scratch/xixu/

# rm -rf /scratch/xixu/*

# echo "Copying dataset."
# /usr/bin/cp -f /data/user_data/yuanjinw/dataset.tar.zst /scratch/xixu/
# echo "Dataset copied."

# echo "Extracting dataset."
# zstd --ultra -1 -d /scratch/xixu/dataset.tar.zst --stdout | tar axf - -C /scratch/xixu/
# # tar -axf /scratch/siqiouya/dataset.tar.zst -C /scratch/siqiouya/
# echo "Dataset extracted."

# llm_model=meta-llama/Llama-2-7b-hf
llm_model=meta-llama/Llama-3.1-8B
ssl_model=/data/user_data/yuanjinw/models/wav2_vec_vox_960h_pl.pt
# ssl_model=/data/user_data/yuanjinw/models/hubert_large_ll60k_finetune_ls960.pt
speech_encoder=w2v
# data_path=/scratch/xixu/dataset/must-c-v1.0/en-es
# data_path=/compute/babel-6-17/xixu/datasets/must-c-v1.0/en-de
data_path=/compute/babel-6-17/xixu/datasets/must-c-v1.0/en-fr
source_lang="English"
target_lang="French"
name="3.1-8B-s1-${source_lang,,}-${target_lang,,}"
save_path=/scratch/xixu/runs/$name
mkdir -p ${save_path}

export PYTHONPATH=/home/yuanjinw/work/sllama
export WANDB_PROJECT="mustc_1.0_fr"
export WANDB_ENTITY="streamllama"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
SLURM_GPUS=8

# python correct_path.py

export CUDA_VISIBLE_DEVICE=0,1,2,3,4,5,6,7


torchrun --nproc_per_node=$SLURM_GPUS --rdzv-endpoint=0.0.0.0:29503\
    stage1.py \
    --model_name_or_path ${llm_model} \
    --speech_tower_path ${ssl_model} \
    --speech_tower_type ${speech_encoder} \
    --ssl_fintuned True \
    --data_path ${data_path} \
    --data_split_train 'train' \
    --data_split_eval 'dev' \
    --source_lang "${source_lang}" \
    --target_lang "${target_lang}" \
    --freeze_speech_foundation False \
    --freeze_backbone True \
    --only_tune_adapter True \
    --output_dir ${save_path} \
    --num_train_epochs  6 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
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
    --fp16 True \
    --deepspeed ../configs/deepspeed_config.json \

# stage2 train
# llm_model=$save_path/checkpoint-6096
llm_model=/compute/babel-14-13/xixu/runs/${name}/checkpoint-6306
name="3.2-8B-s2-${source_lang,,}-${target_lang,,}"
save_path=/scratch/xixu/runs/$name

python ./zero_to_fp32.py ${llm_model} ${llm_model}/pytorch_model.bin


python ./extract_adapter.py \
  --model_name_or_path ${llm_model} \
  --extracted_name 'mm_length_adapter' \
  --output ${llm_model}/length_adapter.bin 
  
python ./extract_adapter.py \
  --model_name_or_path ${llm_model} \
  --extracted_name 'mm_mlp_adapter' \
  --output ${llm_model}/mlp_adapter.bin 

python ./extract_adapter.py \
    --model_name_or_path ${llm_model} \
    --extracted_name 'speech_tower' \
    --output ${llm_model}/speech_tower.bin

speech_encoder_path=${llm_model}/speech_tower.bin

torchrun  --nproc_per_node=$SLURM_GPUS --rdzv-endpoint=0.0.0.0:29503\
    stage2_large.py \
    --model_name_or_path ${llm_model} \
    --speech_tower_path ${speech_encoder_path} \
    --speech_tower_type ${speech_encoder} \
    --ssl_fintuned True \
    --data_path ${data_path} \
    --data_split_train 'train' \
    --data_split_eval 'dev' \
    --source_lang "${source_lang}" \
    --target_lang "${target_lang}" \
    --freeze_speech_foundation False \
    --freeze_backbone False \
    --only_tune_adapter False \
    --output_dir ${save_path} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 200 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 3 \
    --learning_rate 7e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.2 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --gradient_checkpointing True \
    --seed 998244353 \
    --report_to wandb \
    --run_name $name \
    --fp16 True \
    --deepspeed ../configs/deepspeed_config_stage3.json \