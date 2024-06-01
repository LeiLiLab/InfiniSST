#!/usr/bin/env bash

cd train
# llm_model='path to llama2'

# PYTHONPATH=/home/yuanjinw/work/sllama_gemma \
# python extract_embedding.py \
#   --model_name_or_path ${llm_model}

gpus=8

source $HOME/sllama/bin/activate

mkdir -p /scratch/xixu/

rm -rf /scratch/xixu/*

echo "Copying dataset."
/usr/bin/cp -f /data/user_data/yuanjinw/dataset.tar.zst /scratch/xixu/
echo "Dataset copied."

echo "Extracting dataset."
zstd --ultra -1 -d /scratch/xixu/dataset.tar.zst --stdout | tar axf - -C /scratch/xixu/
# tar -axf /scratch/siqiouya/dataset.tar.zst -C /scratch/siqiouya/
echo "Dataset extracted."

llm_model=google/gemma-7b
ssl_model=/data/user_data/yuanjinw/models/wav2_vec_vox_960h_pl.pt
data_path=/scratch/xixu/dataset/must-c-v1.0/en-es
name=stage0-block50-mix
save_path=/scratch/xixu/runs/$name

mkdir -p ${save_path}

# export WANDB_WATCH=all
export WANDB_PROJECT=llm-encoder
export NCCL_DEBUG=INFO

export PYTHONPATH=/home/yuanjinw/work/sllama_gemma
export CUDA_VISIBLE_DEVICE=0,1,2,3,4,5,6,7

python correct_path.py

torchrun --nproc_per_node=$gpus \
    /home/yuanjinw/work/sllama_gemma/train/stage0.py \
    --llm-path ${llm_model} \
    --speech-encoder-path ${ssl_model} \
    --ssl-finetuned \
    --unidirectional \
    --blocksize 50 \
    --temp 0.2 \
    --lr 1e-4 \
    --warmup-updates 25000 \
    \
    --strategy ddp \
    --device-type gpu \
    --n-device $gpus \
    --max-steps 500000 \
    --save-dir $save_path \
    --precision 16-mixed \
    --wandb-run-name $name \
    --eval-step 1000 \
    --save-step 250 \
    --log-step 100 \
    --grad-acc-steps 4 \
    --clip-norm 10.0 \
    \
    --data-path $data_path \
    --train-split train_mfa_30s_mix_filtered \
    --dev-split dev_mfa_30s_mix_filtered \
    --train-batch-size 1000000 \
    --dev-batch-size 1000000

## Before Stage3 training
speech_encoder_dir = $save_path
speech_encoder_name = $name

python ./extract_adapter.py \
  --model_name_or_path ${speech_encoder_dir}/${speech_encoder_name} \
  --speech-encoder-only \
  --extracted_name 'mm_length_adapter' \
  --output ${speech_encoder_dir}/length_adapter.bin 

python ./extract_adapter.py \
  --model_name_or_path ${speech_encoder_dir}/${speech_encoder_name} \
  --speech-encoder-only \
  --extracted_name 'mm_mlp_adapter' \
  --output ${speech_encoder_dir}/mlp_adapter.bin 

python ./extract_adapter.py \
  --model_name_or_path ${speech_encoder_dir}/${speech_encoder_name} \
  --speech-encoder-only \
  --extracted_name 'speech_tower' \
  --output ${speech_encoder_dir}/speech_tower.bin 

## Stage 3 from stage 0

speech_model=/scratch/xixu/runs/speech_encoder_uni_waco_block50_mix
data_path=/scratch/xixu/dataset/must-c-v1.0/en-es
name=stage3-uni-waco-word-block50-fixed-mix-from-stage0
save_path=/scratch/xixu/runs/$name

rm -rf $save_path
mkdir -p $save_path

# export WANDB_WATCH=all
export WANDB_PROJECT=llm-encoder

export PYTHONPATH=/home/yuanjinw/work/sllama_gemma
torchrun  --nproc_per_node=$gpus --rdzv-endpoint=0.0.0.0:9106 \
    /home/yuanjinw/work/sllama_gemma/train/stage3_large_uni_word_from_stage0.py \
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
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 200 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 4 \
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
    --deepspeed /home/yuanjinw/work/sllama_gemma/configs/deepspeed_config_stage3.json \
    --unidirectional True \
    --blocksize 50 \
    --n_word_per_input 3