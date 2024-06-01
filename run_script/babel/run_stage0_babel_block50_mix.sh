#!/usr/bin/env bash

gpus=4

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

export PYTHONPATH=/home/yuanjinw/work/sllama
srun python /home/yuanjinw/work/sllama/train/stage0.py \
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