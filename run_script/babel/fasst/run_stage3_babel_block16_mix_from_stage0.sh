#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512GB
#SBATCH --gres=gpu:A6000:8
##SBATCH --constraint=xeon-4116 
##SBATCH --partition=gemini
#SBATCH --time=1-00:00:00
##SBATCH --dependency=afterok:job_id
##SBATCH --array=1-7
#SBATCH --account=siqiouyang
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
##SBATCH --output=slurm_stage2_A100_40G.txt

gpus=8

# conda config --append envs_dirs /mnt/taurus/home/siqiouyang/anaconda3/envs/
# source /mnt/taurus/home/siqiouyang/anaconda3/bin/activate /mnt/taurus/home/siqiouyang/anaconda3/envs/sllama

source /home/siqiouya/anaconda3/bin/activate sllama_lightning

mkdir -p /scratch/siqiouya/

echo "Copying dataset."
/usr/bin/cp -f /data/user_data/siqiouya/dataset.tar.zst /scratch/siqiouya/
echo "Dataset copied."

echo "Extracting dataset."
zstd --ultra -1 -d /scratch/siqiouya/dataset.tar.zst --stdout | tar axf - -C /scratch/siqiouya/
# tar -axf /scratch/siqiouya/dataset.tar.zst -C /scratch/siqiouya/
echo "Dataset extracted."

cd train

llm_model=/data/user_data/siqiouya/runs/pretrained/llama-2-7b/hf
speech_model=/data/user_data/siqiouya/runs/speech_encoder_uni_waco_block16_mix
data_path=/scratch/siqiouya/dataset/must-c-v1.0/en-es
name=stage3-uni-waco-word-block16-fixed-mix-from-stage0-2epoch
save_path=/scratch/siqiouya/runs/$name

mkdir -p $save_path

# export WANDB_WATCH=all
export WANDB_PROJECT=en-es

export PYTHONPATH=/home/siqiouya/work/sllama
torchrun  --nproc_per_node=$gpus --rdzv-endpoint=0.0.0.0:9106 \
    /home/siqiouya/work/sllama/train/stage3_large_uni_word_from_stage0.py \
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
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 200 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --seed 998244353 \
    --report_to wandb \
    --run_name $name \
    --fp16 True \
    --deepspeed /home/siqiouya/work/sllama/configs/deepspeed_config_stage3.json \
    --unidirectional True \
    --blocksize 16

/home/siqiouya/aws-i/v2/2.15.10/bin/aws s3 sync $save_path s3://must-c-v1.0/checkpoints/$name