#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --gres=gpu:A100_80GB:4
##SBATCH --nodelist=babel-3-17
##SBATCH --constraint=xeon-4116 
##SBATCH --partition=gemini
#SBATCH --time=2-00:00:00
##SBATCH --dependency=afterok:job_id
##SBATCH --array=1-7
#SBATCH --account=siqiouyang
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
##SBATCH --output=/home/xixu/slurm.txts

gpus=4

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

llm_model=/data/user_data/siqiouya/runs/pretrained/llama-2-7b/hf
ssl_model=/data/user_data/siqiouya/runs/pretrained/wav2_vec_vox_960h_pl.pt
stage0_dir=/data/user_data/siqiouya/runs/pretrained/speech_encoder_uni_waco
data_path=/scratch/siqiouya/dataset/must-c-v1.0/en-es
name=stage1-uni-waco-6epoch-warm0.2
save_path=/scratch/siqiouya/runs/$name

mkdir -p ${save_path}

# export WANDB_WATCH=all
export WANDB_PROJECT=en-es

export PYTHONPATH=/home/siqiouya/work/sllama
torchrun --nproc_per_node=$gpus --rdzv-endpoint=0.0.0.0:9105 \
    /home/siqiouya/work/sllama/train/stage1.py \
    --model_name_or_path ${llm_model} \
    --speech_tower_path ${ssl_model} \
    --stage0_ckpt_dir ${stage0_dir} \
    --ssl_fintuned True \
    --data_path ${data_path} \
    --data_split_train 'train' \
    --data_split_eval 'dev' \
    --freeze_backbone True \
    --only_tune_adapter True \
    --output_dir ${save_path} \
    --num_train_epochs 6 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 200 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.2 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --seed 998244353 \
    --report_to wandb \
    --run_name $name \
    --fp16 True \
    --deepspeed /home/siqiouya/work/sllama/configs/deepspeed_config.json \
    --unidirectional True 
    # --freeze_speech_foundation_except_pos_conv_steps 4000 \
    # --freeze_speech_foundation False \

/home/siqiouya/aws-i/v2/2.15.10/bin/aws s3 sync $save_path s3://must-c-v1.0/checkpoints/$name