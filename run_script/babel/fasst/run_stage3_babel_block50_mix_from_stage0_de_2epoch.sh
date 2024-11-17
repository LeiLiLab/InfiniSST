#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --gres=gpu:L40S:4
##SBATCH --constraint=xeon-4116 
#SBATCH --partition=long
#SBATCH --time=1-12:00:00
##SBATCH --dependency=afterok:job_id
##SBATCH --array=1-7
#SBATCH --account=siqiouyang
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
##SBATCH --output=slurm_stage2_A100_40G.txt

gpus=4

# conda config --append envs_dirs /mnt/taurus/home/siqiouyang/anaconda3/envs/
# source /mnt/taurus/home/siqiouyang/anaconda3/bin/activate /mnt/taurus/home/siqiouyang/anaconda3/envs/sllama

source /home/siqiouya/anaconda3/bin/activate sllama_lightning

mkdir -p /scratch/siqiouya/

# echo "Copying dataset."
# /usr/bin/cp -f /data/user_data/siqiouya/dataset.tar.zst /scratch/siqiouya/
# echo "Dataset copied."

echo "Downloading dataset."
/home/siqiouya/aws-i/v2/2.15.10/bin/aws s3 cp s3://must-c-v1.0/en-de.tar.zst /scratch/siqiouya/dataset/must-c-v1.0/en-de.tar.zst
echo "Dataset downloaded."

echo "Extracting dataset."
zstd --ultra -1 -d /scratch/siqiouya/dataset/must-c-v1.0/en-de.tar.zst --stdout | tar axf - -C /scratch/siqiouya/dataset/must-c-v1.0/
sed -i s#/mnt/taurus/data/xixu/datasets/must-c-v1.0/en-de/data#/scratch/siqiouya/dataset/must-c-v1.0/en-de/data#g /scratch/siqiouya/dataset/must-c-v1.0/en-de/*.tsv
rm /scratch/siqiouya/dataset/must-c-v1.0/en-de.tar.zst
# tar -axf /scratch/siqiouya/dataset.tar.zst -C /scratch/siqiouya/
echo "Dataset extracted."

llm_model=/data/user_data/siqiouya/runs/pretrained/llama-2-7b/hf
speech_model=/data/user_data/siqiouya/runs/en_de_stage0_waco_block50_mix
data_path=/scratch/siqiouya/dataset/must-c-v1.0/en-de
name=stage3-uni-waco-word-block50-fixed-mix-from-stage0-2epoch
save_path=/scratch/siqiouya/runs/$name

rm -rf $save_path
mkdir -p $save_path

# export WANDB_WATCH=all
export WANDB_PROJECT=en-de

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
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 11 \
    --evaluation_strategy "steps" \
    --eval_steps 200 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.2 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --seed 998244353 \
    --report_to wandb \
    --run_name $name \
    --bf16 True \
    --deepspeed /home/siqiouya/work/sllama/configs/deepspeed_config_stage3_bf16.json \
    --unidirectional True \
    --blocksize 50 \
    --n_word_per_input 3

/home/siqiouya/aws-i/v2/2.15.10/bin/aws s3 sync $save_path s3://must-c-v1.0/checkpoints/en-de/$name