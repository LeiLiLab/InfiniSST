#!/usr/bin/env bash

##SBATCH --nodelist=babel-4-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=256GB
#SBATCH --gres=gpu:A6000:4
##SBATCH --nodelist=babel-3-17
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
##SBATCH --dependency=afterok:job_id
##SBATCH --array=1-7
##SBATCH --account=siqiouya
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH --output=slurm_logs/slurm-%j.out

source /home/siqiouya/anaconda3/bin/activate sllama_lightning

llm_model=/compute/babel-4-1/siqiouya/llama-3.1-8b-hf
data_path=/data/user_data/siqiouya/dataset/must-c-v1.0/en-de
name=crtl-stage0-cache125-postshrink
save_path=/scratch/siqiouya/runs/$name

mkdir -p ${save_path}

# export WANDB_WATCH=all
export WANDB_PROJECT=mustc_1.0_de
export NCCL_DEBUG=INFO

export PYTHONPATH=/home/siqiouya/work/sllama
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
srun python /home/siqiouya/work/sllama/train/stage0_new.py \
    --feature-extractor-cfg \
    "[(1024, 10, 5)] + [(1024, 3, 2)] * 4 + [(1024,2,2)] * 2" \
    --length-shrink-cfg \
    "[(1024,2,2)] * 2" \
    --block-size 48 \
    --max-cache-size 500 \
    --llm-path ${llm_model} \
    --lr 1e-4 \
    --warmup-updates 25000 \
    --temp 0.2 \
    \
    --strategy ddp \
    --device-type gpu \
    --n-device 4 \
    --max-steps 500000 \
    --save-dir $save_path \
    --precision bf16-mixed \
    --wandb-run-name $name \
    --eval-step 1000 \
    --log-step 100 \
    --grad-acc-steps 4 \
    --clip-norm 10.0 \
    \
    --data-path $data_path \
    --train-split train_st_de_mfa_llama3 \
    --dev-split dev_st_de_mfa_llama3 \
    --train-batch-size 1000000 \
    --dev-batch-size 1000000
