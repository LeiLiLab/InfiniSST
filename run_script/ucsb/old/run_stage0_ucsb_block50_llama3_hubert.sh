#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=256GB
#SBATCH --gpus=4
##SBATCH --constraint=xeon-4116 
#SBATCH --partition=aries
#SBATCH --time=1-00:00:00
##SBATCH --dependency=afterok:job_id
##SBATCH --array=1-7
#SBATCH --account=siqiouyang
#SBATCH --mail-type=ALL
##SBATCH --mail-user=siqiouya@andrew.cmu.edu
##SBATCH --output=/home/xixu/slurm.txts

conda config --append envs_dirs /mnt/taurus/home/siqiouyang/anaconda3/envs/
source /mnt/taurus/home/siqiouyang/anaconda3/bin/activate /mnt/taurus/home/siqiouyang/anaconda3/envs/sllama_lightning

llm_model=/mnt/taurus/data/siqiouyang/download/llama3.1-8b-hf/
ssl_model=/mnt/taurus/data/siqiouyang/download/hubert_large_ll60k_finetune_ls960.pt
data_path=/mnt/aries/data/siqiouyang/datasets/must-c-v1.0
name=fasst-stage0-block50-hubert
save_path=/mnt/taurus/data/siqiouyang/runs/sllama/en-de/$name

mkdir -p ${save_path}

# export WANDB_WATCH=all
# export WANDB_PROJECT=en-fr
export WANDB_PROJECT=mustc_1.0_de
export NCCL_DEBUG=INFO

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONPATH=/home/siqiouyang/work/projects/sllama
srun python /home/siqiouyang/work/projects/sllama/train/stage0.py \
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
    --n-device $SLURM_GPUS_ON_NODE \
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