#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=256GB
#SBATCH --gpus=4
##SBATCH --constraint=xeon-4116 
#SBATCH --partition=aries
#SBATCH --time=7-00:00:00
##SBATCH --dependency=afterok:job_id
##SBATCH --array=1-7
#SBATCH --account=siqiouyang
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH --output=slurm_logs/slurm-%j.out

conda config --append envs_dirs /mnt/taurus/home/siqiouyang/anaconda3/envs/
source /mnt/taurus/home/siqiouyang/anaconda3/bin/activate /mnt/taurus/home/siqiouyang/anaconda3/envs/sllama_lightning

llm_model=/mnt/taurus/data/siqiouyang/download/llama3.1-8b-hf/
w2v2_path=/mnt/taurus/data/siqiouyang/download/hubert_large_ll60k_finetune_ls960.pt
data_path=/mnt/aries/data/siqiouyang/datasets/must-c-v1.0
name=crtl-stage0-hubert-cache10s
save_path=/mnt/taurus/data/siqiouyang/runs/sllama/en-de/$name

mkdir -p ${save_path}

# export WANDB_WATCH=all
export WANDB_PROJECT=mustc_1.0_de
export NCCL_DEBUG=INFO

export PYTHONPATH=/home/siqiouyang/work/projects/sllama
srun python /home/siqiouyang/work/projects/sllama/train/stage0_w2v2rope.py \
    --w2v2-path ${w2v2_path} \
    --hubert \
    --length-shrink-cfg "[(1024,2,2)] * 2" \
    --block-size 48 \
    --max-cache-size 500 \
    --llm-path ${llm_model} \
    --lr 1e-4 \
    --warmup-updates 25000 \
    --temp 0.2 \
    \
    --strategy ddp_find_unused_parameters_true \
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
