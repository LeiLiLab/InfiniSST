#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=256GB
#SBATCH --gpus=4
##SBATCH --constraint=xeon-4116 
#SBATCH --partition=taurus
#SBATCH --time=1-00:00:00
##SBATCH --dependency=afterok:job_id
##SBATCH --array=1-7
#SBATCH --account=siqiouyang
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
##SBATCH --output=/home/xixu/slurm.txts

conda config --append envs_dirs /mnt/taurus/home/siqiouyang/anaconda3/envs/
source /mnt/taurus/home/siqiouyang/anaconda3/bin/activate /mnt/taurus/home/siqiouyang/anaconda3/envs/sllama_lightning

llm_model=/mnt/taurus/data/xixu/llm/llama-2-7b/hf
ssl_model=/mnt/taurus/data/xixu/models/wav2_vec_vox_960h_pl.pt
data_path=/mnt/taurus/data/xixu/datasets/must-c-v1.0/en-de
name=stage0-block50-mix
save_path=/mnt/taurus/data1/siqiouyang/runs/sllama/en-de/7b/uni/$name

mkdir -p ${save_path}
rm -rf ${save_path}/*

# export WANDB_WATCH=all
export WANDB_PROJECT=en-de
export NCCL_DEBUG=INFO

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
    --loss-fn waco \
    \
    --strategy ddp \
    --device-type gpu \
    --n-device $SLURM_GPUS_ON_NODE \
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