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
#SBATCH --account=xixu
#SBATCH --mail-type=ALL
##SBATCH --mail-user=xixu@andrew.cmu.edu
#SBATCH --output=s0_bi.txts


llm_model=/mnt/taurus/data/xixu/llm/llama-2-7b/hf
ssl_model=/mnt/taurus/data/xixu/models/wav2_vec_vox_960h_pl.pt
data_path=/mnt/taurus/data/xixu/datasets/must-c-v1.0/en-es
name=stage0_bi
save_path=/mnt/taurus/data1/xixu/runs/sllama/en-es/7b/bi/$name

mkdir -p ${save_path}

# export WANDB_WATCH=all
export WANDB_PROJECT=en-es
export NCCL_DEBUG=INFO

export PYTHONPATH=/mnt/gemini/home/xixu/sllama_ctc

srun python /mnt/gemini/home/xixu/sllama_ctc/train/stage0.py \
    --llm-path ${llm_model} \
    --speech-encoder-path ${ssl_model} \
    --ssl-finetuned \
    --temp 0.2 \
    --lr 1e-4 \
    --warmup-updates 25000 \
    --strategy ddp \
    --device-type gpu \
    --n-device $SLURM_GPUS_ON_NODE \
    --max-steps 500000 \
    --save-dir $save_path \
    --precision 16-mixed \
    --wandb-run-name $name \
    --eval-step 1000 \
    --log-step 100 \
    --grad-acc-steps 4 \
    --clip-norm 10.0 \
    --data-path $data_path \
    --train-split train_mfa \
    --dev-split dev_mfa \
    --train-batch-size 1000000 \
    --dev-batch-size 1000000