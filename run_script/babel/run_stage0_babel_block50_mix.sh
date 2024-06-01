#!/usr/bin/env bash

##SBATCH --nodelist=babel-4-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=256GB
#SBATCH --gres=gpu:A6000:4
##SBATCH --nodelist=babel-3-17
##SBATCH --constraint=xeon-4116 
##SBATCH --partition=gemini
#SBATCH --time=1-00:00:00
##SBATCH --dependency=afterok:job_id
##SBATCH --array=1-7
#SBATCH --account=siqiouyang
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
##SBATCH --output=/home/xixu/slurm.txts

gpus=4

source /home/siqiouya/anaconda3/bin/activate sllama_lightning
source $HOME/sllama/bin/activate

mkdir -p /scratch/siqiouya/

rm -rf /scratch/siqiouya/*

echo "Copying dataset."
/usr/bin/cp -f /data/user_data/siqiouya/dataset.tar.zst /scratch/siqiouya/
echo "Dataset copied."

echo "Extracting dataset."
zstd --ultra -1 -d /scratch/siqiouya/dataset.tar.zst --stdout | tar axf - -C /scratch/siqiouya/
# tar -axf /scratch/siqiouya/dataset.tar.zst -C /scratch/siqiouya/
echo "Dataset extracted."

llm_model=/data/user_data/siqiouya/runs/pretrained/llama-2-7b/hf
ssl_model=/data/user_data/siqiouya/runs/pretrained/wav2_vec_vox_960h_pl.pt
data_path=/scratch/siqiouya/dataset/must-c-v1.0/en-es
name=stage0-block50-mix
save_path=/scratch/siqiouya/runs/$name

mkdir -p ${save_path}

# export WANDB_WATCH=all
export WANDB_PROJECT=en-es
export NCCL_DEBUG=INFO

export PYTHONPATH=/home/siqiouya/work/sllama
srun python /home/siqiouya/work/sllama/train/stage0.py \
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