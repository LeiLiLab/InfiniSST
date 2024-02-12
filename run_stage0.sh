#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --gres=gpu:A6000:4
##SBATCH --constraint=xeon-4116 
##SBATCH --partition=gemini
#SBATCH --time=2-00:00:00
##SBATCH --dependency=afterok:job_id
##SBATCH --array=1-7
#SBATCH --account=siqiouyang
#SBATCH --mail-type=ALL
##SBATCH --mail-user=siqiouya@andrew.cmu.edu
##SBATCH --output=/home/xixu/slurm.txts

gpus=4
source /home/siqiouya/anaconda3/bin/activate sllama

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
data_path=/scratch/siqiouya/dataset/must-c-v1.0/en-es
name=stage0-uni
save_path=/scratch/siqiouya/runs/$name

mkdir -p ${save_path}

# export WANDB_WATCH=all
export WANDB_PROJECT=en-es

export PYTHONPATH=/home/siqiouya/work/sllama
python /home/siqiouya/work/sllama/train/stage0.py \
    --llm-path ${llm_model} \
    --speech-encoder-path ${ssl_model} \ 
    --ssl-finetuned \
    --unidirectional \
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
    --log-step 100 \
    --grad-acc-steps 4 \
    --clip-norm 10.0 \
    \
    --data-path $data_path \
    --train-split train_mfa \
    --dev-split dev_mfa \
    --train-batch-size 1000000 \
    --dev-batch-size 1000000

/home/siqiouya/aws-i/v2/2.15.10/bin/aws s3 sync $save_path s3://must-c-v1.0/checkpoints/$name