#!/usr/bin/env bash

##SBATCH --nodelist=babel-4-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=512GB
#SBATCH --gres=gpu:L40S:8
##SBATCH --nodelist=babel-3-17
#SBATCH --exclude=babel-13-13,babel-13-29,babel-4-9,babel-3-5
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
##SBATCH --dependency=afterok:job_id
##SBATCH --array=1-7
##SBATCH --account=siqiouya
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH -e slurm_logs/%j.err
#SBATCH -o slurm_logs/%j.out

source /home/siqiouya/anaconda3/bin/activate speechllama

llm_path=/compute/babel-4-1/siqiouya/llama-3.1-8b-hf
sllm_weight_path=/compute/babel-5-23/siqiouya/runs/3.1-8B-s1-lightning-german-w2v2-rope-noxpos-cosine/
w2v2_path=/data/user_data/siqiouya/runs/pretrained/wav2_vec_vox_960h_pl.pt
# w2v2_path=/data/user_data/siqiouya/runs/pretrained/hubert_large_ll60k_finetune_ls960.pt
w2v2_type=w2v2
ctc_finetuned=True
# data_path=/scratch/xixu/dataset/must-c-v1.0/en-es
# data_path=/compute/babel-6-17/xixu/datasets/must-c-v1.0/en-de
# data_path=/compute/babel-6-17/xixu/datasets/must-c-v1.0/en-fr

mkdir -p /scratch/siqiouya/
rsync -r /compute/babel-6-17/xixu/datasets/must-c-v1.0/backup/en-de /scratch/siqiouya/
data_path=/scratch/siqiouya/en-de

source_lang="English"
target_lang="German"
name="3.1-8B-s2-lightning-${target_lang,,}-${w2v2_type}-rope-noxpos-cosine"
save_path=/compute/babel-5-23/siqiouya/runs/$name
rm -rf ${save_path}
mkdir -p ${save_path}

export PYTHONPATH=/home/siqiouya/work/sllama
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="mustc_1.0_de"
export WANDB_ENTITY="streamllama"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
SLURM_GPUS=8

python /home/siqiouya/work/sllama/train/zero_to_fp32.py ${sllm_weight_path} ${sllm_weight_path}/pytorch_model.bin
python /home/siqiouya/work/sllama/train/prune_bin.py ${sllm_weight_path}/pytorch_model.bin

srun python /home/siqiouya/work/sllama/train/main_lightning.py \
    \
    --w2v2_path ${w2v2_path} \
    --w2v2_type ${w2v2_type} \
    --w2v2_freeze True \
    --ctc_finetuned ${ctc_finetuned} \
    --length_shrink_cfg "[(1024,2,2)] * 2" \
    --block_size 48 \
    --max_cache_size 500 \
    --xpos False \
    \
    --llm_path ${llm_path} \
    --sllm_weight_path ${sllm_weight_path}/pytorch_model.bin \
    \
    --data_path ${data_path} \
    --data_split_train 'train' \
    --data_split_eval 'dev' \
    --source_lang "${source_lang}" \
    --target_lang "${target_lang}" \
    \
    --seed 998244353 \
    --stage 2 \
    --train_bsz 800000 \
    --eval_bsz 800000 \
    --learning_rate 7e-6 \
    --warmup_steps 100 \
    --run_name $name \
    \
    --n_device ${SLURM_GPUS} \
    --deepspeed_stage 2 \
    --max_epochs 1 \
    --grad_acc_steps 4 \
    --clip_norm 1.0 \
    --save_dir ${save_path} \
    --log_step 5 \
    --eval_step 200