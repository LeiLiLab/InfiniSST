#!/usr/bin/env bash

##SBATCH --nodelist=babel-4-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=512GB
#SBATCH --gres=gpu:L40S:8
##SBATCH --nodelist=babel-3-17
#SBATCH --exclude=babel-3-[5,9,13,17],babel-4-[9,29],babel-6-29,babel-7-[1,5,9],babel-8-[9,13],babel-10-13,babel-11-25,babel-13-[13,29]
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
##SBATCH --dependency=afterok:job_id
##SBATCH --array=1-7
##SBATCH --account=siqiouya
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH -e slurm_logs/%j.err
#SBATCH -o slurm_logs/%j.out

source /home/siqiouya/anaconda3/bin/activate speechllama2

llm_path=/compute/babel-4-1/siqiouya/llama-3.1-8b-instruct-hf
sllm_weight_path=/compute/babel-5-23/siqiouya/runs/8B-traj-s1-v2.0/epoch\=5-step\=7128.ckpt/
w2v2_path=/data/user_data/siqiouya/runs/pretrained/wav2_vec_vox_960h_pl.pt
# w2v2_path=/data/user_data/siqiouya/runs/pretrained/hubert_large_ll60k_finetune_ls960.pt
w2v2_type=w2v2
ctc_finetuned=True
# data_path=/scratch/xixu/dataset/must-c-v1.0/en-es
# data_path=/compute/babel-6-17/xixu/datasets/must-c-v1.0/en-de
# data_path=/compute/babel-6-17/xixu/datasets/must-c-v1.0/en-fr
data_path=/compute/babel-6-17/xixu/datasets/must-c-v2.0/en-zh

source_lang="English"
target_lang="Chinese"
name="8B-traj-s2-v2.0"
save_path=/compute/babel-5-23/siqiouya/runs/$name
rm -rf ${save_path}
mkdir -p ${save_path}

export PYTHONPATH=/home/siqiouya/work/sllama
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="mustc_1.0_zh"
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
    --data_split_train 'comet_0.50_traj' \
    --data_split_eval 'dev_traj' \
    --source_lang "${source_lang}" \
    --target_lang "${target_lang}" \
    --trajectory 3 \
    \
    --seed 998244353 \
    --stage 2 \
    --train_bsz 1200 \
    --eval_bsz 1200 \
    --learning_rate 7e-6 \
    --warmup_steps 100 \
    --run_name $name \
    \
    --n_device ${SLURM_GPUS} \
    --deepspeed_stage 2 \
    --max_epochs 1 \
    --grad_acc_steps 6 \
    --clip_norm 1.0 \
    --save_dir ${save_path} \
    --log_step 5 \
    --eval_step 200