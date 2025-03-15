#!/usr/bin/env bash

##SBATCH --nodelist=babel-4-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=500GB
#SBATCH --gres=gpu:L40S:8
##SBATCH --nodelist=babel-3-17
#SBATCH --exclude=babel-3-[5,9,13,17],babel-4-[5,9,29],babel-6-29,babel-7-[1,5,9],babel-8-[5,9,13],babel-10-[5,9,13],babel-11-25,babel-12-29,babel-13-[1,13,21,29],babel-14-25
#SBATCH --partition=preempt
#SBATCH --time=2-00:00:00
##SBATCH --dependency=afterok:job_id
##SBATCH --array=1-7
##SBATCH --account=siqiouya
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH -e slurm_logs/%j.err
#SBATCH -o slurm_logs/%j.out

source /home/siqiouya/anaconda3/bin/activate infinisst

llama_path=/compute/babel-4-1/siqiouya/llama-3.1-8b-instruct-hf

mimi_path=/compute/babel-4-1/siqiouya/mimi

ROOT=/compute/babel-14-5/siqiouya
lang_code=zh
lang=Chinese
data_path=$ROOT/en-${lang_code}/

save_dir=/compute/babel-5-23/siqiouya/runs/en-zh/

source_lang="English"
target_lang=${lang} # e.g. German
name="stage1_mimi_6epoch_cosine"
save_path=${save_dir}/${name}
rm -rf ${save_path} # comment this line if you want to resume training
mkdir -p ${save_path}

export PYTHONPATH=$PYTHONPATH:$PWD
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="mustc_${lang_code}"
export WANDB_ENTITY="streamllama"

# disable P2P and InfiniBand for L40S 8-GPU nodes
# if your node supports P2P and InfiniBand, you need to remove these two lines
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
SLURM_GPUS=8

srun python train/main.py \
    \
    --mimi_path ${mimi_path} \
    \
    --block_size 24 \
    \
    --model_type mimi_llama31 \
    --llm_path ${llama_path} \
    --llm_freeze True \
    --llm_emb_freeze True \
    --llm_head_freeze True \
    --use_flash_attn True \
    \
    --data_path ${data_path} \
    --data_split_train 'train_nospeaker_traj_30_filtered' \
    --data_split_eval 'dev_nospeaker_traj_30_filtered' \
    --source_lang "${source_lang}" \
    --target_lang "${target_lang}" \
    --trajectory 9 \
    --trajectory_max_multiplier 4 \
    --trajectory_prob_aug 0.0 \
    \
    --seed 998244353 \
    --stage 1 \
    --train_bsz 3600 \
    --eval_bsz 3600 \
    --bsz_sent 4 \
    --learning_rate 2e-4 \
    --warmup_steps 0 \
    --scheduler cosine \
    --run_name $name \
    \
    --n_device ${SLURM_GPUS} \
    --deepspeed_stage 2 \
    --max_epochs 6 \
    --grad_acc_steps 4 \
    --clip_norm 1.0 \
    --save_dir ${save_path} \
    --log_step 5 \
    --eval_step 200

python train/zero_to_fp32.py ${save_path}/last.ckpt ${save_path}/last.ckpt/pytorch_model.bin
python train/prune_bin.py ${save_path}/last.ckpt/pytorch_model.bin