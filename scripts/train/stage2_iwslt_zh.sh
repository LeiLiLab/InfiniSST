#!/usr/bin/env bash

##SBATCH --nodelist=babel-4-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH --mem=392GB
#SBATCH --gres=gpu:L40S:8
##SBATCH --nodelist=babel-3-17
#SBATCH --exclude=babel-3-[5,9,13,17],babel-4-[5,9,29],babel-6-29,babel-7-[1,5,9],babel-8-[5,9,13],babel-10-[5,9,13],babel-11-25,babel-12-29,babel-13-[1,13,21,29],babel-14-25
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
#SBATCH --dependency=afterok:4564159
##SBATCH --array=1-7
##SBATCH --account=siqiouya
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH -e slurm_logs/%j.err
#SBATCH -o slurm_logs/%j.out

source /home/siqiouya/anaconda3/bin/activate infinisst

llama_path=/compute/babel-4-1/siqiouya/llama-3.1-8b-instruct-hf

w2v2_path=/data/user_data/siqiouya/runs/pretrained/wav2_vec_vox_960h_pl.pt
w2v2_type=w2v2
ctc_finetuned=True

ROOT=/compute/babel-14-5/siqiouya/iwslt25/train/
lang_code=zh
lang=Chinese
data_path=$ROOT

stage1_ckpt_dir="/compute/babel-5-23/siqiouya/runs/iwslt25/en-${lang_code}/stage1_M=12_ls-cv-vp/last.ckpt/"
save_dir=/compute/babel-5-23/siqiouya/runs/iwslt25/en-${lang_code}

source_lang="English"
target_lang=${lang} # e.g. German
name="stage2_M=12_ls-cv-vp"
save_path=${save_dir}/${name}
rm -rf ${save_path} # comment this line if you want to resume training
mkdir -p ${save_path}

export PYTHONPATH=$PYTHONPATH:$PWD
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="iwslt25_${lang_code}"
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
    --w2v2_path ${w2v2_path} \
    --w2v2_type ${w2v2_type} \
    --w2v2_freeze True \
    --ctc_finetuned ${ctc_finetuned} \
    --length_shrink_cfg "[(1024,2,2)] * 2" \
    --block_size 48 \
    --max_cache_size 576 \
    \
    --llm_path ${llama_path} \
    --sllm_weight_path ${stage1_ckpt_dir}/pytorch_model.bin \
    --use_flash_attn True \
    --lora_rank 32 \
    \
    --data_path ${data_path} \
    --data_split_train train_${lang_code}_mfa \
    --data_split_eval dev_${lang_code}_mfa \
    --source_lang "${source_lang}" \
    --target_lang "${target_lang}" \
    --trajectory 4 \
    --trajectory_max_multiplier 12 \
    --trajectory_prob_aug 0.0 \
    --audio_normalize True \
    \
    --seed 42 \
    --stage 2 \
    --train_bsz 1800 \
    --eval_bsz 1800 \
    --bsz_sent 2 \
    --learning_rate 1e-4 \
    --warmup_steps 1000 \
    --run_name $name \
    \
    --n_device ${SLURM_GPUS} \
    --deepspeed_stage 1 \
    --deepspeed_offload False \
    --max_epochs 1 \
    --grad_acc_steps 4 \
    --clip_norm 1.0 \
    --save_dir ${save_path} \
    --save_step 1000 \
    --log_step 100 \
    --eval_step 1000 # \
    # --profile "advanced"

python train/zero_to_fp32.py ${save_path}/last.ckpt ${save_path}/last.ckpt/pytorch_model.bin
python train/prune_bin.py ${save_path}/last.ckpt/pytorch_model.bin