#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=256GB
#SBATCH --gpus=8
##SBATCH --constraint=xeon-4116 
#SBATCH --partition=aries
#SBATCH --time=7-00:00:00
##SBATCH --dependency=afterok:job_id
##SBATCH --array=1-7
#SBATCH --account=siqiouyang
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH --output=slurm_logs/slurm-%j.out

source /mnt/taurus/home/siqiouyang/anaconda3/bin/activate /mnt/taurus/home/siqiouyang/anaconda3/envs/speechllama

llm_path=/mnt/taurus/data/siqiouyang/download/llama3.1-8b-hf/
sllm_weight_path=/mnt/taurus/data/siqiouyang/runs/sllama/en-de/3.1-8B-s1-lightning-german-w2v2-rope-noxpos-bs1/
w2v2_path=/mnt/taurus/data/xixu/models/wav2_vec_vox_960h_pl.pt
# w2v2_path=/data/user_data/siqiouya/runs/pretrained/hubert_large_ll60k_finetune_ls960.pt
w2v2_type=w2v2
ctc_finetuned=True
# data_path=/scratch/xixu/dataset/must-c-v1.0/en-es
data_path=/mnt/aries/data/siqiouyang/datasets/must-c-v1.0
# data_path=/compute/babel-6-17/xixu/datasets/must-c-v1.0/en-fr


source_lang="English"
target_lang="German"
name="3.1-8B-s2-lightning-${target_lang,,}-${w2v2_type}-rope-noxpos-bs1"
save_path=/mnt/taurus/data/siqiouyang/runs/sllama/en-de/$name
# rm -rf ${save_path}
mkdir -p ${save_path}

export PYTHONPATH=/home/siqiouyang/work/projects/sllama
export WANDB_PROJECT="mustc_1.0_de"
export WANDB_ENTITY="streamllama"

export TOKENIZERS_PARALLELISM=false

export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
SLURM_GPUS=8

python /home/siqiouyang/work/projects/sllama/train/zero_to_fp32.py ${sllm_weight_path} ${sllm_weight_path}/pytorch_model.bin
python /home/siqiouyang/work/projects/sllama/train/prune_bin.py ${sllm_weight_path}/pytorch_model.bin

srun python /home/siqiouyang/work/projects/sllama/train/main_lightning.py \
    \
    --w2v2_path ${w2v2_path} \
    --w2v2_type ${w2v2_type} \
    --w2v2_freeze True \
    --ctc_finetuned ${ctc_finetuned} \
    --length_shrink_cfg "[(1024,2,2)] * 2" \
    --block_size 1 \
    --max_cache_size 500 \
    --xpos False \
    \
    --llm_path ${llm_path} \
    --sllm_weight_path ${sllm_weight_path}/pytorch_model.bin \
    \
    --data_path ${data_path} \
    --data_split_train 'train_st_de_mfa_llama3' \
    --data_split_eval 'dev_st_de_mfa_llama3' \
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