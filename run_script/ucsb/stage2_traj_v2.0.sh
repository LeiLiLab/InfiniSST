#!/usr/bin/env bash


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=512GB
#SBATCH --gpus=8
##SBATCH --constraint=xeon-4116 
#SBATCH --partition=aries
#SBATCH --time=7-00:00:00
##SBATCH --dependency=afterok:job_id
##SBATCH --array=1-7
#SBATCH --account=siqiouyang
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH -o slurm_logs/slurm-%j.out
#SBATCH -e slurm_logs/slurm-%j.err

source /mnt/taurus/home/siqiouyang/anaconda3/bin/activate /mnt/taurus/home/siqiouyang/anaconda3/envs/speechllama

llm_path=/mnt/taurus/data/siqiouyang/download/llama3.1-8b-instruct-hf/
sllm_weight_path=/mnt/taurus/data/siqiouyang/runs/sllama/en-zh/traj_instruct/8B-traj-s1-v2.0
w2v2_path=/mnt/taurus/data/xixu/models/wav2_vec_vox_960h_pl.pt
# w2v2_path=/data/user_data/siqiouya/runs/pretrained/hubert_large_ll60k_finetune_ls960.pt
w2v2_type=w2v2
ctc_finetuned=True
# data_path=/scratch/xixu/dataset/must-c-v1.0/en-es
# data_path=/compute/babel-6-17/xixu/datasets/must-c-v1.0/en-de
# data_path=/compute/babel-6-17/xixu/datasets/must-c-v1.0/en-fr
data_path=/mnt/taurus/data/siqiouyang/datasets/must-c-v1.0/mustchinese/en-zh

source_lang="English"
target_lang="Chinese"
name="8B-traj-s2-v2.0"
save_path=/mnt/taurus/data/siqiouyang/runs/sllama/en-zh/$name
rm -rf ${save_path}
mkdir -p ${save_path}

export PYTHONPATH=/home/siqiouyang/work/projects/sllama
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="mustc_1.0_zh"
export WANDB_ENTITY="streamllama"

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
SLURM_GPUS=8

python /home/siqiouyang/work/projects/sllama/train/zero_to_fp32.py ${sllm_weight_path} ${sllm_weight_path}/pytorch_model.bin
python /home/siqiouyang/work/projects/sllama/train/prune_bin.py ${sllm_weight_path}/pytorch_model.bin

srun python /home/siqiouyang/work/projects/sllama15314 /train/main_lightning.py \
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
    --trajectory 2 \
    \
    --seed 998244353 \
    --stage 2 \
    --train_bsz 1500 \
    --eval_bsz 1500 \
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