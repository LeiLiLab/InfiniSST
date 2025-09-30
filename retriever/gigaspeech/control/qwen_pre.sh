#!/bin/bash
#SBATCH --job-name=qwen_pre
#SBATCH --partition=taurus
#SBATCH --mem=96GB
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --output=logs/qwen_pre_%A_%a.out
#SBATCH --error=logs/qwen_pre_%A_%a.err

cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

# 设置内存和性能优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
# --gpu_ids=0,1,2,3,4
python Qwen2_Audio_term_level_train_ddp.py --train_samples_path=data/xl_cleaned_term_level_chunks_merged.json --test_samples_path="" --epochs=20 --batch_size=80 --lr=1e-4 --save_path=data/qwen2_audio_term_level_full.pt --best_model_path=data/qwen2_audio_term_level_best.pt --audio_text_loss_ratio=0.3 --audio_term_loss_ratio=0.7 --glossary_path=data/terms/glossary_merged.json --filter_no_term --model_name=Qwen/Qwen2-Audio-7B-Instruct --lora_r=16 --lora_alpha=32 --lora_dropout=0.1