#!/bin/bash
#SBATCH --job-name=causal-alignatt
#SBATCH --output=./slurm-out/alignatt.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
##SBATCH --gres=gpu:A100_40GB:8
##SBATCH --gres=gpu:A6000:8
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --time 2-00:00:00
#SBATCH --mail-user=xixu@andrew.cmu.edu
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=preempt
##SBATCH --nodelist=babel-15-24
#SBATCH --exclude=babel-1-23,babel-4-21,babel-13-13,babel-12-9,babel-13-17,babel-5-11,babel-5-15,babel-5-19,babel-6-21
#SBATCH --array=2-20:2

PORT=$((12345 + SLURM_ARRAY_TASK_ID))
source $HOME/sllama/bin/activate

# Calculate port based on array task ID
# if SLURM_ARRAY_TASK_ID is not set, use 23456
PORT=${SLURM_ARRAY_TASK_ID:-23456}

src_segment_size=960
# frame_num=2 # 2 to 20
frame_num=$SLURM_ARRAY_TASK_ID
batch_size=1
attn_layer=20

# checkpoint_dir=/compute/babel-5-23/siqiouya/runs/8B-s2-v2.0/last.ckpt/
checkpoint_dir=/compute/babel-5-23/siqiouya/runs/8B-s2-v2.0-bi/last.ckpt/

export PYTHONPATH=/home/xixu/work/data-synthesis/sllama
# export CUDA_VISIBLE_DEVICES=0

simuleval \
  --agent eval/agents/tt_alignatt_sllama3.py \
  --agent-class "agents.AlignAttSpeechLlama3" \
  --source-segment-size ${src_segment_size} \
  --frame-num ${frame_num} \
  --attn-layer ${attn_layer} \
  --model-name "/compute/babel-4-1/siqiouya/llama-3.1-8b-instruct-hf" \
  --state-dict-path ${checkpoint_dir}/pytorch_model.bin \
  --source-lang "English" \
  --target-lang "Chinese" \
  --source /compute/babel-14-5/siqiouya/en-zh//tst-COMMON.source \
  --target /compute/babel-14-5/siqiouya/en-zh/tst-COMMON.target \
  --output result/alignatt_bi_b1/${frame_num} \
  --quality-metrics BLEU \
  --sacrebleu-tokenizer zh \
  --min-start-sec 0.96 \
  --w2v2-path "/data/user_data/xixu/wav2_vec_vox_960h_pl.pt" \
  --w2v2-type "w2v" \
  --ctc-finetuned \
  --block-size 9999 \
  --max-cache-size 9999 \
  --length-shrink-cfg "[(1024,2,2)] * 2" \
  --max-len-a 1 \
  --max-len-b 256 \
  --repetition-penalty 1.2 \
  --beam 1 \
  --eval-latency-unit char \
  --no-repeat-ngram-size 3