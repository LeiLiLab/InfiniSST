#!/usr/bin/env bash
#!/bin/bash
#SBATCH --job-name=llama3-8b
#SBATCH --output=./slurm-out/llama3-8b.out
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
#SBATCH --array=1-7


source $HOME/sllama/bin/activate

# Calculate port based on array task ID (using base port 12345)
PORT=$((12345 + SLURM_ARRAY_TASK_ID))

src_segment_size=960
n_word_per_input=3
batch_size=1
xpos=0

# checkpoint_dir=/compute/babel-5-23/siqiouya/runs/8B-s2-v2.0/last.ckpt/
checkpoint_dir=/compute/babel-5-23/siqiouya/runs/8B-s2-v2.0-bi/last.ckpt/

export PYTHONPATH=/home/xixu/work/data-synthesis/sllama

run_simuleval() {
    local k=$1
    
    simuleval \
      --agent eval/agents/tt_waitk_sllama3_word.py \
      --agent-class "agents.WaitkSpeechLlama3" \
      --source-segment-size ${src_segment_size} \
      --waitk-lagging ${k} \
      --n-word-per-input ${n_word_per_input} \
      --model-name "/compute/babel-4-1/siqiouya/llama-3.1-8b-instruct-hf" \
      --state-dict-path ${checkpoint_dir}/pytorch_model.bin \
      --source-lang "English" \
      --target-lang "Chinese" \
      --source /compute/babel-6-17/xixu/datasets/must-c-v2.0/en-zh/tst-COMMON.source \
      --target /compute/babel-6-17/xixu/datasets/must-c-v2.0/en-zh/tst-COMMON.target \
      --output result/bi_offline_k/${k} \
      --quality-metrics BLEU \
      --sacrebleu-tokenizer zh \
      --min-start-sec 0.96 \
      --w2v2-path "/data/user_data/xixu/wav2_vec_vox_960h_pl.pt" \
      --w2v2-type "w2v" \
      --ctc-finetuned \
      --block-size 9999 \
      --max-cache-size 999 \
      --length-shrink-cfg "[(1024,2,2)] * 2" \
      --xpos ${xpos} \
      --warmup 0 \
      --max-len-a 1 \
      --max-len-b 256 \
      --repetition-penalty 1.2 \
       --eval-latency-unit char \
      --beam 4 \
      --no-repeat-ngram-size 3 
}

# Run with array task ID
run_simuleval $SLURM_ARRAY_TASK_ID
# run_simuleval 1