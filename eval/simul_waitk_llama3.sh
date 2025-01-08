#!/usr/bin/env bash

source $HOME/sllama/bin/activate

src_segment_size=960
k=2
n_word_per_input=3
batch_size=1
xpos=0

checkpoint_dir=/compute/babel-5-23/siqiouya/runs/8B-s2-v2.0/last.ckpt/

export PYTHONPATH=/home/xixu/work/data-synthesis/sllama


export CUDA_VISIBLE_DEVICES=0
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
  --source source \
  --target target \
  --output debug \
  --quality-metrics BLEU \
  --sacrebleu-tokenizer zh \
  --min-start-sec 0.96 \
  --w2v2-path "/data/user_data/xixu/wav2_vec_vox_960h_pl.pt" \
  --w2v2-type "w2v" \
  --ctc-finetuned \
  --block-size 48 \
  --max-cache-size 500 \
  --length-shrink-cfg "[(1024,2,2)] * 2" \
  --xpos ${xpos} \
  --warmup 0 \
  --max-len-a 1 \
  --max-len-b 256 \
  --beam 4 \
  --no-repeat-ngram-size 3 \

# delete repeted penalty and bsz