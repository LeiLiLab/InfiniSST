#!/usr/bin/env bash

source $HOME/sllama/bin/activate

src_segment_size=1000
k=${SLURM_ARRAY_TASK_ID}
n_word_per_input=3
batch_size=1

checkpoint_dir=/compute/babel-9-7/xixu/runs/es-new/stage3-uni-waco-word-block50-fixed-mix-from-stage0/checkpoint-2200

k=1

python train/correct_path.py

export PYTHONPATH=/home/yuanjinw/work/sllama

simuleval \
  --agent eval/agents/tt_waitk_sllama3_word.py \
  --agent-class "agents.WaitkSpeechLlama3" \
  --source-segment-size ${src_segment_size} \
  --waitk-lagging ${k} --n-word-per-input ${n_word_per_input} \
  --repeat-penalty 1.0 \
  --model-dir ${checkpoint_dir} \
  --source /scratch/xixu/en-es/tst-COMMON-profile-60s.source \
  --target /scratch/xixu/en-es/tst-COMMON-profile-60s.source \
  --output debug.txt \
  --quality-metrics BLEU --sacrebleu-tokenizer 13a \
  --batch-size ${batch_size}