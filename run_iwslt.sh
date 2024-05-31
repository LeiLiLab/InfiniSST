#!/usr/bin/env bash

source /app/env/bin/activate

pip install gdown


gdown --id "https://drive.google.com/drive/folders/1ww6pBJeQ1sLAkKfA7Vgel9BzeuHWhZe9?usp=sharing" --output /app/en_de_checkpoint --folder

src_segment_size=2500
hold_n=7
beam=4
min_start_sec=2

checkpoint_dir=/app/en_de_checkpoint
# checkpoint_dir=/mnt/taurus/data1/xixu/runs/sllama/wavlm_clean/stage2/checkpoint-1900

export PYTHONPATH=/app

simuleval \
  --agent eval/agents/tt_holdn_sllama.py \
  --source-segment-size ${src_segment_size} \
  --hold-n $hold_n \
  --beam $beam --min-start-sec $min_start_sec \
  --model-dir ${checkpoint_dir} \
  --speech-tower-path ${checkpoint_dir}/speech_tower.bin \
  --speech-tower-type wavlm \
  --prompt "<speech_here>" \
  --source data/tst-COMMON.source \
  --target data/tst-COMMON.target \
  --output ${src_segment_size}ms_beam${beam}_n${hold_n}_ms${min_start_sec} \
  --quality-metrics BLEU --sacrebleu-tokenizer 13a 