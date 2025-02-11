#!/bin/bash
#SBATCH --job-name=es-streamatt-fw
#SBATCH --output=./slurm-out/alignatt.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100_40GB:8
##SBATCH --gres=gpu:A6000:8
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time 2-00:00:00
#SBATCH --mail-user=xixu@andrew.cmu.edu
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=preempt
##SBATCH --nodelist=babel-15-24
#SBATCH --exclude=babel-1-23,babel-4-21,babel-13-13,babel-12-9,babel-13-17,babel-5-11,babel-5-15,babel-5-19,babel-6-21
#SBATCH --array=1-2:1

source $HOME/sllama/bin/activate


PORT=${SLURM_ARRAY_TASK_ID:-23456}

src_segment_size=960
# frame_num=2 # 2 to 20
frame_num=${SLURM_ARRAY_TASK_ID}
# frame_num=2
# frame_num=1
# frame_num=4
frame_num=5
batch_size=1
attn_layer=-1

# checkpoint_dir=/compute/babel-5-23/siqiouya/runs/8B-s2-v2.0/last.ckpt/
checkpoint_dir=/compute/babel-5-23/siqiouya/runs/en-zh/8B-s2-v2.0-bi/last.ckpt
# checkpoint_dir=/compute/babel-5-23/siqiouya/runs/en-es/8B-s2-bi-v3.5/last.ckpt
checkpoint_dir=/compute/babel-5-23/siqiouya/runs/en-de/8B-s2-bi-v3.5/last.ckpt

export PYTHONPATH=/home/xixu/work/data-synthesis/sllama
# export CUDA_VISIBLE_DEVICES=0

simuleval \
  --agent eval/agents/tt_alignatt_sllama_stream_att_fw.py \
  --agent-class "agents.AlignAttStreamAttFW" \
  --source-segment-size ${src_segment_size} \
  --frame-num ${frame_num} \
  --attn-layer ${attn_layer} \
  --model-name "/compute/babel-4-1/siqiouya/llama-3.1-8b-instruct-hf" \
  --state-dict-path ${checkpoint_dir}/pytorch_model.bin \
  --source-lang "English" \
  --target-lang "German" \
  --source /compute/babel-14-5/siqiouya/en-de/tst-COMMON_full.source \
  --target /compute/babel-14-5/siqiouya/en-de/tst-COMMON_full.target \
  --output /compute/babel-14-5/siqiouya/en-de/streamatt_fw_20/${frame_num} \
  \
  --quality-metrics BLEU \
  --sacrebleu-tokenizer 13a \
  --min-start-sec 0.96 \
  --w2v2-path "/data/user_data/xixu/wav2_vec_vox_960h_pl.pt" \
  --w2v2-type "w2v" \
  --ctc-finetuned \
  \
  --length-shrink-cfg "[(1024,2,2)] * 2" \
  --latency-multiplier 1 \
  --block-size 10000000 \
  --max-cache-size 10000000 \
  --max-len-a 1 \
  --max-len-b 256 \
  --repetition-penalty 1.2 \
  --beam 1 \
  --eval-latency-unit word \
  --no-repeat-ngram-size 5 \
   \
  --speech-preserve-duration 25.92 \
  --text-preserve-num 20

  # --source /compute/babel-14-5/siqiouya/en-zh/tst-COMMON_full.source \
# --output /compute/babel-14-5/siqiouya/en-zh/streamatt_fw_bug_free_40/${frame_num} \
# chmod -R 777 /compute/babel-14-5/siqiouya/en-zh/streamatt_es/
chmod -R 777 /compute/babel-14-5/siqiouya/en-es/streamatt_fw_20/