#!/usr/bin/env bash
##SBATCH --nodelist=babel-4-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:L40S:1
##SBATCH --nodelist=babel-3-17
#SBATCH --partition=array
#SBATCH --time=2-00:00:00
##SBATCH --dependency=afterok:job_id
#SBATCH --array=1-8
##SBATCH --account=siqiouya
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH -e slurm_logs/%A-%a.err
#SBATCH -o slurm_logs/%A-%a.out

source /home/siqiouya/anaconda3/bin/activate speechllama


src_segment_size=960
frame_num=${SLURM_ARRAY_TASK_ID}
batch_size=1
attn_layer=14
preserve_t=40
min_speech_duration=0
max_speech_duration=28.8

checkpoint_dir=/compute/babel-5-23/siqiouya/runs/en-de/8B-s2-bi-v3.5.2/last.ckpt/

export PYTHONPATH=/home/siqiouya/work/sllama

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
  --output ${checkpoint_dir}/streamatt/bsz${batch_size}_layer${attn_layer}_t${preserve_t}_d${min_speech_duration}_fn${frame_num} \
  \
  --quality-metrics BLEU \
  --sacrebleu-tokenizer 13a \
  --eval-latency-unit word \
  --min-start-sec 0. \
  --w2v2-path "/data/user_data/siqiouya/runs/pretrained/wav2_vec_vox_960h_pl.pt" \
  --w2v2-type "w2v2" \
  --ctc-finetuned \
  --xpos 0 \
  \
  --length-shrink-cfg "[(1024,2,2)] * 2" \
  --latency-multiplier 1 \
  --max-latency-multiplier 1 \
  --block-size 10000000 \
  --max-cache-size 10000000 \
  --max-len-a 1 \
  --max-len-b 256 \
  --repetition-penalty 1.2 \
  --beam 4 \
  --no-repeat-ngram-size 5 \
  \
  --text-preserve-num ${preserve_t} \
  --min-speech-duration ${min_speech_duration} \
  --max-speech-duration ${max_speech_duration}