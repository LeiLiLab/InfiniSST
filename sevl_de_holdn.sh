#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --gres=gpu:A6000:1
##SBATCH --nodelist=babel-3-17
##SBATCH --constraint=xeon-4116 
##SBATCH --partition=gemini
#SBATCH --time=1-00:00:00
##SBATCH --dependency=afterok:job_id
#SBATCH --array=6-15:3
#SBATCH --account=siqiouyang
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
##SBATCH --output=/home/xixu/slurm.txts

source /home/siqiouya/anaconda3/bin/activate sllama_lightning

src_segment_size=$1
# src_segment_size=640
hold_n=${SLURM_ARRAY_TASK_ID}
# agreement_threshold=0.8
beam=$2
min_start_sec=$3
# min_start_sec=1.0

checkpoint_dir=/data/user_data/siqiouya/runs/iwslt-en-de-wavlm-llama2
# checkpoint_dir=/mnt/data/xixu/runs/sllama/en-es/7b/wWav2vec/stage2/checkpoint-2000

# mkdir -p ${checkpoint_dir}/debug_lp

export PYTHONPATH=/home/siqiouya/work/sllama:/home/siqiouya/work/SimulEval
simuleval \
  --agent eval/agents/tt_holdn_sllama.py \
  --source-segment-size ${src_segment_size} \
  --hold-n $hold_n \
  --beam $beam --min-start-sec $min_start_sec \
  --model-dir ${checkpoint_dir} \
  --speech-tower-path ${checkpoint_dir}/speech_tower.bin \
  --speech-tower-type wavlm \
  --prompt "<speech_here>" \
  --source /data/user_data/siqiouya/dataset/en-de-tst-COMMON-v2/tst-COMMON.source \
  --target /data/user_data/siqiouya/dataset/en-de-tst-COMMON-v2/tst-COMMON.target \
  --output ${checkpoint_dir}/iwslt/tst-COMMON-v2-holdn-word/${src_segment_size}ms_beam${beam}_n${hold_n}_ms${min_start_sec} \
  --quality-metrics BLEU --sacrebleu-tokenizer 13a