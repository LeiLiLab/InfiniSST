#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
##SBATCH --nodelist=babel-3-17
##SBATCH --constraint=xeon-4116 
#SBATCH --partition=aries
#SBATCH --time=1-00:00:00
##SBATCH --dependency=afterok:job_id
#SBATCH --array=10-35:5
#SBATCH --account==xixu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xixu@cs.cmu.edu
#SBATCH --output=/mnt/taurus/home/xixu/iwslt/results/out/de_holdn_%a.txt


# src_segment_size=$2000
# src_segment_size=640
# hold_n=${SLURM_ARRAY_TASK_ID}
# agreement_threshold=0.8
# min_start_sec=$2
# min_start_sec=1.0

hold_n=4
beam=4
# src_segment_size=2000
src_segment_size=${SLURM_ARRAY_TASK_ID*100}
# min_start_sec=$(echo "scale=1; $SLURM_ARRAY_TASK_ID * 0.1" | bc)
# min_start_sec=$((SLURM_ARRAY_TASK_ID / 10))

checkpoint_dir=/mnt/taurus/data1/xixu/runs/sllama/wavlm_clean/stage2/checkpoint-1900
# checkpoint_dir=/mnt/data/xixu/runs/sllama/en-es/7b/wWav2vec/stage2/checkpoint-2000

# mkdir -p ${checkpoint_dir}/debug_lp

export PYTHONPATH=/mnt/taurus/home/xixu/iwslt

simuleval \
  --agent eval/agents/tt_holdn_sllama.py \
  --source-segment-size ${src_segment_size} \
  --hold-n $hold_n \
  --beam $beam --min-start-sec $src_segment_size \
  --model-dir ${checkpoint_dir} \
  --speech-tower-path ${checkpoint_dir}/speech_tower.bin \
  --speech-tower-type wavlm \
  --prompt "<speech_here>" \
  --source /mnt/taurus/data1/xixu/en-de/tst-COMMON.source \
  --target /mnt/taurus/data1/xixu/en-de/tst-COMMON.target \
  --output /mnt/taurus/home/xixu/iwslt/results/de/holdn/${src_segment_size}ms_beam${beam}_n${hold_n}_ms${min_start_sec} \
  --quality-metrics BLEU --sacrebleu-tokenizer 13a