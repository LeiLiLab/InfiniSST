#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gpus=1
##SBATCH --constraint=xeon-4116 
#SBATCH --partition=aries
#SBATCH --time=1-2:34:56 
##SBATCH --dependency=afterok:job_id
#SBATCH --array=15-17:2
#SBATCH --account=xixu
#SBATCH --mail-type=ALL
##SBATCH --mail-user=xixu@andrew.cmu.edu
#SBATCH --output=/mnt/taurus/home/xixu/iwslt/results/out/wait-k%a.txt

# conda config --append envs_dirs /mnt/taurus/home/siqiouyang/anaconda3/envs/
# source /mnt/taurus/home/siqiouyang/anaconda3/bin/activate sllama

src_segment_size=320
k=$SLURM_ARRAY_TASK_ID
# k=1001

checkpoint_dir=/mnt/taurus/data1/xixu/runs/sllama/wavlm_clean/stage2/checkpoint-1900
# checkpoint_dir=/mnt/data/xixu/runs/sllama/en-es/7b/wWav2vec/stage2/checkpoint-2000

# mkdir -p ${checkpoint_dir}/debug_lp

export PYTHONPATH=/mnt/taurus/home/xixu/iwslt

simuleval \
  --agent /mnt/taurus/home/xixu/sllama_new/eval/agents/tt_waitk_sllama.py \
  --source-segment-size ${src_segment_size} \
  --waitk-lagging ${k} --repeat-penalty 1.0 \
  --model-dir ${checkpoint_dir} \
  --speech-tower-path ${checkpoint_dir}/speech_tower.bin \
  --speech-tower-type wavlm \
  --prompt "<speech_here>" \
  --source /mnt/taurus/data1/xixu/en-de/tst-COMMON.source \
  --target /mnt/taurus/data1/xixu/en-de/tst-COMMON.target \
  --output /mnt/taurus/home/xixu/iwslt/results/de/waik/${k} \
  --quality-metrics BLEU --sacrebleu-tokenizer 13a \