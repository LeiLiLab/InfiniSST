#!/usr/bin/env bash

##SBATCH --nodelist=babel-4-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:L40S:2
#SBATCH --partition=general
#SBATCH --exclude=babel-q5-28
#SBATCH --time=2-00:00:00
##SBATCH --dependency=afterok:job_id
#SBATCH --array=1
##SBATCH --account=siqiouya
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH -e slurm_logs/%A-%a.err
#SBATCH -o slurm_logs/%A-%a.out

source /home/siqiouya/miniconda3/bin/activate omni_inference

src_segment_size=$(($SLURM_ARRAY_TASK_ID * 960))
max_cache_chunks=$((60 / $SLURM_ARRAY_TASK_ID))
keep_cache_chunks=$((30 / $SLURM_ARRAY_TASK_ID))

lang_code=zh
lang=Chinese

tokenizer=zh
unit=char

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/siqiouya/code/infinisst-omni

uv run simuleval \
    --agent agents/infinisst_omni.py \
    --agent-class agents.InfiniSSTOmni \
    --source-segment-size ${src_segment_size} \
    --output /data/user_data/siqiouya/ckpts/infinisst-omni/gigaspeech-zh-s_origin-bsz4/v1-20260122-055820-hf/evaluation/eso/en2zh/seg${src_segment_size} \
    --max-new-tokens 128 \
    --max-cache-chunks ${max_cache_chunks} \
    --keep-cache-chunks ${keep_cache_chunks} \
    --source-lang English \
    --target-lang ${lang} \
    --min-start-sec 0 \
    --source /data/group_data/li_lab/siqiouya/datasets/eso-dataset/test/sources.txt \
    --target /data/group_data/li_lab/siqiouya/datasets/eso-dataset/test/targets.txt \
    --use-vllm 1 \
    --temperature 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --model-name /data/user_data/siqiouya/ckpts/infinisst-omni/gigaspeech-zh-s_origin-bsz4/v1-20260122-055820-hf/ \
    --quality-metrics BLEU \
    --eval-latency-unit ${unit} \
    --sacrebleu-tokenizer ${tokenizer}