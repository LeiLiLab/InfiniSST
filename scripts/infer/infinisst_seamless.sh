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
#SBATCH --array=1-4
##SBATCH --account=siqiouya
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH -e slurm_logs/%A-%a.err
#SBATCH -o slurm_logs/%A-%a.out

source /home/siqiouya/anaconda3/bin/activate infinisst

checkpoint_dir=/compute/babel-5-23/siqiouya/runs/en-zh/stage2_seamless/last.ckpt/
llama_path=/compute/babel-4-1/siqiouya/llama-3.1-8b-instruct-hf

model_type=seamless_llama31
seamless_path=/compute/babel-4-1/siqiouya/seamless-m4t-v2-large

ROOT=/compute/babel-14-5/siqiouya
lang_code=zh
lang=Chinese

# if evaluating on German and Spanish
# tokenizer=13a
# unit=word

# if evaluating on Chinese
tokenizer=zh
unit=char

# agent specific parameters
src_segment_size=$(($SLURM_ARRAY_TASK_ID * 960))
latency_multiplier=$SLURM_ARRAY_TASK_ID
max_llm_cache_size=1000
no_repeat_ngram_lookback=100
no_repeat_ngram_size=5
max_new_tokens=$(($SLURM_ARRAY_TASK_ID * 10))
beam=4
ms=0

# use your own path to repo
export PYTHONPATH=/home/siqiouya/work/sllama

simuleval \
    --agent agents/infinisst.py \
    --source-segment-size ${src_segment_size} \
    --latency-multiplier ${latency_multiplier} \
    --source-lang English \
    --target-lang ${lang} \
    --min-start-sec ${ms} \
    --source ${ROOT}/en-${lang_code}/tst-COMMON_full.source \
    --target ${ROOT}/en-${lang_code}/tst-COMMON_full.target \
    --output ${checkpoint_dir}/infinisst/cache${max_llm_cache_size}_seg${src_segment_size}_beam${beam}_ms${ms}_nrnl${no_repeat_ngram_lookback}_nrns${no_repeat_ngram_size} \
    \
    --model-type ${model_type} \
    --seamless-path ${seamless_path} \
    \
    --block-size 48 \
    --max-cache-size 576 \
    \
    --max-llm-cache-size ${max_llm_cache_size} \
    --always-cache-system-prompt \
    \
    --max-new-tokens ${max_new_tokens} \
    --beam ${beam} \
    --no-repeat-ngram-lookback ${no_repeat_ngram_lookback} \
    --no-repeat-ngram-size ${no_repeat_ngram_size} \
    --repetition-penalty 1.2 \
    \
    --model-name ${llama_path} \
    --state-dict-path ${checkpoint_dir}/pytorch_model.bin \
    \
    --quality-metrics BLEU \
    --eval-latency-unit ${unit} \
    --sacrebleu-tokenizer ${tokenizer}