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

source /home/siqiouya/anaconda3/bin/activate speechllama

ckpt_dir=/compute/babel-5-23/siqiouya/runs/en-zh/8B-traj-s2-v3.5/last.ckpt/
src_segment_size=960
latency_multiplier=1
max_speech_cache_size=480
max_llm_cache_size_list=(500 1000 2000 4000)
max_llm_cache_size=${max_llm_cache_size_list[$SLURM_ARRAY_TASK_ID-1]}
no_repeat_ngram_lookback=100
no_repeat_ngram_size=5
max_new_tokens=10
pseudo_batch_size=1
beam=4
ms=0

export PYTHONPATH=/home/siqiouya/work/sllama
simuleval \
    --agent eval/agents/streamllama.py \
    --source-segment-size ${src_segment_size} \
    --latency-multiplier ${latency_multiplier} \
    --pseudo-batch-size ${pseudo_batch_size} \
    --source-lang English \
    --target-lang Chinese \
    --min-start-sec ${ms} \
    --source /compute/babel-14-5/siqiouya/en-zh/tst-COMMON_full.source \
    --target /compute/babel-14-5/siqiouya/en-zh/tst-COMMON_full.target \
    --output ${ckpt_dir}/simul-results-full-betterfilterbadwords/cache${max_llm_cache_size}_seg${src_segment_size}_beam${beam}_ms${ms}_nrnl${no_repeat_ngram_lookback}_nrns${no_repeat_ngram_size}_bsz${pseudo_batch_size} \
    --w2v2-path /data/user_data/siqiouya/runs/pretrained/wav2_vec_vox_960h_pl.pt \
    --w2v2-type w2v2 \
    --ctc-finetuned \
    \
    --length-shrink-cfg "[(1024,2,2)] * 2" \
    --block-size 48 \
    --max-cache-size ${max_speech_cache_size} \
    --xpos 0 \
    \
    --max-llm-cache-size ${max_llm_cache_size} \
    --always-cache-system-prompt \
    \
    --max-len-a 10 \
    --max-len-b 20 \
    --max-new-tokens ${max_new_tokens} \
    --beam ${beam} \
    --no-repeat-ngram-lookback ${no_repeat_ngram_lookback} \
    --no-repeat-ngram-size ${no_repeat_ngram_size} \
    --repetition-penalty 1.2 \
    \
    --model-name /compute/babel-4-1/siqiouya/llama-3.1-8b-instruct-hf \
    --state-dict-path ${ckpt_dir}/pytorch_model.bin \
    \
    --quality-metrics BLEU \
    --eval-latency-unit char \
    --sacrebleu-tokenizer zh