#!/usr/bin/env bash

##SBATCH --nodelist=babel-4-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:L40S:1
##SBATCH --nodelist=babel-3-17
##SBATCH --exclude=babel-3-[5,9,13,17],babel-4-[5,9,29],babel-6-29,babel-7-[1,5,9],babel-8-[5,9,13],babel-10-[5,9,13],babel-11-25,babel-12-29,babel-13-[13,21,29],babel-14-[5,25]
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
##SBATCH --dependency=afterok:job_id
#SBATCH --array=1,2,4,8
##SBATCH --account=siqiouya
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH -e slurm_logs/%A-%a.err
#SBATCH -o slurm_logs/%A-%a.out

source /home/siqiouya/anaconda3/bin/activate infinisst

state_dict_path="/compute/babel-5-23/siqiouya/runs/gigaspeech/en-zh/stage1_M=12/last.ckpt/pytorch_model.bin"
lora_path="/compute/babel-5-23/siqiouya/runs/gigaspeech/en-zh/stage2_8b_lora_rank32_M=12/last.ckpt/pytorch_model.bin"
lora_rank=32
save_dir="/compute/babel-5-23/siqiouya/runs/gigaspeech/en-zh/stage2_8b_lora_rank32_M=12/last.ckpt/"

llama_path=/compute/babel-4-1/siqiouya/llama-3.1-8b-instruct-hf

w2v2_path=/data/user_data/siqiouya/runs/pretrained/wav2_vec_vox_960h_pl.pt
w2v2_type=w2v2
ctc_finetuned=True

ROOT=/compute/babel-14-5/siqiouya/iwslt25/acl_6060
lang_code=zh
lang=Chinese

# if evaluating on German and Spanish
tokenizer=zh
unit=char

# if evaluating on Chinese
# tokenizer=zh
# unit=char

# agent specific parameters
audio_normalize=1
src_segment_size=960
latency_multiplier=1
max_llm_cache_size=1000
no_repeat_ngram_lookback=100
no_repeat_ngram_size=5
max_new_tokens=10
max_latency_multiplier=12
beam=4
ms=0

pseudo_batch_size=$SLURM_ARRAY_TASK_ID

# use your own path to repo
export PYTHONPATH=/home/siqiouya/work/sllama

simuleval \
    --agent agents/infinisst.py \
    --source-segment-size ${src_segment_size} \
    --latency-multiplier ${latency_multiplier} \
    --max-latency-multiplier ${max_latency_multiplier} \
    --source-lang English \
    --target-lang ${lang} \
    --min-start-sec ${ms} \
    --source ${ROOT}/dev.source \
    --target ${ROOT}/dev.target \
    --output ${save_dir}/profile/cache${max_llm_cache_size}_seg${src_segment_size}_beam${beam}_nrns${no_repeat_ngram_size}_bsz${pseudo_batch_size} \
    --pseudo-batch-size ${pseudo_batch_size} \
    --w2v2-path ${w2v2_path} \
    --w2v2-type ${w2v2_type} \
    --ctc-finetuned ${ctc_finetuned} \
    --audio-normalize ${audio_normalize} \
    --length-shrink-cfg "[(1024,2,2)] * 2" \
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
    --state-dict-path ${state_dict_path} \
    --lora-path ${lora_path} \
    --lora-rank ${lora_rank} \
    \
    --quality-metrics BLEU \
    --eval-latency-unit ${unit} \
    --sacrebleu-tokenizer ${tokenizer}