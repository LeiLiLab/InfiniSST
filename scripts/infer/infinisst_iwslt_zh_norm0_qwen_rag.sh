#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --partition=taurus
#SBATCH --array=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jaxanluo@gmail.com
#SBATCH -e logs/infer_infinisst_%A_%a.err
#SBATCH -o logs/infer_infinisst_%A_%a.out



source /home/jiaxuanluo/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

state_dict_path="/mnt/gemini/data2/jiaxuanluo/stage1_M=12_norm0_qwen2.5-7b-instruct_rope.bin"
lora_path="/mnt/gemini/data2/jiaxuanluo/stage2_M=12_norm0_qwen2.5-7b-instruct_rope.bin"
# state_dict_path="/mnt/aries/data6/jiaxuanluo/demo/en-zh/pytorch_model.bin"
# lora_path="/mnt/aries/data6/jiaxuanluo/demo/en-zh/lora.bin"
lora_rank=32
save_dir="/mnt/gemini/data2/jiaxuanluo/"
llm_path="/mnt/aries/data6/jiaxuanluo/Qwen2.5-7B-Instruct"
w2v2_path="/mnt/aries/data6/xixu/demo/wav2_vec_vox_960h_pl.pt"
w2v2_type=w2v2
ctc_finetuned=True

#ROOT="/mnt/gemini/data2/jiaxuanluo/2/acl_6060/dev"
ROOT="/mnt/taurus/data/siqiouyang/datasets/acl6060"
lang_code=zh
lang=Chinese

# if evaluating on German and Spanish
# tokenizer=13a
# unit=word

# if evaluating on Chinese
tokenizer=zh
unit=char

# agent specific parameters
audio_normalize=0
src_segment_size=$(($SLURM_ARRAY_TASK_ID * 960))
latency_multiplier=$SLURM_ARRAY_TASK_ID
max_llm_cache_size=1000
no_repeat_ngram_lookback=100
no_repeat_ngram_size=5
max_new_tokens=$(($SLURM_ARRAY_TASK_ID * 10))
max_latency_multiplier=12
beam=4
ms=0

# use your own path to repo
export PYTHONPATH=/home/jiaxuanluo/InfiniSST

# Change to data directory so relative paths in dev.source work
cd ${ROOT}

simuleval \
    --agent /home/jiaxuanluo/InfiniSST/agents/infinisst.py \
    --source-segment-size ${src_segment_size} \
    --latency-multiplier ${latency_multiplier} \
    --max-latency-multiplier ${max_latency_multiplier} \
    --source-lang English \
    --target-lang ${lang} \
    --min-start-sec ${ms} \
    --source dev.source \
    --target dev.target.${lang_code} \
    --output ${save_dir}/infinisst_acl6060/cache${max_llm_cache_size}_seg${src_segment_size}_beam${beam}_ms${ms}_nrnl${no_repeat_ngram_lookback}_nrns${no_repeat_ngram_size} \
    --model-type w2v2_qwen25 \
    --w2v2-path ${w2v2_path} \
    --w2v2-type ${w2v2_type} \
    --ctc-finetuned ${ctc_finetuned} \
    --audio-normalize ${audio_normalize} \
    \
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
    --model-name ${llm_path} \
    --state-dict-path ${state_dict_path} \
    --lora-path ${lora_path} \
    --lora-rank ${lora_rank} \
    \
    --quality-metrics BLEU \
    --eval-latency-unit ${unit} \
    --sacrebleu-tokenizer ${tokenizer}