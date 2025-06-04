#!/usr/bin/env bash

# Define the Python interpreter path and environment bin
source /opt/conda/etc/profile.d/conda.sh
conda activate infinisst
# Default to German if no argument is provided
lang_arg=${1:-"--de"}
source_file=$2
target_file=$3
output_file=$4

if [ -z "$lang_arg" ] || [ -z "$source_file" ] || [ -z "$target_file" ] || [ -z "$output_file" ]; then
    echo "Usage: $0 [--de|--zh] /path/to/source /path/to/target /path/to/output"
    exit 1
fi

# Set language-specific configurations based on argument
if [ "$lang_arg" == "--de" ]; then
    # German configuration
    state_dict_path="/app/iwslt2025/en-de_state_dict.bin"
    lora_path="/app/iwslt2025/en-de_lora.bin"
    lang_code=de
    lang=German
    tokenizer=13a
    unit=word
    latency_multiplier=2
elif [ "$lang_arg" == "--zh" ]; then
    # Chinese configuration
    state_dict_path="/app/iwslt2025/en-zh_state_dict.bin"
    lora_path="/app/iwslt2025/en-zh_lora.bin"
    lang_code=zh
    lang=Chinese
    tokenizer=zh
    unit=char
    latency_multiplier=3
else
    echo "Invalid argument. Use --de for German or --zh for Chinese."
    exit 1
fi

# Common configurations
lora_rank=32
llm_path=Qwen/Qwen2.5-7B-Instruct
w2v2_path=/app/iwslt2025/wav2_vec_vox_960h_pl.pt
w2v2_type=w2v2
ctc_finetuned=True

# Agent specific parameters
audio_normalize=0
src_segment_size=$(($latency_multiplier * 960))
max_llm_cache_size=1000
no_repeat_ngram_lookback=100
no_repeat_ngram_size=5
max_new_tokens=$(($latency_multiplier * 10))
max_latency_multiplier=12
beam=4
ms=0

# Set path to repo
export PYTHONPATH=/app

source /opt/conda/etc/profile.d/conda.sh
conda activate infinisst

# Call Python directly with simuleval module to avoid shebang issues
simuleval  \
    --agent agents/infinisst.py \
    --source-segment-size ${src_segment_size} \
    --latency-multiplier ${latency_multiplier} \
    --max-latency-multiplier ${max_latency_multiplier} \
    --source-lang English \
    --target-lang ${lang} \
    --min-start-sec ${ms} \
    --source "${source_file}" \
    --target "${target_file}" \
    --output "${output_file}" \
    --model-type w2v2_qwen25 \
    --w2v2-path ${w2v2_path} \
    --w2v2-type ${w2v2_type} \
    --ctc-finetuned ${ctc_finetuned} \
    --audio-normalize ${audio_normalize} \
    \
    --length-shrink-cfg "[(1024,2,2)] * 2" \
    --block-size 48 \
    --max-cache-size 576 \
    --xpos 0 \
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