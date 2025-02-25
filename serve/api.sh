export PYTHONPATH=/home/siqiouya/work/sllama
python api.py \
    --latency-multiplier 2 \
    --min-start-sec 0 \
    --w2v2-path /data/user_data/siqiouya/runs/pretrained/wav2_vec_vox_960h_pl.pt \
    --w2v2-type w2v2 \
    --ctc-finetuned \
    \
    --length-shrink-cfg "[(1024,2,2)] * 2" \
    --block-size 48 \
    --max-cache-size 576 \
    --xpos 0 \
    \
    --max-llm-cache-size 1000 \
    --always-cache-system-prompt \
    \
    --max-len-a 10 \
    --max-len-b 20 \
    --max-new-tokens 10 \
    --beam 4 \
    --no-repeat-ngram-lookback 100 \
    --no-repeat-ngram-size 5 \
    --repetition-penalty 1.2 \
    --suppress-non-language \
    \
    --model-name /compute/babel-4-1/siqiouya/llama-3.1-8b-instruct-hf