PYTHONPATH=/home/siqiouya/work/sllama simuleval \
    --agent eval/agents/streamllama.py \
    --source-segment-size 960 \
    --source-lang English \
    --target-lang Chinese \
    --min-start-sec 1.92 \
    --source /compute/babel-6-17/xixu/datasets/must-c-v2.0/en-zh/tst-COMMON.source \
    --target /compute/babel-6-17/xixu/datasets/must-c-v2.0/en-zh/tst-COMMON.target \
    --output debug-streamllama \
    --w2v2-path /data/user_data/siqiouya/runs/pretrained/wav2_vec_vox_960h_pl.pt \
    --w2v2-type w2v2 \
    --ctc-finetuned \
    --length-shrink-cfg "[(1024,2,2)] * 2" \
    --block-size 48 \
    --max-cache-size 500 \
    --xpos 0 \
    \
    --max-len-a 10 \
    --max-len-b 20 \
    --beam 1 \
    --no-repeat-ngram-size 3 \
    --repetition-penalty 1.2 \
    \
    --model-name /compute/babel-4-1/siqiouya/llama-3.1-8b-hf \
    --state-dict-path "/compute/babel-5-23/siqiouya/runs/8B-traj-s2-v1.2/epoch=0-step=1213.ckpt/pytorch_model.bin" \
    \
    --quality-metrics BLEU \
    --eval-latency-unit char \
    --sacrebleu-tokenizer zh