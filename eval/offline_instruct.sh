llm_model=/compute/babel-5-23/siqiouya/runs/en-zh/8B-s2-bi-v3.5/last.ckpt/
# llm_model=/compute/babel-5-23/siqiouya/runs/8B-traj-s2-v2.2/last.ckpt/
w2v2_path=/data/user_data/siqiouya/runs/pretrained/wav2_vec_vox_960h_pl.pt
# w2v2_path=/data/user_data/siqiouya/runs/pretrained/hubert_large_ll60k_finetune_ls960.pt
# w2v2_path=/data/user_data/siqiouya/runs/pretrained/w2v-bert-2.0
w2v2_type=w2v2
# data_path=/compute/babel-6-17/xixu/datasets/must-c-v1.0/en-de
# data_path=/compute/babel-6-17/xixu/datasets/must-c-v2.0/en-zh
data_path=/compute/babel-14-5/siqiouya/en-zh
source_lang=English
# target_lang=German
target_lang=Chinese
xpos=0

beam=4

# python /home/siqiouya/work/sllama/train/zero_to_fp32.py ${llm_model} ${llm_model}/pytorch_model.bin
# python /home/siqiouya/work/sllama/train/prune_bin.py ${llm_model}/pytorch_model.bin

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/siqiouya/work/sllama
export CUDA_VISIBLE_DEVICES=0

# ---------- causal -----------
# python /home/siqiouya/work/sllama/eval/test_dataset_instruct.py \
#     --w2v2-path ${w2v2_path} \
#     --w2v2-type ${w2v2_type} \
#     --ctc-finetuned \
#     --length-shrink-cfg "[(1024,2,2)] * 2" \
#     --block-size 48 \
#     --max-cache-size 500 \
#     --xpos ${xpos} \
#     \
#     --model-name /compute/babel-4-1/siqiouya/llama-3.1-8b-instruct-hf \
#     --state-dict-path ${llm_model}/pytorch_model.bin \
#     --data-path ${data_path} \
#     --data-split tst-COMMON \
#     \
#     --beam ${beam} \
#     --result ${llm_model}/offline_beam${beam} \
#     --batch-size 3000 \
#     \
#     --source-lang ${source_lang} \
#     --target-lang ${target_lang}

# ---------- bi -----------
python /home/siqiouya/work/sllama/eval/test_dataset_instruct.py \
    --w2v2-path ${w2v2_path} \
    --w2v2-type ${w2v2_type} \
    --ctc-finetuned \
    --length-shrink-cfg "[(1024,2,2)] * 2" \
    --block-size 100000000 \
    --max-cache-size 100000000 \
    --xpos ${xpos} \
    \
    --model-name /compute/babel-4-1/siqiouya/llama-3.1-8b-instruct-hf \
    --state-dict-path ${llm_model}/pytorch_model.bin \
    --data-path ${data_path} \
    --data-split tst-COMMON \
    \
    --beam ${beam} \
    --result ${llm_model}/offline_beam${beam} \
    --batch-size 2500 \
    \
    --source-lang ${source_lang} \
    --target-lang ${target_lang}

python /home/siqiouya/work/sllama/eval/compute_bleu.py \
    ${llm_model}/offline_beam${beam}/tst-COMMON