llm_model=$1
w2v2_path=$2
w2v2_type=$3
data_path=$4
source_lang=$5
target_lang=$6
xpos=$7

beam=4

python /home/siqiouya/work/sllama/train/zero_to_fp32.py ${llm_model} ${llm_model}/pytorch_model.bin

python /home/siqiouya/work/sllama/train/extract_adapter.py \
    --model_name_or_path ${llm_model} \
    --extracted_name 'speech_encoder' \
    --output ${llm_model}/speech_encoder.bin

python /home/siqiouya/work/sllama/eval/test_dataset_new.py \
    --w2v2-path ${w2v2_path} \
    --w2v2-type ${w2v2_type} \
    --ctc-finetuned \
    --length-shrink-cfg "[(1024,2,2)] * 2" \
    --block-size 48 \
    --max-cache-size 500 \
    --xpos ${xpos} \
    \
    --model-name ${llm_model} \
    --data-path ${data_path} \
    --data-split tst-COMMON \
    \
    --beam ${beam} \
    --result ${llm_model}/offline_beam${beam} \
    --batch-size 4 \
    \
    --source-lang ${source_lang} \
    --target-lang ${target_lang}

python /home/siqiouya/work/sllama/eval/compute_bleu.py \
    ${llm_model}/offline_beam${beam}/tst-COMMON