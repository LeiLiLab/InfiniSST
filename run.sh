cd train

export CUDA_VISIBLE_DEVICES=4,5,6,7

llm_model=/mnt/taurus/data/xixu/llm/llama-2-7b/hf
ssl_model=/mnt/taurus/data/xixu/models/wav2_vec_vox_960h_pl.pt
data_path=/mnt/taurus/data/xixu/datasets/must-c-v1.0/en-fr/
save_path_xis=/mnt/taurus/data/xixu/runs/sllama/en-fr/7b/wWav2vec/stage1
save_path=/mnt/taurus/data1/chinmay/sllama/en-fr/7b/wWav2vec/stage1


torchrun --nproc_per_node=1 \
    stage1.py \
    --model_name_or_path ${llm_model} \
    --speech_tower_path ${ssl_model} \
    --ssl_fintuned True \
    --data_path ${data_path} \
    --data_split_train 'train' \
    --data_split_eval 'dev' \
    --freeze_speech_foundation False \
    --freeze_backbone True \
    --only_tune_adapter True \
    --output_dir ${save_path} \
    --num_train_epochs  6 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.2 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --seed 998244353 \
    --report_to none \
    --fp16 True \
    --deepspeed ../configs/deepspeed_config.json \

# # stage2 train
llm_model=/mnt/taurus/data1/chinmay/sllama/en-fr/7b/wWav2vec/stage1/checkpoint-13000
data_path=/mnt/taurus/data/xixu/datase /must-c-v1.0/en-fr
save_path=/mnt/taurus/data1/chinmay/sllama/en-fr/7b/wWav2vec/stage2
save_path_xis=/mnt/taurus/data/xixu/runs/sllama/7b/wWav2vec/stage2

python ./zero_to_fp32.py ${llm_model} ${llm_model}/pytorch_model.bin


python ./extract_adapter.py \
  --model_name_or_path ${llm_model} \
  --extracted_name 'mm_length_adapter' \
  --output ${llm_model}/length_adapter.bin 
  
python ./extract_adapter.py \
  --model_name_or_path ${llm_model} \
  --extracted_name 'mm_mlp_adapter' \
  --output ${llm_model}/mlp_adapter.bin 

python ./extract_adapter.py \
    --model_name_or_path ${llm_model} \
    --extracted_name 'speech_tower' \
    --output ${llm_model}/speech_tower.bin

ssl_model=${llm_model}/speech_tower.bin

torchrun  --nproc_per_node=$SLURM_GPUS\
    stage2_large.py \
    --model_name_or_path ${llm_model} \
    --speech_tower_path ${ssl_model} \
    --ssl_fintuned True \
    --data_path ${data_path} \
    --data_split_train 'train' \
    --data_split_eval 'dev' \
    --freeze_speech_foundation False \
    --freeze_backbone False \
    --only_tune_adapter False \
    --output_dir ${save_path} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 200 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.2 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --seed 998244353 \
    --report_to none \
    --fp16 True \
    --deepspeed ../configs/deepspeed_config_stage3.json \
