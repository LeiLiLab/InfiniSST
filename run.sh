cd train

llm_model=/mnt/data/xixu/llm/llama-2-7b/hf
ssl_model=/mnt/data/xixu/models/wav2_vec_vox_960h_pl.pt
data_path=/mnt/data/xixu/datasets/must-c-v1.0/en-es
save_path=/mnt/data/xixu/runs/sllama/en-es/7b/run2/stage1

torchrun --nproc_per_node=$SLURM_GPUS \
    stage1.py \
    --model_name_or_path ${llm_model} \
    --speech_tower_path ${ssl_model} \
    --ssl_fintuned True \
    --data_path ${data_path} \
    --data_split_train 'train' \
    --data_split_eval 'dev' \
    --freeze_speech_foundation True \
    --freeze_backbone True \
    --only_tune_adapter True \
    --output_dir ${save_path} \
    --num_train_epochs  6 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --seed 1234 \
    --report_to none \
    --fp16 True \
    --deepspeed ../configs/deepspeed_config.json

# # stage2 train
llm_model=/mnt/data/xixu/runs/sllama/en-es/7b/run2/stage1/
ssl_model=/mnt/data/xixu/models/wav2_vec_vox_960h_pl.pt
data_path=/mnt/data/xixu/datasets/must-c-v1.0/en-es
save_path=/mnt/data/xixu/runs/sllama/en-es/7b/run2/stage2

python ./zero_to_fp32.py ${llm_model}/checkpoint-12000 ${llm_model}/checkpoint-12000/pytorch_model.bin

model_path=/mnt/data/xixu/runs/sllama/en-es/13b/stage1
python ./extract_adapter.py \
  --model_name_or_path ${model_path}/checkpoint-12000 \
  --extracted_name 'mm_length_adapter' \
  --output ${model_path}/length_adapter.bin 
python ./extract_adapter.py \
  --model_name_or_path ${model_path}/checkpoint-12000 \
  --extracted_name 'mm_mlp_adapter' \
  --output ${model_path}/mlp_adapter.bin 

torchrun  --nproc_per_node=$SLURM_GPUS\
    stage2_large.py \
    --model_name_or_path ${llm_model} \
    --speech_tower_path ${ssl_model} \
    --ssl_fintuned True \
    --data_path ${data_path} \
    --data_split_train 'train' \
    --data_split_eval 'dev' \
    --freeze_speech_foundation True \
    --freeze_backbone False \
    --only_tune_adapter False \
    --output_dir ${save_path} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 6 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --seed 1234 \
    --report_to none \
    --fp16 True \
    --deepspeed ../configs/deepspeed_config_stage3.json

