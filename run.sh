cd train

source $HOME/sllama/bin/activate

# mkdir -p /scratch/xixu/

# rm -rf /scratch/xixu/*

# echo "Copying dataset."
# /usr/bin/cp -f /data/user_data/yuanjinw/dataset.tar.zst /scratch/xixu/
# echo "Dataset copied."

# echo "Extracting dataset."
# zstd --ultra -1 -d /scratch/xixu/dataset.tar.zst --stdout | tar axf - -C /scratch/xixu/
# # tar -axf /scratch/siqiouya/dataset.tar.zst -C /scratch/siqiouya/
# echo "Dataset extracted."

llm_model=meta-llama/Llama-2-7b-hf
ssl_model=/data/user_data/yuanjinw/models/hubert_large_ll60k_finetune_ls960.pt
speech_encoder=hubert
data_path=/scratch/xixu/dataset/must-c-v1.0/en-es
name=hubert-stage1
save_path=/scratch/xixu/runs/$name
mkdir -p ${save_path}

export PYTHONPATH=/home/yuanjinw/work/sllama
SLURM_GPUS=8

python correct_path.py
export CUDA_VISIBLE_DEVICE=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=$SLURM_GPUS \
    stage1.py \
    --model_name_or_path ${llm_model} \
    --speech_tower_path ${ssl_model} \
    --speech_tower_type ${speech_encoder} \
    --ssl_fintuned True \
    --data_path ${data_path} \
    --data_split_train 'train' \
    --data_split_eval 'dev' \
    --freeze_speech_foundation False \
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
    --save_total_limit 4 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.2 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --seed 998244353 \
    --report_to wandb \
    --fp16 True \
    --deepspeed ../configs/deepspeed_config.json \

# # stage2 train
llm_model=$save_path/checkpoint-12000
name=hubert-stage2
save_path=/scratch/xixu/runs/$name

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
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 200 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.2 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --seed 998244353 \
    --report_to wandb \
    --fp16 True \
    --deepspeed ../configs/deepspeed_config_stage3.json \