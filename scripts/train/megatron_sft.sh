lang_code=zh

TAGS=(
    "v4_ner_baseline_aligned_rate1.0_k20_final"
    # "origin"
)

for TAG in "${TAGS[@]}"; do
    export tag=${TAG}
    export train_dataset=/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests_rag/train_s_${lang_code}_${tag}.jsonl
    # export BASE_MODEL_PATH=/data/user_data/siqiouya/ckpts/pretrained/llm/Qwen3-Omni-30B-A3B-Instruct-mcore/
    # export BASE_MODEL_PATH=/data/user_data/siqiouya/ckpts/pretrained/llm/Qwen3-Omni-30B-A3B-Instruct-mcore/

    # swift export \
    #     --model /data/user_data/siqiouya/ckpts/infinisst-omni/gigaspeech-zh-s_origin-bsz4/v1-20260122-055820-hf/ \
    #     --to_mcore true \
    #     --torch_dtype bfloat16 \
    #     --output_dir /data/user_data/siqiouya/ckpts/infinisst-omni/gigaspeech-zh-s_origin-bsz4/v1-20260122-055820-mcore
    export BASE_MODEL_PATH=/data/user_data/siqiouya/ckpts/infinisst-omni/gigaspeech-zh-s_origin-bsz4/v1-20260122-055820-mcore/

    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
    NPROC_PER_NODE=4 \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    ENABLE_AUDIO_OUTPUT=False \
    megatron sft \
        --load ${BASE_MODEL_PATH} \
        --dataset ${train_dataset} \
        --split_dataset_ratio 0.01 \
        --load_from_cache_file true \
        --train_type lora \
        --lora_rank 32 \
        --lora_alpha 32 \
        --target_modules all-linear \
        --freeze_llm false \
        --freeze_vit true \
        --freeze_aligner true \
        --vit_gradient_checkpointing false \
        --packing true \
        --expert_model_parallel_size 4 \
        --moe_permute_fusion true \
        --moe_grouped_gemm true \
        --moe_shared_expert_overlap true \
        --moe_aux_loss_coeff 1e-3 \
        --micro_batch_size 1 \
        --global_batch_size 4 \
        --recompute_granularity full \
        --recompute_method uniform \
        --recompute_num_layers 1 \
        --finetune true \
        --cross_entropy_loss_fusion true \
        --lr 1e-4 \
        --lr_warmup_fraction 0.05 \
        --min_lr 1e-5 \
        --weight_decay 0.01 \
        --clip_grad 1.0 \
        --max_epochs 1 \
        --save /data/user_data/siqiouya/ckpts/infinisst-omni/gigaspeech-${lang_code}-s_origin-${tag}-bsz4/ \
        --log_interval 10 \
        --eval_interval 200 \
        --save_interval 200 \
        --max_length 2048 \
        --num_workers 8 \
        --dataset_num_proc 8 \
        --no_save_optim true \
        --no_save_rng true \
        --attention_backend flash \
        --wandb_project gigaspeech_${lang_code} \
        --wandb_exp_name omni-gigaspeech-${lang_code}-s_origin-${tag}-bsz4
      
    BASE_DIR=/data/user_data/siqiouya/ckpts/infinisst-omni/gigaspeech-${lang_code}-s_origin-${TAG}-bsz4
    LATEST_CKPT=$(ls -td "$BASE_DIR"/v*-* 2>/dev/null | head -n 1)
    
    if [ -z "$LATEST_CKPT" ]; then
        echo "Warning: No checkpoint found for ${TAG}"
        continue
    fi
    
    echo "Exporting checkpoint: $LATEST_CKPT"
    
    swift export \
        --mcore_adapters "${LATEST_CKPT}/" \
        --to_hf true \
        --torch_dtype bfloat16 \
        --output_dir "${LATEST_CKPT}-hf/"
    
    # swift export \
    #     --mcore_adapters "${LATEST_CKPT}/" \
    #     --to_mcore true \
    #     --torch_dtype bfloat16 \
    #     --output_dir "${LATEST_CKPT}-mcore/"
    
    hf upload owaski/gigaspeech-${lang_code}-s_origin-${TAG}-bsz4 ${LATEST_CKPT}-hf
done