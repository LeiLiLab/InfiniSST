#!/bin/bash

# Qwen2-Audio Term-Level训练流水线
# 为每个ground truth term生成单独的chunk进行训练
# 参数: $1 = text_field (可选，默认为term), $2 = single_slice (可选，用于快速验证)
#       $3 = audio_text_loss_ratio (可选，默认0.3), $4 = audio_term_loss_ratio (可选，默认0.7)
#       $5 = enable_full_eval (可选，true|false，默认false)
#       $6 = enable_hard_neg (可选，true|false，默认true)
#       $7 = full_eval_every_n_epochs (可选，整数，默认1)
#       $8 = test_samples_path (可选，默认为data/samples/xl/term_level_chunks_500000_1000000.json)
#       $9 = best_model_path (可选，默认data/qwen2_audio_term_level_best.pt)
#       $10 = gpu_ids (可选，指定GPU编号，如"0,1"或"2"，默认为空使用所有可用GPU)

# 设置参数
text_field=${1:-term}  # 默认使用term字段
single_slice=${2:-false}  # 默认使用完整数据集
audio_text_loss_ratio=${3:-0.3}  # 默认使用0.3
audio_term_loss_ratio=${4:-0.7}  # 默认使用0.7
enable_full_eval=${5:-false}
enable_hard_neg=${6:-true}
full_eval_every_n_epochs=${7:-1}
test_samples_path=${8:-"data/samples/xl/term_level_chunks_500000_1000000.json"}  # 默认测试数据集路径
best_model_path=${9:-"data/qwen2_audio_term_level_best.pt"}  # 默认best model路径
gpu_ids=${10:-""}  # GPU编号，默认为空使用所有可用GPU

# 训练数据集路径
TRAIN_TSV="/mnt/data/siqiouyang/datasets/gigaspeech/manifests/train_xl.tsv"

# 创建日志文件
LOG_FILE="logs/qwen2_audio_term_level_pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "=== Qwen2-Audio Term-Level Pipeline Started ===" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "Parameters:" | tee -a "$LOG_FILE"
echo "  - text_field: ${text_field}" | tee -a "$LOG_FILE"
echo "  - single_slice: ${single_slice}" | tee -a "$LOG_FILE"
echo "  - audio_text_loss_ratio: ${audio_text_loss_ratio}" | tee -a "$LOG_FILE"
echo "  - audio_term_loss_ratio: ${audio_term_loss_ratio}" | tee -a "$LOG_FILE"
echo "  - test_samples_path: ${test_samples_path}" | tee -a "$LOG_FILE"
echo "  - enable_hard_neg: ${enable_hard_neg}" | tee -a "$LOG_FILE"
echo "  - enable_full_eval: ${enable_full_eval}" | tee -a "$LOG_FILE"
echo "  - full_eval_every_n_epochs: ${full_eval_every_n_epochs}" | tee -a "$LOG_FILE"
echo "  - best_model_path: ${best_model_path}" | tee -a "$LOG_FILE"
echo "  - gpu_ids: ${gpu_ids:-'auto (all available)'}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# === 1. Handle MFA term-level chunks ===
echo "[INFO] Step 1: Handling MFA term-level chunks..." | tee -a "$LOG_FILE"

if [[ "$single_slice" == "true" ]]; then
    # 单分片term-level处理
    if [[ "$text_field" == "term" ]]; then
        input_samples="data/samples/xl/term_preprocessed_samples_0_500000.json"
        output_samples="data/samples/xl/term_level_chunks_0_500000.json"
    else
        input_samples="data/samples/xl/preprocessed_samples_0_500000.json"
        output_samples="data/samples/xl/term_level_chunks_single_0_500000.json"
    fi
    
    if [[ ! -f "$output_samples" ]]; then
        echo "[INFO] Processing single slice term-level chunks..." | tee -a "$LOG_FILE"
        
        mfa_job=$(sbatch \
            --job-name=term_level_single \
            --partition=taurus \
            --mem=32GB \
            --cpus-per-task=4 \
            --ntasks=1 \
            --output=logs/term_level_single_%j.out \
            --error=logs/term_level_single_%j.err \
            --wrap="#!/bin/bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
. ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst
python3 handle_MFA_term_level_chunks.py \
    --input_json=${input_samples} \
    --output_json=${output_samples} \
    --textgrid_dir=/mnt/data/siqiouyang/datasets/gigaspeech/textgrids \
    --output_audio_dir=/mnt/gemini/data1/jiaxuanluo/term_chunks" | awk '{print $4}')
        
        echo "term_level_single: $mfa_job" | tee -a "$LOG_FILE"
        dependency_job_step1=$mfa_job
    else
        echo "[INFO] Using existing single slice term-level chunks: $output_samples" | tee -a "$LOG_FILE"
        dependency_job_step1=""
    fi
else
    # 完整数据集term-level处理
    final_merged="data/xl_term_level_chunks_merged.json"
    
    if [[ ! -f "$final_merged" ]]; then
        # 检查是否需要生成term-level chunks
        need_generation=false
        for i in {0..16}; do
            start_idx=$((i * 500000))
            if [ $i -eq 16 ]; then
                chunk_file="data/samples/xl/term_level_chunks_${start_idx}_end.json"
            else
                end_idx=$((start_idx + 500000))
                chunk_file="data/samples/xl/term_level_chunks_${start_idx}_${end_idx}.json"
            fi
            if [[ ! -f "$chunk_file" ]]; then
                need_generation=true
                break
            fi
        done
        
        if [[ "$need_generation" == "true" ]]; then
            echo "[INFO] Generating term-level chunks for full dataset..." | tee -a "$LOG_FILE"
            mfa_job=$(sbatch handle_MFA_term_level_chunks.sh ${text_field}_preprocessed_samples /mnt/gemini/data1/jiaxuanluo/term_chunks | awk '{print $4}')
            echo "term_level_chunks_generation: $mfa_job" | tee -a "$LOG_FILE"
            dependency_job_step1=$mfa_job
        else
            echo "[INFO] Term-level chunks exist, skipping generation..." | tee -a "$LOG_FILE"
            dependency_job_step1=""
        fi
    else
        echo "[INFO] Using existing merged term-level chunks: $final_merged" | tee -a "$LOG_FILE"
        dependency_job_step1=""
    fi
fi

# === 2. Merge term-level samples (如果不是单分片模式) ===
if [[ "$single_slice" != "true" ]]; then
    final_samples="data/xl_term_level_chunks_merged.json"
    
    if [[ ! -f "$final_samples" ]]; then
        echo "[INFO] Step 2: Merging term-level processed samples..." | tee -a "$LOG_FILE"
        
        # 设置依赖关系
        if [[ -n "$dependency_job_step1" ]]; then
            dependency_option="--dependency=afterok:$dependency_job_step1"
        else
            dependency_option=""
        fi
        
        merge_job=$(sbatch \
            $dependency_option \
            --job-name=merge_term_level \
            --partition=taurus \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=16 \
            --mem=96GB \
            --output=logs/merge_term_level_%j.out \
            --error=logs/merge_term_level_%j.err \
            --wrap="#!/bin/bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
. ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst
python3 -c \"
import json, glob
files = sorted(glob.glob('data/samples/xl/term_level_chunks_*.json'))
merged = []
for f in files:
    with open(f, encoding='utf-8') as j:
        merged.extend(json.load(j))
print(f'Merged total {len(merged)} term-level samples')
with open('data/xl_term_level_chunks_merged.json', 'w', encoding='utf-8') as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)
\"" | awk '{print $4}')
        
        echo "merge_term_level_samples: $merge_job" | tee -a "$LOG_FILE"
        dependency_job=$merge_job
    else
        echo "[INFO] Step 2: Using existing merged term-level samples: $final_samples" | tee -a "$LOG_FILE"
        dependency_job=$dependency_job_step1
    fi
else
    # 单分片模式直接使用单个文件
    final_samples=$output_samples
    dependency_job=$dependency_job_step1
fi

# === 3. Train Qwen2-Audio model for term-level chunks ===
echo "[INFO] Step 3: Training Qwen2-Audio model for term-level chunks..." | tee -a "$LOG_FILE"

# 根据模式设置模型保存路径
if [[ "$single_slice" == "true" ]]; then
    model_save_path="data/qwen2_audio_term_level_single.pt"
    job_name="qwen2_audio_train_term_level_single"
else
    model_save_path="data/qwen2_audio_term_level_full.pt"
    job_name="qwen2_audio_train_term_level_full"
fi

echo "[INFO] Training Qwen2-Audio model: $model_save_path" | tee -a "$LOG_FILE"

# 设置依赖关系
if [[ -n "$dependency_job" ]]; then
    dependency_option="--dependency=afterok:$dependency_job"
else
    dependency_option=""
fi

# 依据开关构建可选训练/评估参数
extra_flags=""
if [[ "$enable_hard_neg" == "true" ]]; then
  extra_flags+=" --enable_hard_neg"
  extra_flags+=" --hard_neg_source used"  # 默认使用used terms，速度更快
  # 如果需要启用glossary hard negative，取消下面的注释并设置相应参数
  # extra_flags+=" --enable_glossary_hard_neg"
  # extra_flags+=" --hard_neg_source glossary"
  # extra_flags+=" --hard_neg_index_path data/glossary_emb.ivfpq.faiss"
  # extra_flags+=" --hard_neg_term2idx_path data/glossary_term2idx.json"
  # extra_flags+=" --hard_neg_metric ip"
  # extra_flags+=" --hard_neg_nprobe 16"
  # extra_flags+=" --hard_neg_candidates 100"
  extra_flags+=" --hard_neg_k 10"
fi
if [[ "$enable_full_eval" == "true" ]]; then
  extra_flags+=" --enable_full_eval --full_eval_every_n_epochs=${full_eval_every_n_epochs}"
fi

# 构建Python命令
python_cmd="python3 Qwen2_Audio_term_level_train.py \
    --train_samples_path=${final_samples} \
    --test_samples_path=${test_samples_path} \
    --epochs=20 \
    --batch_size=32 \
    --lr=1e-4 \
    --save_path=${model_save_path} \
    --best_model_path=${best_model_path} \
    --audio_text_loss_ratio=${audio_text_loss_ratio} \
    --audio_term_loss_ratio=${audio_term_loss_ratio} \
    --glossary_path=data/terms/glossary_filtered.json \
    --filter_no_term \
    --force_single_gpu \
    --model_name=Qwen/Qwen2-Audio-7B-Instruct \
    --lora_r=16 \
    --lora_alpha=32 \
    --lora_dropout=0.1"

# 如果指定了GPU，则添加GPU参数
if [[ -n "$gpu_ids" ]]; then
    python_cmd+=" --gpu_ids=${gpu_ids}"
fi

# 添加其他额外参数
python_cmd+=" ${extra_flags}"

# 执行命令并获取job ID
train_job=$(sbatch \
    $dependency_option \
    --job-name=$job_name \
    --partition=taurus \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=64GB \
    --gres=gpu:1 \
    --output=logs/${job_name}_%j.out \
    --error=logs/${job_name}_%j.err \
    --wrap="#!/bin/bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
. ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst
$python_cmd" | awk '{print $4}')

echo "qwen2_audio_term_level_train: $train_job" | tee -a "$LOG_FILE"
dependency_job_step3=$train_job

# === 总结 ===
echo "" | tee -a "$LOG_FILE"
echo "=== Qwen2-Audio Term-Level Pipeline Summary ===" | tee -a "$LOG_FILE"
echo "text_field: ${text_field}" | tee -a "$LOG_FILE"
echo "single_slice: ${single_slice}" | tee -a "$LOG_FILE"
echo "Input TSV: ${TRAIN_TSV}" | tee -a "$LOG_FILE"
echo "Final samples: ${final_samples}" | tee -a "$LOG_FILE"
echo "Model save path: ${model_save_path}" | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "Job IDs:" | tee -a "$LOG_FILE"
if [[ "$single_slice" == "true" ]]; then
    if [[ -n "$dependency_job_step1" ]]; then
        echo "  - Term-level chunks (single): $dependency_job_step1" | tee -a "$LOG_FILE"
    else
        echo "  - Term-level chunks (single): Skipped (existing file)" | tee -a "$LOG_FILE"
    fi
    if [[ -n "$dependency_job_step3" && "$dependency_job_step3" != "$dependency_job_step1" ]]; then
        echo "  - Training (single): $dependency_job_step3" | tee -a "$LOG_FILE"
    else
        echo "  - Training (single): Skipped (existing model)" | tee -a "$LOG_FILE"
    fi
else
    if [[ -n "$dependency_job_step1" ]]; then
        echo "  - Term-level chunks (full): $dependency_job_step1" | tee -a "$LOG_FILE"
    else
        echo "  - Term-level chunks (full): Skipped (existing files)" | tee -a "$LOG_FILE"
    fi
    if [[ -n "$merge_job" ]]; then
        echo "  - Merge samples: $merge_job" | tee -a "$LOG_FILE"
    else
        echo "  - Merge samples: Skipped (existing merged file)" | tee -a "$LOG_FILE"
    fi
    if [[ -n "$dependency_job_step3" && "$dependency_job_step3" != "$dependency_job" ]]; then
        echo "  - Training (full): $dependency_job_step3" | tee -a "$LOG_FILE"
    else
        echo "  - Training (full): Skipped (existing model)" | tee -a "$LOG_FILE"
    fi
fi

echo "" | tee -a "$LOG_FILE"
echo "Monitor progress with:" | tee -a "$LOG_FILE"
echo "  squeue -u \$USER" | tee -a "$LOG_FILE"
echo "  tail -f ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [[ "$single_slice" == "true" ]]; then
    echo "✅ Qwen2-Audio term-level single-slice pipeline submitted successfully!" | tee -a "$LOG_FILE"
    echo "📝 Quick validation mode: using only first 500K samples" | tee -a "$LOG_FILE"
else
    echo "✅ Qwen2-Audio term-level full pipeline submitted successfully!" | tee -a "$LOG_FILE"
    echo "📝 Full dataset mode: processing all samples" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "Key features:" | tee -a "$LOG_FILE"
echo "  - Each term gets its own audio chunk (no aggregation)" | tee -a "$LOG_FILE"
echo "  - Perfect MFA alignment for each term" | tee -a "$LOG_FILE"
echo "  - Specialized training for term-level retrieval" | tee -a "$LOG_FILE"
echo "  - Optional rejection capability for no-term samples (disabled by default)" | tee -a "$LOG_FILE"
echo "  - Qwen2-Audio-7B-Instruct based encoder" | tee -a "$LOG_FILE"
echo "  - Multi-GPU support with DataParallel" | tee -a "$LOG_FILE"
echo "  - Hard negative mining (in-memory mode by default, FAISS optional)" | tee -a "$LOG_FILE"
echo "  - Baseline evaluation without noise interference" | tee -a "$LOG_FILE"
echo "  - Intelligent step skipping when files already exist" | tee -a "$LOG_FILE"
echo "  - Automatic dependency management" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "Usage examples:" | tee -a "$LOG_FILE"
echo "  # Full term-level pipeline with default test samples" | tee -a "$LOG_FILE"
echo "  bash Qwen2_Audio_term_level_pipeline.sh term" | tee -a "$LOG_FILE"
echo "  # Single slice quick validation" | tee -a "$LOG_FILE"
echo "  bash Qwen2_Audio_term_level_pipeline.sh term true" | tee -a "$LOG_FILE"
echo "  # Custom test samples path" | tee -a "$LOG_FILE"
echo "  # Enable hard neg & full eval every epoch with custom best model" | tee -a "$LOG_FILE"
echo "  bash Qwen2_Audio_term_level_pipeline.sh term ${single_slice} ${audio_text_loss_ratio} ${audio_term_loss_ratio} true true 1 ${test_samples_path} ${best_model_path}" | tee -a "$LOG_FILE"
echo "  # Disable full eval & hard neg" | tee -a "$LOG_FILE"
echo "  bash Qwen2_Audio_term_level_pipeline.sh term ${single_slice} ${audio_text_loss_ratio} ${audio_term_loss_ratio} false false 1 ${test_samples_path} ${best_model_path}" | tee -a "$LOG_FILE"
echo "  # Load from specific best model" | tee -a "$LOG_FILE"
echo "  bash Qwen2_Audio_term_level_pipeline.sh term false 0.3 0.7 true true 1 ${test_samples_path} data/your_best_model.pt" | tee -a "$LOG_FILE"
echo "  # Specify specific GPUs (e.g., use GPU 0 and 1)" | tee -a "$LOG_FILE"
echo "  bash Qwen2_Audio_term_level_pipeline.sh term false 0.3 0.7 true true 1 ${test_samples_path} ${best_model_path} \"0,1\"" | tee -a "$LOG_FILE"
echo "  # Use only GPU 2" | tee -a "$LOG_FILE"
echo "  bash Qwen2_Audio_term_level_pipeline.sh term false 0.3 0.7 true true 1 ${test_samples_path} ${best_model_path} \"2\"" | tee -a "$LOG_FILE"
