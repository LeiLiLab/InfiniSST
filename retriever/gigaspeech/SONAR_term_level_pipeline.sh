#!/bin/bash

# SONAR Term-Level训练流水线
# 为每个ground truth term生成单独的chunk进行训练
# 参数: $1 = text_field (可选，默认为term), $2 = single_slice (可选，用于快速验证)

# 设置参数
text_field=${1:-term}  # 默认使用term字段
single_slice=${2:-false}  # 默认使用完整数据集

# 训练数据集路径
TRAIN_TSV="/mnt/data/siqiouyang/datasets/gigaspeech/manifests/train_xl.tsv"

# 创建日志文件
LOG_FILE="logs/sonar_term_level_pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "=== SONAR Term-Level Pipeline Started ===" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "Parameters:" | tee -a "$LOG_FILE"
echo "  - text_field: ${text_field}" | tee -a "$LOG_FILE"
echo "  - single_slice: ${single_slice}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# === 1. Handle MFA term-level chunks ===
echo "[INFO] Step 1: Handling MFA term-level chunks..." | tee -a "$LOG_FILE"

if [[ "$single_slice" == "true" ]]; then
    # 单分片term-level处理
    if [[ "$text_field" == "term" ]]; then
        input_samples="data/samples/xl/term_preprocessed_samples_0_500000.json"
        output_samples="data/samples/xl/term_level_chunks_single_0_500000.json"
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

# === 3. Train SONAR model for term-level chunks ===
echo "[INFO] Step 3: Training SONAR model for term-level chunks..." | tee -a "$LOG_FILE"

# 根据模式设置模型保存路径
if [[ "$single_slice" == "true" ]]; then
    model_save_path="data/clap_sonar_term_level_single.pt"
    job_name="sonar_train_term_level_single"
else
    model_save_path="data/clap_sonar_term_level_full.pt"
    job_name="sonar_train_term_level_full"
fi

echo "[INFO] Training SONAR model: $model_save_path" | tee -a "$LOG_FILE"

# 设置依赖关系
if [[ -n "$dependency_job" ]]; then
    dependency_option="--dependency=afterok:$dependency_job"
else
    dependency_option=""
fi

# 创建适配term-level数据格式的训练脚本
train_job=$(sbatch \
    $dependency_option \
    --job-name=$job_name \
    --partition=taurus \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --gres=gpu:1 \
    --mem=32GB \
    --output=logs/${job_name}_%j.out \
    --error=logs/${job_name}_%j.err \
    --wrap="#!/bin/bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
. ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst
python3 SONAR_term_level_train.py \
    --train_samples_path=${final_samples} \
    --epochs=20 \
    --batch_size=512 \
    --lr=5e-5 \
    --save_path=${model_save_path} \
    --glossary_path=data/terms/glossary_filtered.json" | awk '{print $4}')

echo "sonar_term_level_train: $train_job" | tee -a "$LOG_FILE"
dependency_job_step3=$train_job
# === 总结 ===
echo "" | tee -a "$LOG_FILE"
echo "=== SONAR Term-Level Pipeline Summary ===" | tee -a "$LOG_FILE"
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
    echo "  - Evaluation: $eval_job" | tee -a "$LOG_FILE"
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
    echo "  - Evaluation: $eval_job" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "Monitor progress with:" | tee -a "$LOG_FILE"
echo "  squeue -u \$USER" | tee -a "$LOG_FILE"
echo "  tail -f ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [[ "$single_slice" == "true" ]]; then
    echo "✅ SONAR term-level single-slice pipeline submitted successfully!" | tee -a "$LOG_FILE"
    echo "📝 Quick validation mode: using only first 500K samples" | tee -a "$LOG_FILE"
else
    echo "✅ SONAR term-level full pipeline submitted successfully!" | tee -a "$LOG_FILE"
    echo "📝 Full dataset mode: processing all samples" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "Key features:" | tee -a "$LOG_FILE"
echo "  - Each term gets its own audio chunk (no aggregation)" | tee -a "$LOG_FILE"
echo "  - Perfect MFA alignment for each term" | tee -a "$LOG_FILE"
echo "  - Specialized training for term-level retrieval" | tee -a "$LOG_FILE"
echo "  - Baseline evaluation without noise interference" | tee -a "$LOG_FILE"
echo "  - Intelligent step skipping when files already exist" | tee -a "$LOG_FILE"
echo "  - Automatic dependency management" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "Usage examples:" | tee -a "$LOG_FILE"
echo "  # Full term-level pipeline" | tee -a "$LOG_FILE"
echo "  bash SONAR_term_level_pipeline.sh term" | tee -a "$LOG_FILE"
echo "  # Single slice quick validation" | tee -a "$LOG_FILE"
echo "  bash SONAR_term_level_pipeline.sh term true" | tee -a "$LOG_FILE" 