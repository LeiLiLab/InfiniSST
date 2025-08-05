#!/bin/bash

# SONAR Term-Level Control Group Pipeline
# 使用预训练编码器直接评估精准对齐的term-level chunks，不进行训练
# 参数: $1 = single_slice (可选，用于快速验证)

# 设置参数
single_slice=${1:-false}  # 默认使用完整数据集

# 基础路径
BASE_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech"
DATA_DIR="${BASE_DIR}/data"

# 创建日志文件
LOG_FILE="${BASE_DIR}/logs/sonar_term_level_control_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${BASE_DIR}/logs"

echo "=== SONAR Term-Level Control Group Pipeline Started ===" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "Parameters:" | tee -a "$LOG_FILE"
echo "  - single_slice: ${single_slice}" | tee -a "$LOG_FILE"
echo "  - base_dir: ${BASE_DIR}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# === 1. Handle MFA term-level chunks (如果数据不存在) ===
if [[ "$single_slice" == "true" ]]; then
    input_samples="${DATA_DIR}/samples/xl/term_preprocessed_samples_0_500000.json"
    output_samples="${DATA_DIR}/samples/xl/term_level_chunks_single_0_500000.json"
    final_samples="$output_samples"
    
    if [[ ! -f "$output_samples" ]]; then
        echo "[INFO] Step 1: Processing single slice term-level chunks..." | tee -a "$LOG_FILE"
        
        mfa_job=$(sbatch \
            --job-name=term_level_control_single \
            --partition=taurus \
            --mem=32GB \
            --cpus-per-task=4 \
            --ntasks=1 \
            --output="${BASE_DIR}/logs/term_level_control_single_%j.out" \
            --error="${BASE_DIR}/logs/term_level_control_single_%j.err" \
            --wrap="#!/bin/bash

cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
. ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

python3 handle_MFA_term_level_chunks.py \
    --input_json=${input_samples} \
    --output_json=${output_samples} \
    --textgrid_dir=/mnt/data/siqiouyang/datasets/gigaspeech/textgrids \
    --output_audio_dir=/mnt/gemini/data1/jiaxuanluo/term_chunks" | awk '{print $4}')
        
        echo "term_level_control_single: $mfa_job" | tee -a "$LOG_FILE"
        dependency_job=$mfa_job
    else
        echo "[INFO] Step 1: Using existing single slice term-level chunks: $output_samples" | tee -a "$LOG_FILE"
        dependency_job=""
    fi
else
    final_samples="${DATA_DIR}/xl_term_level_chunks_merged.json"
    
    if [[ ! -f "$final_samples" ]]; then
        echo "[INFO] Step 1: Processing full dataset term-level chunks..." | tee -a "$LOG_FILE"
        
        # 检查是否需要生成term-level chunks
        need_generation=false
        for i in {0..16}; do
            start_idx=$((i * 500000))
            if [ $i -eq 16 ]; then
                chunk_file="${DATA_DIR}/samples/xl/term_level_chunks_${start_idx}_end.json"
            else
                end_idx=$((start_idx + 500000))
                chunk_file="${DATA_DIR}/samples/xl/term_level_chunks_${start_idx}_${end_idx}.json"
            fi
            if [[ ! -f "$chunk_file" ]]; then
                need_generation=true
                break
            fi
        done
        
        if [[ "$need_generation" == "true" ]]; then
            # 生成term-level chunks
            mfa_job=$(sbatch ${BASE_DIR}/handle_MFA_term_level_chunks.sh term_preprocessed_samples /mnt/gemini/data1/jiaxuanluo/term_chunks | awk '{print $4}')
            echo "term_level_chunks_generation: $mfa_job" | tee -a "$LOG_FILE"
            
            # 合并数据
            merge_job=$(sbatch \
                --dependency=afterok:$mfa_job \
                --job-name=merge_term_level_control \
                --partition=taurus \
                --nodes=1 \
                --ntasks=1 \
                --cpus-per-task=8 \
                --mem=64GB \
                --output="${BASE_DIR}/logs/merge_term_level_control_%j.out" \
                --error="${BASE_DIR}/logs/merge_term_level_control_%j.err" \
                --wrap="#!/bin/bash

cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
. ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

python3 -c \"
import json, glob
files = sorted(glob.glob('${DATA_DIR}/samples/xl/term_level_chunks_*.json'))
merged = []
for f in files:
    with open(f, encoding='utf-8') as j:
        merged.extend(json.load(j))
print(f'Merged total {len(merged)} term-level samples')
with open('${final_samples}', 'w', encoding='utf-8') as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)
\"" | awk '{print $4}')
            
            echo "merge_term_level_control: $merge_job" | tee -a "$LOG_FILE"
            dependency_job=$merge_job
        else
            # 直接合并现有文件
            echo "[INFO] Term-level chunks exist, merging..." | tee -a "$LOG_FILE"
            merge_job=$(sbatch \
                --job-name=merge_existing_term_level \
                --partition=taurus \
                --nodes=1 \
                --ntasks=1 \
                --cpus-per-task=8 \
                --mem=64GB \
                --output="${BASE_DIR}/logs/merge_existing_term_level_%j.out" \
                --error="${BASE_DIR}/logs/merge_existing_term_level_%j.err" \
                --wrap="#!/bin/bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
. ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst
python3 -c \"
import json, glob
files = sorted(glob.glob('${DATA_DIR}/samples/xl/term_level_chunks_*.json'))
merged = []
for f in files:
    with open(f, encoding='utf-8') as j:
        merged.extend(json.load(j))
print(f'Merged total {len(merged)} term-level samples')
with open('${final_samples}', 'w', encoding='utf-8') as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)
\"" | awk '{print $4}')
            
            echo "merge_existing_term_level: $merge_job" | tee -a "$LOG_FILE"
            dependency_job=$merge_job
        fi
    else
        echo "[INFO] Step 1: Using existing merged term-level chunks: $final_samples" | tee -a "$LOG_FILE"
        dependency_job=""
    fi
fi

# === 2. Run Control Group Evaluation ===
echo "[INFO] Step 2: Running control group evaluation..." | tee -a "$LOG_FILE"

# 设置作业依赖
if [[ -n "$dependency_job" ]]; then
    dependency_option="--dependency=afterok:$dependency_job"
else
    dependency_option=""
fi

eval_job=$(sbatch \
    $dependency_option \
    --job-name=term_level_control_eval \
    --partition=taurus \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --gres=gpu:1 \
    --mem=48GB \
    --output="${BASE_DIR}/logs/term_level_control_eval_%j.out" \
    --error="${BASE_DIR}/logs/term_level_control_eval_%j.err" \
    --wrap="#!/bin/bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
. ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst
python3 SONAR_term_level_control.py \
    --samples_path=${final_samples} \
    --glossary_path=${DATA_DIR}/terms/glossary_filtered.json \
    --max_eval=1000 \
    --audio_batch_size=32 \
    --text_batch_size=512 \
    --output_dir=${DATA_DIR}" | awk '{print $4}')

echo "term_level_control_eval: $eval_job" | tee -a "$LOG_FILE"

# === 总结 ===
echo "" | tee -a "$LOG_FILE"
echo "=== SONAR Term-Level Control Group Pipeline Summary ===" | tee -a "$LOG_FILE"
echo "single_slice: ${single_slice}" | tee -a "$LOG_FILE"
echo "Final samples: ${final_samples}" | tee -a "$LOG_FILE"
echo "Base directory: ${BASE_DIR}" | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "Job IDs:" | tee -a "$LOG_FILE"
if [[ -n "$dependency_job" ]]; then
    if [[ "$single_slice" == "true" ]]; then
        echo "  - Term-level chunks (single): $mfa_job" | tee -a "$LOG_FILE"
    else
        echo "  - Data preparation: $dependency_job" | tee -a "$LOG_FILE"
    fi
fi
echo "  - Control group evaluation: $eval_job" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Monitor progress with:" | tee -a "$LOG_FILE"
echo "  squeue -u \$USER" | tee -a "$LOG_FILE"
echo "  tail -f ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [[ "$single_slice" == "true" ]]; then
    echo "✅ SONAR term-level control group single-slice pipeline submitted!" | tee -a "$LOG_FILE"
    echo "📝 Quick validation mode: using only first 500K samples" | tee -a "$LOG_FILE"
else
    echo "✅ SONAR term-level control group full pipeline submitted!" | tee -a "$LOG_FILE"
    echo "📝 Full dataset mode: processing all samples" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "Key advantages of control group approach:" | tee -a "$LOG_FILE"
echo "  - No training required - direct evaluation with pre-trained encoders" | tee -a "$LOG_FILE"
echo "  - Clean baseline performance on perfectly aligned term chunks" | tee -a "$LOG_FILE"
echo "  - Much faster execution (~30 minutes vs hours of training)" | tee -a "$LOG_FILE"
echo "  - Pure assessment of MFA alignment quality" | tee -a "$LOG_FILE"
echo "  - Upper bound performance for term-level retrieval" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "Usage examples:" | tee -a "$LOG_FILE"
echo "  # Full control group evaluation" | tee -a "$LOG_FILE"
echo "  bash SONAR_term_level_control_pipeline.sh" | tee -a "$LOG_FILE"
echo "  # Single slice quick validation" | tee -a "$LOG_FILE"
echo "  bash SONAR_term_level_control_pipeline.sh true" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "Expected output:" | tee -a "$LOG_FILE"
echo "  - Results will be saved to: ${DATA_DIR}/term_level_control_results.json" | tee -a "$LOG_FILE"
echo "  - Check logs in: ${BASE_DIR}/logs/" | tee -a "$LOG_FILE" 