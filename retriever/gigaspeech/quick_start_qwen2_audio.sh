#!/bin/bash

# Qwen2-Audio Term-Level Training Quick Start Script
# 快速启动Qwen2-Audio术语级训练的便捷脚本

set -e  # 遇到错误立即退出

echo "🚀 Qwen2-Audio Term-Level Training Quick Start"
echo "=============================================="

# 检查当前目录
if [[ ! -f "Qwen2_Audio_term_level_train.py" ]]; then
    echo "❌ Error: Please run this script from the retriever/gigaspeech directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected files: Qwen2_Audio_term_level_train.py"
    exit 1
fi

# 检查conda环境
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "⚠️  Warning: No conda environment detected"
    echo "   Please activate your environment first: conda activate infinisst"
else
    echo "✅ Conda environment: $CONDA_DEFAULT_ENV"
fi

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | head -2
else
    echo "⚠️  Warning: nvidia-smi not found, may not have CUDA support"
fi

echo ""
echo "📋 Available options:"
echo "1. Quick test (single slice, fast validation)"
echo "2. Full training (complete dataset)"
echo "3. Term-only training (disable no-term samples)"
echo "4. Hard negative mining training"
echo "5. Custom configuration"
echo "6. Run integration test only"
echo "7. Exit"

read -p "Please select an option (1-7): " choice

case $choice in
    1)
        echo ""
        echo "🎯 Starting quick test mode..."
        echo "   - Using single slice (first 500K samples)"
        echo "   - Fast validation mode"
        echo "   - Recommended for first-time users"
        echo ""
        
        # 检查必要文件
        if [[ ! -f "data/samples/xl/term_preprocessed_samples_0_500000.json" ]]; then
            echo "⚠️  Warning: term_preprocessed_samples_0_500000.json not found"
            echo "   The pipeline will attempt to generate it automatically"
        fi
        
        read -p "Continue with quick test? (y/N): " confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            bash Qwen2_Audio_term_level_pipeline.sh term true 0.3 0.7 false
        else
            echo "Operation cancelled."
        fi
        ;;
        
    2)
        echo ""
        echo "🎯 Starting full training mode..."
        echo "   - Using complete dataset"
        echo "   - Full evaluation enabled"
        echo "   - Estimated time: several hours"
        echo ""
        
        # 检查必要文件
        missing_files=()
        if [[ ! -f "data/xl_term_level_chunks_merged.json" ]]; then
            missing_files+=("data/xl_term_level_chunks_merged.json")
        fi
        if [[ ! -f "data/terms/glossary_filtered.json" ]]; then
            missing_files+=("data/terms/glossary_filtered.json")
        fi
        
        if [[ ${#missing_files[@]} -gt 0 ]]; then
            echo "⚠️  Warning: Missing files:"
            for file in "${missing_files[@]}"; do
                echo "   - $file"
            done
            echo "   The pipeline will attempt to generate them automatically"
        fi
        
        read -p "Continue with full training? (y/N): " confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            bash Qwen2_Audio_term_level_pipeline.sh term false 0.3 0.7 true
        else
            echo "Operation cancelled."
        fi
        ;;
        
    3)
        echo ""
        echo "🎯 Starting term-only training mode..."
        echo "   - Using complete dataset"
        echo "   - No-term samples disabled"
        echo "   - Focus on term retrieval only"
        echo ""
        
        read -p "Continue with term-only training? (y/N): " confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            bash Qwen2_Audio_term_level_pipeline.sh term false 0.3 0.7 false "data/samples/xl/term_level_chunks_500000_1000000.json" "data/qwen2_audio_term_level_best.pt" "" "Qwen/Qwen2-Audio-7B-Instruct" false false
        else
            echo "Operation cancelled."
        fi
        ;;
        
    4)
        echo ""
        echo "🎯 Starting hard negative mining training..."
        echo "   - Using complete dataset"
        echo "   - Hard negative mining enabled"
        echo "   - Enhanced contrastive learning"
        echo ""
        
        # 检查FAISS索引文件
        if [[ ! -f "data/glossary_emb.ivfpq.faiss" ]]; then
            echo "⚠️  Warning: FAISS index file not found (data/glossary_emb.ivfpq.faiss)"
            echo "   Hard negative mining will use in-memory mode (slower)"
        else
            echo "✅ FAISS index found, using optimized hard negative mining"
        fi
        
        read -p "Continue with hard negative mining training? (y/N): " confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            bash Qwen2_Audio_term_level_pipeline.sh term false 0.3 0.7 false "data/samples/xl/term_level_chunks_500000_1000000.json" "data/qwen2_audio_term_level_best.pt" "" "Qwen/Qwen2-Audio-7B-Instruct" true true
        else
            echo "Operation cancelled."
        fi
        ;;
        
    5)
        echo ""
        echo "🎯 Custom configuration mode..."
        echo ""
        
        # 收集用户输入
        read -p "Text field (default: term): " text_field
        text_field=${text_field:-term}
        
        read -p "Single slice mode? (true/false, default: false): " single_slice
        single_slice=${single_slice:-false}
        
        read -p "Audio-text loss ratio (default: 0.3): " audio_text_ratio
        audio_text_ratio=${audio_text_ratio:-0.3}
        
        read -p "Audio-term loss ratio (default: 0.7): " audio_term_ratio
        audio_term_ratio=${audio_term_ratio:-0.7}
        
        read -p "Enable full evaluation? (true/false, default: false): " enable_eval
        enable_eval=${enable_eval:-false}
        
        read -p "GPU IDs (e.g., '0,1' or '2', default: auto): " gpu_ids
        gpu_ids=${gpu_ids:-""}
        
        read -p "Enable no-term samples? (true/false, default: true): " enable_no_term
        enable_no_term=${enable_no_term:-true}
        
        read -p "Enable hard negative mining? (true/false, default: false): " enable_hard_neg
        enable_hard_neg=${enable_hard_neg:-false}
        
        echo ""
        echo "Configuration summary:"
        echo "  - Text field: $text_field"
        echo "  - Single slice: $single_slice"
        echo "  - Audio-text ratio: $audio_text_ratio"
        echo "  - Audio-term ratio: $audio_term_ratio"
        echo "  - Full evaluation: $enable_eval"
        echo "  - GPU IDs: ${gpu_ids:-auto}"
        echo "  - No-term samples: $enable_no_term"
        echo "  - Hard negative mining: $enable_hard_neg"
        echo ""
        
        read -p "Start training with this configuration? (y/N): " confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            bash Qwen2_Audio_term_level_pipeline.sh "$text_field" "$single_slice" "$audio_text_ratio" "$audio_term_ratio" "$enable_eval" "data/samples/xl/term_level_chunks_500000_1000000.json" "data/qwen2_audio_term_level_best.pt" "$gpu_ids" "Qwen/Qwen2-Audio-7B-Instruct" "$enable_no_term" "$enable_hard_neg"
        else
            echo "Operation cancelled."
        fi
        ;;
        
    6)
        echo ""
        echo "🧪 Running integration test..."
        python test_qwen2_audio.py
        ;;
        
    7)
        echo "Goodbye! 👋"
        exit 0
        ;;
        
    *)
        echo "❌ Invalid option. Please select 1-7."
        exit 1
        ;;
esac

echo ""
echo "✅ Script completed!"
echo ""
echo "📚 For more information, see:"
echo "   - README_Qwen2_Audio.md (detailed documentation)"
echo "   - test_qwen2_audio.py (integration testing)"
echo "   - Qwen2_Audio_term_level_pipeline.sh (full pipeline)"
echo ""
echo "🔍 Monitor training progress:"
echo "   - squeue -u \$USER (check job status)"
echo "   - tail -f logs/qwen2_audio_term_level_pipeline_*.log (view logs)"
