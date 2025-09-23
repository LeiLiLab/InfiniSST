#!/bin/bash

# 直接评估脚本示例
# 使用已有的checkpoint进行评估，跳过训练过程

echo "=== Direct Evaluation Example ==="
echo "Using checkpoint: data/clap_term_level_epoch1.pt"

# 基本直接评估（使用训练时的术语集）
python3 SONAR_term_level_train_glossary.py \
    --direct_evaluate \
    --checkpoint_path=data/clap_term_level_epoch1.pt \
    --train_samples_path=data/xl_term_level_chunks_merged.json \
    --test_samples_path=data/samples/xl/term_level_chunks_500000_1000000.json \
    --glossary_path=data/terms/glossary_filtered.json \
    --filter_no_term

echo ""
echo "=== Direct Evaluation with Full Glossary ==="

# 带完整词汇表评估的直接评估
python3 SONAR_term_level_train_glossary.py \
    --direct_evaluate \
    --checkpoint_path=data/clap_term_level_epoch1.pt \
    --train_samples_path=data/xl_term_level_chunks_merged.json \
    --test_samples_path=data/samples/xl/term_level_chunks_500000_1000000.json \
    --glossary_path=data/terms/glossary_filtered.json \
    --enable_full_eval \
    --filter_no_term

echo ""
echo "=== Direct Evaluation with GPU Selection ==="

# 指定GPU的直接评估
python3 SONAR_term_level_train_glossary.py \
    --direct_evaluate \
    --checkpoint_path=data/clap_term_level_epoch1.pt \
    --train_samples_path=data/xl_term_level_chunks_merged.json \
    --test_samples_path=data/samples/xl/term_level_chunks_500000_1000000.json \
    --glossary_path=data/terms/glossary_filtered.json \
    --enable_full_eval \
    --filter_no_term \
    --gpu_ids="0"

echo ""
echo "Direct evaluation examples completed!"
echo ""
echo "Usage:"
echo "  # Basic evaluation"
echo "  python3 SONAR_term_level_train_glossary.py --direct_evaluate --checkpoint_path=data/clap_term_level_epoch1.pt"
echo ""
echo "  # With full glossary evaluation"
echo "  python3 SONAR_term_level_train_glossary.py --direct_evaluate --checkpoint_path=data/clap_term_level_epoch1.pt --enable_full_eval"
echo ""
echo "  # With specific GPU"
echo "  python3 SONAR_term_level_train_glossary.py --direct_evaluate --checkpoint_path=data/clap_term_level_epoch1.pt --gpu_ids=\"0\""


