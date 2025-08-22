#!/bin/bash

# SONAR Term-Level Training 使用示例脚本
# 展示如何使用各种参数进行训练

echo "=== SONAR Term-Level Training Examples ==="
echo ""

# 示例1: 基础训练（从随机初始化开始）
echo "1. 基础训练（随机初始化）:"
echo "bash SONAR_term_level_pipeline_glossary.sh term false 0.3 0.7 false false 1"
echo ""

# 示例2: 从best model继续训练，启用hard negative mining
echo "2. 从best model继续训练，启用hard negative mining:"
echo "bash SONAR_term_level_pipeline_glossary.sh term false 0.3 0.7 false true 1 data/samples/xl/term_level_chunks_500000_1000000.json data/clap_sonar_full_n2_best.pt"
echo ""

# 示例3: 完整配置（hard neg + full eval + custom best model）
echo "3. 完整配置（hard neg + full eval + custom best model）:"
echo "bash SONAR_term_level_pipeline_glossary.sh term false 0.3 0.7 true true 1 data/samples/xl/term_level_chunks_500000_1000000.json data/clap_sonar_full_n2_best.pt"
echo ""

# 示例4: 单分片快速验证
echo "4. 单分片快速验证:"
echo "bash SONAR_term_level_pipeline_glossary.sh term true 0.3 0.7 false true 1 data/samples/xl/term_level_chunks_500000_1000000.json data/clap_sonar_full_n2_best.pt"
echo ""

# 示例5: 自定义损失权重
echo "5. 自定义损失权重（强化audio-term loss）:"
echo "bash SONAR_term_level_pipeline_glossary.sh term false 0.1 0.9 false true 1 data/samples/xl/term_level_chunks_500000_1000000.json data/clap_sonar_full_n2_best.pt"
echo ""

echo "=== 参数说明 ==="
echo "参数顺序: text_field single_slice audio_text_loss_ratio audio_term_loss_ratio enable_full_eval enable_hard_neg full_eval_every_n_epochs test_samples_path best_model_path"
echo ""
echo "text_field: 文本字段类型 (term)"
echo "single_slice: 是否使用单分片 (true/false)"
echo "audio_text_loss_ratio: 音频-文本对比损失权重 (默认0.3)"
echo "audio_term_loss_ratio: 音频-术语对比损失权重 (默认0.7)"
echo "enable_full_eval: 是否启用完整评估 (true/false)"
echo "enable_hard_neg: 是否启用hard negative mining (true/false)"
echo "full_eval_every_n_epochs: 每N个epoch运行完整评估 (默认1)"
echo "test_samples_path: 测试样本路径"
echo "best_model_path: 预训练best model路径"
echo ""
echo "=== Hard Negative Mining 配置 ==="
echo "当 enable_hard_neg=true 时，自动启用以下配置:"
echo "- hard_neg_source: glossary"
echo "- hard_neg_index_path: data/glossary_emb.ivfpq.faiss"
echo "- hard_neg_term2idx_path: data/glossary_term2idx.json"
echo "- hard_neg_metric: ip"
echo "- hard_neg_nprobe: 16"
echo "- hard_neg_candidates: 100"
echo "- hard_neg_k: 10"
echo ""
echo "=== 注意事项 ==="
echo "1. 确保 data/glossary_emb.ivfpq.faiss 和 data/glossary_term2idx.json 文件存在"
echo "2. best_model_path 应该指向训练好的模型权重文件"
echo "3. 单分片模式适合快速验证，完整模式适合最终训练"
echo "4. hard negative mining 需要预构建的FAISS索引" 