#!/bin/bash
#SBATCH --job-name=hard_negative_training
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/hn_train_%j.out
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/hn_train_%j.err
#SBATCH --partition=aries
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
# ===================================================================
# Hard Negative Mining + 训练完整流程
# 专门提升 Recall@5/10 成功率
# 
# 挖矿模式配置 (HN_MODE):
#   - "overwrite": 覆盖已存在的HN文件（会先备份）
#   - "skip": 跳过已存在的HN文件
#   - "update": 在已有基础上增量更新（推荐，合并新旧HN）
# 
# 多GPU并行挖矿:
#   - 使用 MINE_NUM_GPUS 控制挖矿时的GPU数量
#   - 使用 MINE_BATCH_SIZE 控制每个GPU的batch size
#   - 显著加速HN挖掘过程
# ===================================================================

set -e  # 遇到错误立即退出

map_path() { [[ "$1" = /* ]] && echo "/mnt/taurus$1" || echo "$1"; }

# 用环境里的绝对路径，避免依赖 conda activate
PY="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/infinisst/bin/python"
export PATH="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/infinisst/bin:$PATH"

# ===================== 配置参数 =====================

# 激活环境
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate infinisst
# —— Conda 环境激活（容错）——
# 尝试 3 种来源：/mnt/taurus 上的 conda.sh、本机 $HOME、系统 anaconda 模块
if [ -f /mnt/taurus/home/jiaxuanluo/miniconda3/etc/profile.d/conda.sh ]; then
  source /mnt/taurus/home/jiaxuanluo/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
  source ~/miniconda3/etc/profile.d/conda.sh
elif command -v module >/dev/null 2>&1; then
  module load anaconda || module load anaconda3 || true
  # 有些集群需要这句把 conda 注入 shell
  eval "$(conda shell.bash hook)" 2>/dev/null || true
fi

# 路径配置
WORK_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal"
DATA_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data"
MODEL_DIR="/mnt/gemini/data2/jiaxuanluo/models"
MMAP_DIR="/mnt/gemini/data1/jiaxuanluo/mmap_shards"

# 数据文件
TRAIN_SAMPLES="$DATA_DIR/balanced_train_set.json"
TEST_SAMPLES="$DATA_DIR/balanced_test_set.json"
GLOSSARY="$DATA_DIR/terms/glossary_cleaned.json"

# 模型文件
BEST_MODEL="$MODEL_DIR/qwen2_audio_term_level_modal_v2_best.pt"
NEW_MODEL="$MODEL_DIR/qwen2_audio_term_level_hn_v1.pt"
FAISS_INDEX="/mnt/gemini/data2/jiaxuanluo/indices/qwen2_audio_term_index.pkl"

# Hard Negative 文件
HN_TRAIN="$MODEL_DIR/hard_negs_train.jsonl"
HN_TEST="$MODEL_DIR/hard_negs_test.jsonl"

# 模型配置
MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct"
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.1

# 挖矿配置
TOPK=200              # 检索的候选数
MINE_BATCH_SIZE=128    # 挖矿时每个GPU的batch size
MINE_NUM_GPUS=6       # 挖矿时使用的GPU数量

# 训练配置
NUM_GPUS=6            # 训练时使用的GPU数量
BATCH_SIZE=256        # 总batch size
GRAD_ACCUM=8          # 梯度累积步数
EPOCHS=15             # 训练轮数
LR=5e-5               # 学习率（降低以微调）

# Hard Negative 配置
MAX_HN=15             # 每个样本使用的hard negative数量
RAND_NEG=5            # 每个样本使用的随机负例数量
HN_LOSS_RATIO=0.5     # Hard negative loss权重
AUDIO_TEXT_RATIO=0.2  # 音频-文本对比损失权重
AUDIO_TERM_RATIO=0.3  # 音频-术语对比损失权重

# 挖矿行为配置
HN_MODE="update"      # 挖矿模式: "overwrite"=覆盖, "skip"=跳过, "update"=更新已有文件

cd "$WORK_DIR"

# ===================== 步骤 0: 检查前置条件 =====================

echo ""
echo "=========================================="
echo "检查前置条件"
echo "=========================================="

# 检查最佳模型是否存在
if [ ! -f "$BEST_MODEL" ]; then
    echo "❌ 错误: 找不到最佳模型: $BEST_MODEL"
    echo "请先训练一个基础模型"
    exit 1
fi
echo "✅ 找到最佳模型: $BEST_MODEL"

# 检查FAISS索引是否存在
if [ ! -f "$FAISS_INDEX" ]; then
    echo "⚠️  警告: 找不到FAISS索引: $FAISS_INDEX"
    echo "将在挖矿前先构建索引..."
    
    # 构建索引
    $PY build_index_multi_gpu.py \
        --model_path "$BEST_MODEL" \
        --glossary_path "$GLOSSARY" \
        --output_path "$FAISS_INDEX" \
        --model_name "$MODEL_NAME" \
        --lora_r "$LORA_R" \
        --lora_alpha "$LORA_ALPHA" \
        --num_gpus 6 \
        --batch_size 4
else
    echo "✅ 找到FAISS索引: $FAISS_INDEX"
fi

# ===================== 步骤 1: 挖掘训练集 Hard Negatives =====================

echo ""
echo "=========================================="
echo "步骤 1/4: 挖掘训练集 Hard Negatives"
echo "=========================================="

if [ -f "$HN_TRAIN" ]; then
    echo "⚠️  发现已存在的训练集HN文件: $HN_TRAIN"
    echo "当前挖矿模式: $HN_MODE"
    
    if [ "$HN_MODE" == "skip" ]; then
        echo "⏭️  跳过训练集挖矿（skip模式）"
    elif [ "$HN_MODE" == "update" ]; then
        echo "🔄 更新模式: 在现有基础上增量更新"
        # 备份原文件
        cp "$HN_TRAIN" "${HN_TRAIN}.backup_$(date +%Y%m%d_%H%M%S)"
        echo "✅ 已备份原文件"
        
        $PY mine_hard_negatives_multi_gpu.py \
            --samples_path "$TRAIN_SAMPLES" \
            --mmap_dir "$MMAP_DIR" \
            --faiss_index_pkl "$FAISS_INDEX" \
            --model_path "$BEST_MODEL" \
            --model_name "$MODEL_NAME" \
            --lora_r "$LORA_R" \
            --lora_alpha "$LORA_ALPHA" \
            --out_path "$HN_TRAIN" \
            --topk "$TOPK" \
            --batch_size "$MINE_BATCH_SIZE" \
            --num_gpus "$MINE_NUM_GPUS" \
            --update_existing
        echo "✅ 训练集HN更新完成"
    else
        echo "🔄 覆盖模式: 重新生成所有hard negatives"
        # 备份原文件
        cp "$HN_TRAIN" "${HN_TRAIN}.backup_$(date +%Y%m%d_%H%M%S)"
        echo "✅ 已备份原文件"
        
        $PY mine_hard_negatives_multi_gpu.py \
            --samples_path "$TRAIN_SAMPLES" \
            --mmap_dir "$MMAP_DIR" \
            --faiss_index_pkl "$FAISS_INDEX" \
            --model_path "$BEST_MODEL" \
            --model_name "$MODEL_NAME" \
            --lora_r "$LORA_R" \
            --lora_alpha "$LORA_ALPHA" \
            --out_path "$HN_TRAIN" \
            --topk "$TOPK" \
            --batch_size "$MINE_BATCH_SIZE" \
            --num_gpus "$MINE_NUM_GPUS"
        echo "✅ 训练集HN挖矿完成"
    fi
else
    echo "📝 首次生成训练集HN文件"
    $PY mine_hard_negatives_multi_gpu.py \
        --samples_path "$TRAIN_SAMPLES" \
        --mmap_dir "$MMAP_DIR" \
        --faiss_index_pkl "$FAISS_INDEX" \
        --model_path "$BEST_MODEL" \
        --model_name "$MODEL_NAME" \
        --lora_r "$LORA_R" \
        --lora_alpha "$LORA_ALPHA" \
        --out_path "$HN_TRAIN" \
        --topk "$TOPK" \
        --batch_size "$MINE_BATCH_SIZE" \
        --num_gpus "$MINE_NUM_GPUS"
    echo "✅ 训练集HN挖矿完成"
fi

# ===================== 步骤 2: 挖掘测试集 Hard Negatives（可选） =====================

echo ""
echo "=========================================="
echo "步骤 2/4: 挖掘测试集 Hard Negatives (可选)"
echo "=========================================="

# 检查是否在SLURM作业中运行
if [ -n "$SLURM_JOB_ID" ]; then
    # SLURM作业中，根据HN_MODE自动决定
    MINE_TEST_HN=true
    echo "检测到SLURM作业环境，自动处理测试集HN"
else
    # 交互式环境，询问用户
    read -p "是否挖掘测试集HN以供分析? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        MINE_TEST_HN=true
    else
        MINE_TEST_HN=false
    fi
fi

if [ "$MINE_TEST_HN" = true ]; then
    if [ -f "$HN_TEST" ] && [ "$HN_MODE" == "skip" ]; then
        echo "⏭️  跳过测试集挖矿（skip模式）"
    elif [ -f "$HN_TEST" ] && [ "$HN_MODE" == "update" ]; then
        echo "🔄 更新测试集HN"
        cp "$HN_TEST" "${HN_TEST}.backup_$(date +%Y%m%d_%H%M%S)"
        $PY mine_hard_negatives_multi_gpu.py \
            --samples_path "$TEST_SAMPLES" \
            --mmap_dir "$MMAP_DIR" \
            --faiss_index_pkl "$FAISS_INDEX" \
            --model_path "$BEST_MODEL" \
            --model_name "$MODEL_NAME" \
            --lora_r "$LORA_R" \
            --lora_alpha "$LORA_ALPHA" \
            --out_path "$HN_TEST" \
            --topk "$TOPK" \
            --batch_size "$MINE_BATCH_SIZE" \
            --num_gpus "$MINE_NUM_GPUS" \
            --update_existing
        echo "✅ 测试集HN更新完成"
    else
        $PY mine_hard_negatives_multi_gpu.py \
            --samples_path "$TEST_SAMPLES" \
            --mmap_dir "$MMAP_DIR" \
            --faiss_index_pkl "$FAISS_INDEX" \
            --model_path "$BEST_MODEL" \
            --model_name "$MODEL_NAME" \
            --lora_r "$LORA_R" \
            --lora_alpha "$LORA_ALPHA" \
            --out_path "$HN_TEST" \
            --topk "$TOPK" \
            --batch_size "$MINE_BATCH_SIZE" \
            --num_gpus "$MINE_NUM_GPUS"
        echo "✅ 测试集HN挖矿完成"
    fi
else
    echo "⏭️  跳过测试集HN挖矿"
fi

# ===================== 步骤 3: 使用 Hard Negatives 进行训练 =====================

echo ""
echo "=========================================="
echo "步骤 3/4: 使用 Hard Negatives 训练"
echo "=========================================="
echo "配置摘要:"
echo "  - 训练GPUs: $NUM_GPUS"
echo "  - Batch Size: $BATCH_SIZE (per GPU: $((BATCH_SIZE / NUM_GPUS)))"
echo "  - Gradient Accumulation: $GRAD_ACCUM"
echo "  - Effective Batch: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  - Learning Rate: $LR"
echo "  - Epochs: $EPOCHS"
echo "  - Max HN per sample: $MAX_HN"
echo "  - Random Neg per sample: $RAND_NEG"
echo "  - Loss Weights: Text=$AUDIO_TEXT_RATIO, Term=$AUDIO_TERM_RATIO, HN=$HN_LOSS_RATIO"
echo ""
echo "注: HN挖矿使用了 $MINE_NUM_GPUS 个GPU，每个GPU batch size为 $MINE_BATCH_SIZE"
echo ""

# 设置CUDA环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 启动DDP训练
$PY -m torch.distributed.run \
    --nproc_per_node="$NUM_GPUS" \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    train_ddp_simplified.py \
    --train_samples_path "$TRAIN_SAMPLES" \
    --test_samples_path "$TEST_SAMPLES" \
    --glossary_path "$GLOSSARY" \
    --mmap_shard_dir "$MMAP_DIR" \
    --save_path "$NEW_MODEL" \
    --best_model_path "$BEST_MODEL" \
    --model_name "$MODEL_NAME" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --lr "$LR" \
    --patience 3 \
    --audio_text_loss_ratio "$AUDIO_TEXT_RATIO" \
    --audio_term_loss_ratio "$AUDIO_TERM_RATIO" \
    --hard_neg_jsonl "$HN_TRAIN" \
    --max_hn_per_sample "$MAX_HN" \
    --rand_neg_per_sample "$RAND_NEG" \
    --hard_neg_loss_ratio "$HN_LOSS_RATIO"

echo "✅ Hard Negative 训练完成"

# ===================== 步骤 4: 评估新模型 =====================

echo ""
echo "=========================================="
echo "步骤 4/4: 评估新模型"
echo "=========================================="

# 生成新的FAISS索引
NEW_FAISS_INDEX="${FAISS_INDEX%.pkl}_hn_v1.pkl"
echo "为新模型构建FAISS索引..."
$PY build_index_multi_gpu.py \
    --model_path "${NEW_MODEL%.pt}_best.pt" \
    --glossary_path "$GLOSSARY" \
    --output_path "$NEW_FAISS_INDEX" \
    --model_name "$MODEL_NAME" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --num_gpus 6 \
    --batch_size 4

echo "运行评估..."
$PY eval_local.py \
    --model_path "${NEW_MODEL%.pt}_best.pt" \
    --test_samples_path "$TEST_SAMPLES" \
    --glossary_path "$GLOSSARY" \
    --mmap_shard_dir "$MMAP_DIR" \
    --model_name "$MODEL_NAME" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --max_eval 1000 \
    --num_failed_cases 10 \
    --device "cuda:0" \
    --prebuilt_index "$NEW_FAISS_INDEX"

echo ""
echo "=========================================="
echo "✅ 完整流程执行完毕!"
echo "=========================================="
echo "生成的文件:"
echo "  - 训练集HN: $HN_TRAIN"
if [ -f "$HN_TEST" ]; then
    echo "  - 测试集HN: $HN_TEST"
fi
echo "  - 新模型: ${NEW_MODEL%.pt}_best.pt"
echo "  - 新索引: $NEW_FAISS_INDEX"
echo ""
echo "下一步建议:"
echo "  1. 对比评估结果，重点关注 Recall@5 和 Recall@10"
echo "  2. 如果效果好，可以进行第二轮挖矿+训练（在线刷新）"
echo "  3. 调整超参数（MAX_HN, HN_LOSS_RATIO等）继续优化"
echo ""

