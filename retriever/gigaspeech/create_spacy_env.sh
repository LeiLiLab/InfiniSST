#!/bin/bash
# 这个脚本将创建一个专门用于 spaCy GPU 加速的环境，避开 vLLM 的依赖冲突
ENV_NAME="spacy_gpu_env"

echo "Creating new conda environment: $ENV_NAME..."
source ~/miniconda3/etc/profile.d/conda.sh

# 1. 创建环境
conda create -y -n $ENV_NAME python=3.10

# 2. 激活并安装依赖
conda activate $ENV_NAME

# 安装支持 CUDA 12 的 cupy (请根据您的驱动版本确认，12.x 通用)
pip install cupy-cuda12x

# 安装 spacy 及其 GPU 插件
pip install "spacy[cuda12x]"

# 安装 transformer 插件 (满足 trf 模型需要，这里可以用旧版版本)
pip install "transformers<4.50.0" spacy-transformers

# 安装其他必要库
pip install zhconv tqdm

# 3. 下载模型
python -m spacy download en_core_web_trf

echo "Environment $ENV_NAME created successfully."


















