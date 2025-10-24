#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=392GB
#SBATCH --gres=gpu:4
#SBATCH --partition=taurus
##SBATCH --array=1-5
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jaxanluo@gmail.com
#SBATCH -e train_infinisst_%j.err
#SBATCH -o train_infinisst_%j.out

# ========================================
# 使用说明：
# 1. 正常提交（自动断点续传）: sbatch stage1_gigaspeech_zh_norm0_qwen_rope_rag.sh
# 2. 从头开始训练: CLEAN_START=1 sbatch stage1_gigaspeech_zh_norm0_qwen_rope_rag.sh
# 3. 查看checkpoint信息: cat /path/to/save_dir/checkpoint_info.txt
# ========================================



set -euo pipefail

# Activate Conda environment
if [ -f "/home/jiaxuanluo/miniconda3/etc/profile.d/conda.sh" ]; then
    source /home/jiaxuanluo/miniconda3/etc/profile.d/conda.sh
    conda activate infinisst
else
    source /home/jaxanluo/anaconda3/bin/activate infinisst
fi

# ------------------------------
# Model and encoder paths (aligned with serve/inference_engine.py)
# ------------------------------
llm_path="/mnt/aries/data6/jiaxuanluo/Qwen2.5-7B-Instruct"
w2v2_path="/mnt/aries/data6/xixu/demo/wav2_vec_vox_960h_pl.pt"

w2v2_type=w2v2
ctc_finetuned=True

# ------------------------------
# Data configuration
# ------------------------------
preferred_root="/mnt/data/siqiouyang/datasets/gigaspeech/"
ROOT=${preferred_root}

lang_code=zh
lang=Chinese
data_path=${ROOT}

# ------------------------------
# Checkpoint/output configuration
# ------------------------------
model_export_dir="/mnt/aries/data6/jiaxuanluo/demo/en-${lang_code}"
run_root="${model_export_dir}/runs"

source_lang="English"
target_lang=${lang} # e.g. German
name="stage1_M=12_ls-cv-vp_norm0_qwen_rope_vv2_gigaspeech_v2"
save_path=${run_root}/${name}

# === 断点续传逻辑 ===
# 检查是否需要清理旧checkpoint从头开始
if [ "${CLEAN_START:-0}" = "1" ]; then
    if [ -d "${save_path}" ]; then
        echo "[INFO] =========================================="
        echo "[INFO] CLEAN_START=1 检测到，删除旧checkpoint"
        echo "[INFO] 将从头开始训练"
        echo "[INFO] =========================================="
        rm -rf "${save_path}"
    fi
fi

# 检查是否存在checkpoint
RESUME_TRAINING=false
CHECKPOINT_INFO=""

if [ -d "${save_path}" ]; then
    # 检查 DeepSpeed checkpoint 格式：last.ckpt/checkpoint/ 目录
    if [ -d "${save_path}/last.ckpt/checkpoint" ] && [ "$(ls -A ${save_path}/last.ckpt/checkpoint 2>/dev/null)" ]; then
        RESUME_TRAINING=true
        CHECKPOINT_PATH="${save_path}/last.ckpt"
        echo "[INFO] =========================================="
        echo "[INFO] 发现 DeepSpeed checkpoint，将从断点继续训练"
        echo "[INFO] Checkpoint: ${CHECKPOINT_PATH}"
        
        # 尝试从DeepSpeed checkpoint提取信息
        MODEL_STATE="${save_path}/last.ckpt/checkpoint/mp_rank_00_model_states.pt"
        if [ -f "${MODEL_STATE}" ]; then
            CHECKPOINT_INFO=$(python3 -c "
import torch
try:
    ckpt = torch.load('${MODEL_STATE}', map_location='cpu', weights_only=False)
    epoch = ckpt.get('epoch', 'N/A')
    step = ckpt.get('global_step', 'N/A')
    print(f'Epoch: {epoch}, Global Step: {step}')
except Exception as e:
    print(f'DeepSpeed checkpoint (无法提取详细信息)')
" 2>/dev/null)
            
            if [ -n "${CHECKPOINT_INFO}" ]; then
                echo "[INFO] ${CHECKPOINT_INFO}"
            fi
        fi
        echo "[INFO] =========================================="
    # 检查 Lightning 标准格式：last.ckpt/checkpoint 文件
    elif [ -f "${save_path}/last.ckpt/checkpoint" ]; then
        RESUME_TRAINING=true
        CHECKPOINT_PATH="${save_path}/last.ckpt/checkpoint"
        echo "[INFO] =========================================="
        echo "[INFO] 发现 Lightning checkpoint"
        echo "[INFO] Checkpoint: ${CHECKPOINT_PATH}"
        
        # 提取checkpoint信息
        CHECKPOINT_INFO=$(python3 -c "
import torch
try:
    ckpt = torch.load('${CHECKPOINT_PATH}', map_location='cpu', weights_only=False)
    epoch = ckpt.get('epoch', 'N/A')
    step = ckpt.get('global_step', 'N/A')
    print(f'Epoch: {epoch}, Global Step: {step}')
except Exception as e:
    print(f'无法读取checkpoint: {e}')
" 2>/dev/null)
        
        if [ -n "${CHECKPOINT_INFO}" ]; then
            echo "[INFO] ${CHECKPOINT_INFO}"
        fi
        echo "[INFO] =========================================="
    # 检查 Lightning 1.x 格式：last.ckpt 文件
    elif [ -f "${save_path}/last.ckpt" ]; then
        RESUME_TRAINING=true
        CHECKPOINT_PATH="${save_path}/last.ckpt"
        echo "[INFO] =========================================="
        echo "[INFO] 发现 Lightning 1.x checkpoint"
        echo "[INFO] Checkpoint: ${CHECKPOINT_PATH}"
        echo "[INFO] =========================================="
    else
        echo "[INFO] =========================================="
        echo "[INFO] 未发现checkpoint，将从头开始训练"
        echo "[INFO] =========================================="
    fi
else
    echo "[INFO] =========================================="
    echo "[INFO] 目录不存在，创建新目录并从头训练"
    echo "[INFO] =========================================="
fi

mkdir -p ${save_path}
mkdir -p ${model_export_dir}

export PYTHONPATH=${PYTHONPATH:-}:$PWD
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="infinisst"
export WANDB_ENTITY="luojiaxuan1215-johns-hopkins-university"

# Specify which GPUs to use (single task with 2 GPUs)
export CUDA_VISIBLE_DEVICES=2,3  # 改回2 GPU，3 GPU DDP可能不稳定

# disable P2P and InfiniBand for L40S 8-GPU nodes
# if your node supports P2P and InfiniBand, you need to remove these two lines
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1  # 更好的错误处理
export NCCL_TIMEOUT=1800  # 增加超时到30分钟

if [ -n "${SLURM_GPUS_PER_NODE:-}" ]; then
    NUM_GPUS=${SLURM_GPUS_PER_NODE}
elif [ -n "${SLURM_GPUS:-}" ]; then
    if [[ ${SLURM_GPUS} =~ ^[0-9]+$ ]]; then
        NUM_GPUS=${SLURM_GPUS}
    else
        NUM_GPUS=$(echo "${SLURM_GPUS}" | awk -F',' '{print NF}')
    fi
else
    NUM_GPUS=2  # 默认2 GPU
fi

python train/main.py \
    \
    --model_type w2v2_qwen25 \
    \
    --w2v2_path ${w2v2_path} \
    --w2v2_type ${w2v2_type} \
    --ctc_finetuned ${ctc_finetuned} \
    --length_shrink_cfg "[(1024,2,2)] * 2" \
    --block_size 48 \
    --max_cache_size 576 \
    \
    --llm_path ${llm_path} \
    --llm_freeze True \
    --llm_emb_freeze True \
    --llm_head_freeze True \
    --use_flash_attn True \
    \
    --data_path ${data_path} \
    --data_split_train 'train_xl_case_ft-qwen2.5-32b-instruct_marked_mfa_punc_asr' \
    --data_split_eval 'dev_case_ft-qwen2.5-32b-instruct_marked_mfa_punc' \
    --source_lang "${source_lang}" \
    --target_lang "${target_lang}" \
    --trajectory 9 \
    --trajectory_max_multiplier 12 \
    --trajectory_prob_aug 0.0 \
    --audio_normalize False \
    \
    --seed 998244353 \
    --stage 1 \
    --train_bsz 1800 \
    --eval_bsz 1800 \
    --bsz_sent 2 \
    --learning_rate 2e-4 \
    --warmup_steps 1000 \
    --run_name $name \
    \
    --n_device ${NUM_GPUS} \
    --max_epochs 1 \
    --grad_acc_steps 4 \
    --clip_norm 1.0 \
    --save_dir ${save_path} \
    --save_step 1000 \
    --log_step 100 \
    --eval_step 1000

# Using DDPStrategy (no DeepSpeed), checkpoint is saved in Lightning format directly
echo "[INFO] =========================================="
echo "[INFO] 训练完成，处理checkpoint..."
echo "[INFO] =========================================="

# Extract model weights from Lightning checkpoint
if [ -f "${save_path}/last.ckpt/checkpoint" ]; then
    # 提取并保存checkpoint信息
    python3 -c "
import torch
from datetime import datetime

try:
    ckpt = torch.load('${save_path}/last.ckpt/checkpoint', map_location='cpu', weights_only=False)
    
    # 提取模型权重
    torch.save(ckpt['state_dict'], '${save_path}/last.ckpt/pytorch_model.bin')
    print('[INFO] ✓ 已提取模型权重')
    
    # 保存checkpoint信息到文本文件
    with open('${save_path}/checkpoint_info.txt', 'w') as f:
        f.write('=' * 60 + '\n')
        f.write('InfiniSST Training Checkpoint Info\n')
        f.write('=' * 60 + '\n')
        f.write(f'保存时间: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}\n')
        f.write(f'Epoch: {ckpt.get(\"epoch\", \"N/A\")}\n')
        f.write(f'Global Step: {ckpt.get(\"global_step\", \"N/A\")}\n')
        f.write(f'Checkpoint Path: ${save_path}/last.ckpt/checkpoint\n')
        f.write('=' * 60 + '\n')
    
    print('[INFO] ✓ 已保存checkpoint信息到 ${save_path}/checkpoint_info.txt')
    
    # 打印信息
    print(f'[INFO] Epoch: {ckpt.get(\"epoch\", \"N/A\")}')
    print(f'[INFO] Global Step: {ckpt.get(\"global_step\", \"N/A\")}')
    
except Exception as e:
    print(f'[ERROR] 处理checkpoint失败: {e}')
    import traceback
    traceback.print_exc()
"
    
    # Export to the location expected by the inference service
    if [ -f "${save_path}/last.ckpt/pytorch_model.bin" ]; then
        cp -f ${save_path}/last.ckpt/pytorch_model.bin ${model_export_dir}/pytorch_model.bin
        echo "[INFO] ✓ 已导出模型到 ${model_export_dir}/pytorch_model.bin"
    fi
    
    echo "[INFO] =========================================="
    echo "[INFO] Checkpoint处理完成！"
    echo "[INFO] 查看详细信息: cat ${save_path}/checkpoint_info.txt"
    echo "[INFO] =========================================="
else
    echo "[WARN] =========================================="
    echo "[WARN] 未找到checkpoint: ${save_path}/last.ckpt/checkpoint"
    echo "[WARN] 训练可能未正常完成或保存失败"
    echo "[WARN] =========================================="
fi
