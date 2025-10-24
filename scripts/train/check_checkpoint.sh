#!/usr/bin/env bash
# ========================================
# Checkpoint 信息查看工具
# 用法: ./check_checkpoint.sh [checkpoint_dir]
# ========================================

set -euo pipefail

CHECKPOINT_DIR="${1:-/mnt/aries/data6/jiaxuanluo/demo/en-zh/runs/stage1_M=12_ls-cv-vp_norm0_qwen_rope_vv2_gigaspeech_v2}"

echo "=========================================="
echo "检查 Checkpoint: ${CHECKPOINT_DIR}"
echo "=========================================="

if [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "[ERROR] 目录不存在: ${CHECKPOINT_DIR}"
    exit 1
fi

# 检查是否存在 checkpoint_info.txt
if [ -f "${CHECKPOINT_DIR}/checkpoint_info.txt" ]; then
    echo ""
    echo "=== Checkpoint 信息 ==="
    cat "${CHECKPOINT_DIR}/checkpoint_info.txt"
    echo ""
fi

# 检查 Lightning 2.x 格式
if [ -f "${CHECKPOINT_DIR}/last.ckpt/checkpoint" ]; then
    echo "=== Lightning Checkpoint (2.x格式) ==="
    echo "路径: ${CHECKPOINT_DIR}/last.ckpt/checkpoint"
    echo "大小: $(du -h ${CHECKPOINT_DIR}/last.ckpt/checkpoint | cut -f1)"
    echo ""
    
    # 提取详细信息
    python3 -c "
import torch
import sys

try:
    ckpt = torch.load('${CHECKPOINT_DIR}/last.ckpt/checkpoint', map_location='cpu', weights_only=False)
    
    print('详细信息:')
    print(f'  Epoch: {ckpt.get(\"epoch\", \"N/A\")}')
    print(f'  Global Step: {ckpt.get(\"global_step\", \"N/A\")}')
    print(f'  Checkpoint Keys: {list(ckpt.keys())}')
    
    # 如果有 optimizer state，显示
    if 'optimizer_states' in ckpt:
        print(f'  Optimizer States: {len(ckpt[\"optimizer_states\"])} optimizer(s)')
    
    # 如果有 lr_schedulers，显示
    if 'lr_schedulers' in ckpt:
        print(f'  LR Schedulers: {len(ckpt[\"lr_schedulers\"])} scheduler(s)')
        
except Exception as e:
    print(f'[ERROR] 无法读取checkpoint: {e}', file=sys.stderr)
    sys.exit(1)
"
elif [ -f "${CHECKPOINT_DIR}/last.ckpt" ]; then
    echo "=== Lightning Checkpoint (1.x格式) ==="
    echo "路径: ${CHECKPOINT_DIR}/last.ckpt"
    echo "大小: $(du -h ${CHECKPOINT_DIR}/last.ckpt | cut -f1)"
    echo ""
else
    echo "[WARN] 未找到 Lightning checkpoint"
fi

# 列出所有 epoch/step checkpoint
echo "=== 所有 Checkpoint 文件 ==="
find "${CHECKPOINT_DIR}" -type f -name "checkpoint" -o -name "*.ckpt" | while read -r ckpt; do
    rel_path=$(realpath --relative-to="${CHECKPOINT_DIR}" "$ckpt")
    size=$(du -h "$ckpt" | cut -f1)
    echo "  - ${rel_path} (${size})"
done

# 显示目录总大小
echo ""
echo "=== 目录统计 ==="
echo "总大小: $(du -sh ${CHECKPOINT_DIR} | cut -f1)"
echo "文件数: $(find ${CHECKPOINT_DIR} -type f | wc -l)"

echo "=========================================="

