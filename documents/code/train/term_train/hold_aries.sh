#!/bin/bash
# Aries 占坑 job: exclusive 占满 1 node (全部 8 卡 + 全部 CPU + 全部内存),
# 每 2 天自动 resubmit 一次, 形成连续占坑链.
#
# 使用方式:
#   1. sbatch /mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/hold_aries.sh
#   2. squeue -u $USER 确认 job state = R
#   3. ssh aries (如果 cluster 有 pam_slurm_adopt, 这一步要求你在 aries 上有 active job,
#      hold job 正好满足) OR  srun --jobid=<HOLD_JOBID> --overlap --pty bash
#   4. 脚本内 `export CUDA_VISIBLE_DEVICES=0,1,...,7` 之后直接起训练, 不经 sbatch.
#
# 停止占坑:
#   scancel <HOLD_JOBID>
#   并立刻确认它有没有 trap-resubmit 了新 job (squeue 看), 如果有就再 scancel 新的.
#   或者 `touch $HOLD_STOP_SENTINEL` 让 trap 看到就不 resubmit.
#
# ===================== Configuration =====================
#SBATCH --job-name=aries_hold
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=2-00:00:00
#SBATCH --exclusive
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/hold_aries_%j.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/hold_aries_%j.err
# ==========================================================

set -u

HOLD_DURATION_S=$((2*24*3600 - 300))    # 2d - 5min, 留时间给 trap resubmit
HOLD_STOP_SENTINEL="/mnt/taurus/home/jiaxuanluo/.aries_hold.stop"
HOLD_SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/hold_aries.sh"

echo "[HOLD] start job=${SLURM_JOB_ID} node=$(hostname) t0=$(date -u +%FT%TZ)"
echo "[HOLD] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[HOLD] cpus=${SLURM_CPUS_ON_NODE:-?} mem=${SLURM_MEM_PER_NODE:-?}MB"
echo "[HOLD] duration=${HOLD_DURATION_S}s  stop_sentinel=${HOLD_STOP_SENTINEL}"

resubmit_chain() {
    if [ -e "${HOLD_STOP_SENTINEL}" ]; then
        echo "[HOLD] stop sentinel found at ${HOLD_STOP_SENTINEL}, NOT resubmitting."
        return 0
    fi
    if [ ! -f "${HOLD_SCRIPT_PATH}" ]; then
        echo "[HOLD][FATAL] cannot find self at ${HOLD_SCRIPT_PATH}, NOT resubmitting." >&2
        return 1
    fi
    echo "[HOLD] resubmitting: sbatch ${HOLD_SCRIPT_PATH}"
    sbatch "${HOLD_SCRIPT_PATH}" || echo "[HOLD][WARN] resubmit failed"
}

trap 'resubmit_chain' EXIT

# idle sleep, 心跳打印一下不然 log 空空如也排查困难
HEARTBEAT_EVERY_S=3600
elapsed=0
while [ "${elapsed}" -lt "${HOLD_DURATION_S}" ]; do
    if [ -e "${HOLD_STOP_SENTINEL}" ]; then
        echo "[HOLD] stop sentinel detected, exiting cleanly (no resubmit)."
        exit 0
    fi
    sleep "${HEARTBEAT_EVERY_S}"
    elapsed=$((elapsed + HEARTBEAT_EVERY_S))
    echo "[HOLD] heartbeat ${elapsed}/${HOLD_DURATION_S}s  $(date -u +%FT%TZ)"
done

echo "[HOLD] reached time budget, EXIT (trap will resubmit next hold)."
exit 0
