#!/usr/bin/env bash

##SBATCH --nodelist=babel-4-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:L40S:2
#SBATCH --partition=preempt
#SBATCH --time=2-00:00:00
##SBATCH --dependency=afterok:job_id
#SBATCH --array=1-4
##SBATCH --account=siqiouya
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH -e slurm_logs/%A-%a.err
#SBATCH -o slurm_logs/%A-%a.out

src_segment_size=$(($SLURM_ARRAY_TASK_ID * 960))
max_cache_chunks=$((30 / $SLURM_ARRAY_TASK_ID))
keep_cache_chunks=$((30 / $SLURM_ARRAY_TASK_ID - 1))

lang_code=zh
lang=Chinese

tokenizer=zh
unit=char

cd /home/siqiouya/code/infinisst-omni
apptainer exec \
    --nv \
    --env "VLLM_USE_V1=0" \
    --env "NCCL_P2P_DISABLE=1" \
    --env "NCCL_IB_DISABLE=1" \
    --env "PYTHONPATH=/home/siqiouya/code/infinisst-omni" \
    --env "VLLM_WORKER_MULTIPROC_METHOD=spawn" \
    --env "PYTHONUNBUFFERED=1" \
    docker://qwenllm/qwen3-omni:3-cu124 \
    /home/siqiouya/.local/bin/simuleval \
    --agent agents/infinisst_omni.py \
    --agent-class agents.InfiniSSTOmni \
    --source-segment-size ${src_segment_size} \
    --output /data/user_data/siqiouya/ckpts/test_swift/Qwen3-Omni-30B-A3B-Instruct-lora/v1-20251104-033331-hf/evaluation/RealSI/en2zh/seg${src_segment_size} \
    --max-new-tokens 128 \
    --max-cache-chunks ${max_cache_chunks} \
    --keep-cache-chunks ${keep_cache_chunks} \
    --source-lang English \
    --target-lang ${lang} \
    --min-start-sec 0 \
    --source /data/group_data/li_lab/siqiouya/datasets/RealSI/data/en2zh/sources.txt \
    --target /data/group_data/li_lab/siqiouya/datasets/RealSI/data/en2zh/targets.txt \
    --use-vllm 1 \
    --temperature 0.6 \
    --top-p 0.95 \
    --top-k 1 \
    --model-name /data/user_data/siqiouya/ckpts/test_swift/Qwen3-Omni-30B-A3B-Instruct-lora/v1-20251104-033331-hf \
    --quality-metrics BLEU \
    --eval-latency-unit ${unit} \
    --sacrebleu-tokenizer ${tokenizer}