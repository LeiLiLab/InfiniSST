#!/usr/bin/env bash

##SBATCH --nodelist=babel-4-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:L40S:1
#SBATCH --partition=preempt
#SBATCH --time=2-00:00:00
##SBATCH --dependency=afterok:job_id
#SBATCH --array=0-23
##SBATCH --account=siqiouya
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH -e slurm_logs/%A-%a.err
#SBATCH -o slurm_logs/%A-%a.out

source /home/siqiouya/miniconda3/bin/activate cosyvoice_vllm

if [ "$SLURM_ARRAY_TASK_ID" -ge 0 ] && [ "$SLURM_ARRAY_TASK_ID" -le 9 ]; then
    PART_ID="0${SLURM_ARRAY_TASK_ID}"
else
    PART_ID="${SLURM_ARRAY_TASK_ID}"
fi
DATA_PATH="/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests_rag/term_train_dataset_final_part_${PART_ID}.jsonl"
# DATA_PATH="/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests_rag/term_dev_dataset_final.jsonl"

echo "Processing data from $DATA_PATH"

python /home/siqiouya/code/infinisst-omni/preprocess/rag_tts.py \
    --data $DATA_PATH
