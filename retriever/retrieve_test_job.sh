#!/usr/bin/env bash

#SBATCH --output=logs/retrieve_test_%A_%a.out
#SBATCH --error=logs/retrieve_test_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gpus=1
##SBATCH --constraint=xeon-4116 (some node property to request)
#SBATCH --partition=taurus
##SBATCH --time=1-2:34:56 (1 day 2 hour 34 min 56 sec)
##SBATCH --dependency=afterok:job_id
#SBATCH --array=0-1
#SBATCH --account=jiaxuanluo
#SBATCH --mail-type=all
#SBATCH --mail-user=luojiaxuan1215@gmail.com

# The rest are your jobs

## Use environment from taurus
conda config --append envs_dirs /mnt/taurus/home/jiaxuanluo/miniconda3/envs/
source /mnt/taurus/home/jiaxuanluo/miniconda3/bin/activate infinisst

## Run your job

INPUT_FILE="/home/jiaxuanluo/InfiniSST/retriever/final_split_terms/"

MODES=("flexible" "safe")
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

echo "Running mode: $MODE"


srun python3 retriever.py \
  --input "$INPUT_FILE" \
  --mode "$MODE" \
  --max_gpu 1 \
  --max_limit 300 \
  --max_terms 1000000 \
  --filter_missing_gt
