#!/usr/bin/env bash

##SBATCH --nodelist=babel-4-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:L40S:1
##SBATCH --nodelist=babel-3-17
#SBATCH --partition=preempt
#SBATCH --time=2-00:00:00
##SBATCH --dependency=afterok:job_id
#SBATCH --array=16-23
##SBATCH --account=siqiouya
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH -e slurm_logs/%A-%a.err
#SBATCH -o slurm_logs/%A-%a.out

source /home/siqiouya/anaconda3/bin/activate speechllama

cd /home/siqiouya/work/sllama/data
python asr.py --num_splits 24 --split_id $SLURM_ARRAY_TASK_ID