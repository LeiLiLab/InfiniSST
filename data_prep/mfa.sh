#!/usr/bin/env bash

##SBATCH --nodelist=babel-4-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
##SBATCH --gres=gpu:L40S:1
##SBATCH --nodelist=babel-3-17
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
##SBATCH --dependency=afterok:job_id
##SBATCH --array=0-7
##SBATCH --account=siqiouya
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH -e slurm_logs/%A-%a.err
#SBATCH -o slurm_logs/%A-%a.out

source /home/siqiouya/anaconda3/bin/activate mfa

ROOT=/compute/babel-14-5/siqiouya/

cd $ROOT/en-de/data/train/mfa
mfa align --clean --final_clean --overwrite --output_format long_textgrid . english_mfa english_mfa textgrids