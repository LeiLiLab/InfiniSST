#!/usr/bin/env bash

##SBATCH --nodelist=babel-4-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:L40S:1
#SBATCH --partition=general
#SBATCH --time=1-00:00:00
##SBATCH --dependency=afterok:job_id
##SBATCH --array=1-7
##SBATCH --account=siqiouya
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH -e slurm_logs/%j.err
#SBATCH -o slurm_logs/%j.out

root=/compute/babel-14-5/siqiouya/en-zh/
split=dev_st_zh_traj_30_gpt-4o-mini-2024-07-18
src_lang=en
tgt_lang=zh

source /home/siqiouya/anaconda3/bin/activate seamless
cd /home/siqiouya/work/seamless_communication
python forced_alignment.py --root $root --split ${split} --src-lang $src_lang --tgt-lang $tgt_lang --max-duration 28.8

source /home/siqiouya/anaconda3/bin/activate simalign
cd /home/siqiouya/work/simalign
python build_trajectory_seamless.py --data-root $root --lang $tgt_lang --split ${split}_fa --output-split ${split}_fa_traj