#!/bin/bash
#SBATCH --job-name=build_gsv2_f031
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_build_gsv2_f031.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_build_gsv2_f031.err
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST

set -euo pipefail

export PYTHONUNBUFFERED=1

bash documents/code/data_pre/wiki_synth/3variant/run_build_train_gsv2_full0_31.sh
