#!/bin/bash
#SBATCH --job-name=build_gsv2_p020
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_build_gsv2_p020.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_build_gsv2_p020.err
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST

set -euo pipefail

export PYTHONUNBUFFERED=1

bash documents/code/data_pre/wiki_synth/3variant/run_build_train_gsv2_partial0_20.sh
