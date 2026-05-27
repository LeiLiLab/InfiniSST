#!/bin/bash
#SBATCH --job-name=extract_wavs
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_extract_wavs.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_extract_wavs.err

set -euo pipefail

export PYTHONPATH="/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

python /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/extract_wavs_from_shards.py \
  --train_shards_dir /mnt/data2/jiaxuanluo/gigaspeech_webdataset_v2/train \
  --dev_shards_dir /mnt/data2/jiaxuanluo/gigaspeech_webdataset_v2/dev \
  --output_root /mnt/data2/jiaxuanluo \
  --dest_subdir local_wavs_from_shards \
  --out_train_jsonl /mnt/data2/jiaxuanluo/local_train_dataset.jsonl \
  --out_dev_jsonl /mnt/data2/jiaxuanluo/local_dev_dataset.jsonl

echo "Done extracting wavs and writing local jsonl."


