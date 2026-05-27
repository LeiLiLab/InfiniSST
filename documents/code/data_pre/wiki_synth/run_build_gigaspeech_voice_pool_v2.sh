#!/bin/bash
#SBATCH --job-name=build_gs_voice_pool_v2
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --time=0-06:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_build_gs_voice_pool_v2.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_build_gs_voice_pool_v2.err

# v2 GigaSpeech voice-reference pool (2026-04-23):
#   - --one-per-opus dedup so 10k prompts = 10k distinct long-form recordings
#     (proxy for distinct speakers). v1 pool had many repeat-opus segs which
#     collapsed true diversity.
#   - Output relocated to /mnt/gemini/home (taurus/data is 97% full).
# Opus census: audiobook=1092, podcast=14602, youtube=18823. Auto-split:
#     audiobook capped at supply (1092); podcast+youtube fill the rest.

set -euo pipefail
export PATH="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin:${PATH}"
export PYTHONNOUSERSITE=1

SCRIPT=/home/jiaxuanluo/InfiniSST/documents/code/data_pre/wiki_synth/build_gigaspeech_voice_pool.py
OUT=/mnt/gemini/home/jiaxuanluo/gigaspeech_speaker_prompts
mkdir -p "${OUT}"

echo "[BUILD v2] starting at $(date)"
df -h /mnt/gemini/home | sed 's/^/  /'

python3 "${SCRIPT}" \
    --n_voices 10000 \
    --one_per_opus \
    --output_dir "${OUT}" \
    --seed 42

echo "[BUILD v2] done at $(date)"
NUM=$(find "${OUT}" -maxdepth 1 -name '*.wav' | wc -l)
echo "[BUILD v2] wav count = ${NUM}"
du -sh "${OUT}"
