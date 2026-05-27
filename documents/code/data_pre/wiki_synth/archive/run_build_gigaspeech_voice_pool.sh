#!/bin/bash
#SBATCH --job-name=build_gs_voice_pool
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --time=0-03:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_build_gs_voice_pool.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_build_gs_voice_pool.err

# Build the 10k GigaSpeech voice-reference pool for CosyVoice zero-shot TTS.
# CPU-only; reads opus from taurus, writes wav+json to taurus.

set -euo pipefail
export PATH="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin:${PATH}"
export PYTHONNOUSERSITE=1

SCRIPT=/home/jiaxuanluo/InfiniSST/documents/code/data_pre/wiki_synth/build_gigaspeech_voice_pool.py
OUT=/mnt/taurus/data/jiaxuanluo/gigaspeech_speaker_prompts
mkdir -p "${OUT}"

echo "[BUILD] starting at $(date)"
python3 "${SCRIPT}" \
    --n_voices 10000 \
    --output_dir "${OUT}" \
    --seed 42

echo "[BUILD] done at $(date)"
ls "${OUT}" | wc -l
