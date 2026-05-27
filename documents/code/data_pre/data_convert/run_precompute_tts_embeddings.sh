#!/bin/bash
#SBATCH --job-name=precompute_tts_emb_v2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=taurus
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_precompute_tts_emb_v2.out
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_precompute_tts_emb_v2.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
VLLM_ENV_PATH="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
conda activate "${VLLM_ENV_PATH}"
export PYTHONPATH="/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"
export HF_HOME="/mnt/gemini/data1/jiaxuanluo/huggingface_cache"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ============================== Configuration ==============================
TERMS_NPY="/mnt/gemini/data/jiaxuanluo/tts_bank_from_term_train_v2/terms.npy"
WAV_DIR="/mnt/gemini/data/jiaxuanluo/tts_bank_from_term_train_v2/wav"
MODEL_PATH="/mnt/gemini/data/jiaxuanluo/q3rag_tts_lora-r32-tr16_bs4k_ttsw0.5_ttm=query key value_temperature=0.03_v2_epoch_5.pt"
GLOSSARY_JSON="/mnt/gemini/data1/jiaxuanluo/glossary_for_zh_rate1.0_k20.json"
OUTPUT_NPZ="/mnt/gemini/data/jiaxuanluo/tts_bank_from_term_train_v2/tts_embeddings_cache.npz"
BATCH_SIZE=512
# ===========================================================================

echo "[INFO] Pre-computing TTS embeddings for TTS bank v2 (tts_audio_path + per-proto)"
echo "[INFO]   TERMS_NPY: ${TERMS_NPY}"
echo "[INFO]   WAV_DIR:   ${WAV_DIR}"
echo "[INFO]   OUTPUT:    ${OUTPUT_NPZ}"

CUDA_VISIBLE_DEVICES=0 python \
    /home/jiaxuanluo/InfiniSST/documents/code/data_pre/data_convert/precompute_tts_embeddings.py \
    --terms-npy "${TERMS_NPY}" \
    --wav-dir "${WAV_DIR}" \
    --model-path "${MODEL_PATH}" \
    --glossary-json "${GLOSSARY_JSON}" \
    --output-npz "${OUTPUT_NPZ}" \
    --target-lang-code zh \
    --batch-size "${BATCH_SIZE}"

echo "[INFO] Done. Output: ${OUTPUT_NPZ}"
