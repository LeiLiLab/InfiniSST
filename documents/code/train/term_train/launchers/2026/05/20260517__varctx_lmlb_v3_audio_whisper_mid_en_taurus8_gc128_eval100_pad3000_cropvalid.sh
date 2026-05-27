#!/bin/bash
#SBATCH --job-name=q3_vctx_wmid_crop
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --gres=gpu:8
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_vctx_wmid_crop_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_vctx_wmid_crop_%x.err

set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"
PARENT_LAUNCHER="${REPO_ROOT}/documents/code/train/term_train/launchers/2026/05/20260517__varctx_lmlb_v3_audio_whisper_mid_en_taurus8_gc128_eval100_pad3000.sh"

export GRAD_CACHE_CHUNK_SIZE="${GRAD_CACHE_CHUNK_SIZE:-128}"
export VARIANT_TAG="${VARIANT_TAG:-vctx576_aud_wmid_t8_d100_tau1_g128crop}"
export VERSION="${VERSION:-3var_gsv2full_gsdedup_varctx576_aud_wmid_bs8k_gc${GRAD_CACHE_CHUNK_SIZE}_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep${EPOCHS:-6}_v3_dev100Tau1_eval100_pad3000_cropvalid_taurus8}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-variantE_varctx576_aud_wmid_taurus8_bs8k_gc${GRAD_CACHE_CHUNK_SIZE}_dev100Tau1_eval100_cropvalid}"
export NOTES_FILE="${NOTES_FILE:-${REPO_ROOT}/documents/code/train/term_train/notes/2026/05/20260517__varctx_lmlb_v3_audio_whisper_mid_en_taurus8_gc128_eval100_pad3000_cropvalid.md}"
export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:vctx576_aud_wmid_t8_d100_tau1_g128crop compute:taurus-8gpu}"
export MASTER_PORT="${MASTER_PORT:-30453}"
export RUN_VERDICT="${RUN_VERDICT:-Whisper-medium.en speech-encoder ablation with BGE-M3 text encoder; pad/truncate Whisper mel features to 3000 frames and crop MaxSim pooling to valid hidden length; gc128 Taurus 8GPU eval100.}"

if [ ! -f "${PARENT_LAUNCHER}" ]; then
  echo "[ERROR] required parent launcher missing: ${PARENT_LAUNCHER}" >&2
  exit 2
fi
if [ ! -f "${NOTES_FILE}" ]; then
  echo "[ERROR] required notes file missing: ${NOTES_FILE}" >&2
  exit 2
fi

exec bash "${PARENT_LAUNCHER}"
