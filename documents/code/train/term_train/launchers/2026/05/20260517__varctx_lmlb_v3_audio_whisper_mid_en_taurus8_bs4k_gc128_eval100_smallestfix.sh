#!/bin/bash
#SBATCH --job-name=q3_vctx_wmid_bs4k
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --gres=gpu:8
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_vctx_wmid_bs4k_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_vctx_wmid_bs4k_%x.err

set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"
PARENT_LAUNCHER="${REPO_ROOT}/documents/code/train/term_train/launchers/2026/05/20260517__varctx_lmlb_v3_audio_whisper_mid_en_taurus8_gc128_eval100_pad3000_cropvalid_smallestfix.sh"

export PER_GPU_BATCH="${PER_GPU_BATCH:-512}"
export GRAD_CACHE_CHUNK_SIZE="${GRAD_CACHE_CHUNK_SIZE:-128}"
export VARIANT_TAG="${VARIANT_TAG:-vctx576_aud_wmid_t8_bs4k_g128sfix}"
export VERSION="${VERSION:-3var_gsv2full_gsdedup_varctx576_aud_wmid_bs4k_gc${GRAD_CACHE_CHUNK_SIZE}_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep${EPOCHS:-6}_v3_dev100Tau1_eval100_smallestfix_taurus8}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-variantE_varctx576_aud_wmid_taurus8_bs4k_gc${GRAD_CACHE_CHUNK_SIZE}_dev100Tau1_eval100_smallestfix}"
export NOTES_FILE="${NOTES_FILE:-${REPO_ROOT}/documents/code/train/term_train/notes/2026/05/20260517__varctx_lmlb_v3_audio_whisper_mid_en_taurus8_bs4k_gc128_eval100_smallestfix.md}"
export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:vctx576_aud_wmid_t8_bs4k_g128sfix compute:taurus-8gpu}"
export MASTER_PORT="${MASTER_PORT:-30456}"
export RUN_VERDICT="${RUN_VERDICT:-Whisper-medium.en speech-encoder ablation with BGE-M3 text encoder; smallestfix low-memory MaxSim; per-rank batch 512/global batch 4096 to fit Whisper gc128 re-forward on Taurus A6000.}"

if [ ! -f "${PARENT_LAUNCHER}" ]; then
  echo "[ERROR] required parent launcher missing: ${PARENT_LAUNCHER}" >&2
  exit 2
fi
if [ ! -f "${NOTES_FILE}" ]; then
  echo "[ERROR] required notes file missing: ${NOTES_FILE}" >&2
  exit 2
fi

exec bash "${PARENT_LAUNCHER}"
