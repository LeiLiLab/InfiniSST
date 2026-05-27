#!/bin/bash
#SBATCH --job-name=q3_vctx_aud_wavlm
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --gres=gpu:8
#SBATCH --time=4-00:00:00
#SBATCH --output=/mnt/gemini/home/jiaxuanluo/logs/%j_q3_vctx_aud_wavlm_%x.out
#SBATCH --error=/mnt/gemini/home/jiaxuanluo/logs/%j_q3_vctx_aud_wavlm_%x.err

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export AUDIO_ENCODER_PRESET="${AUDIO_ENCODER_PRESET:-wavlm-large-plus}"
export AUDIO_ENCODER_TYPE="${AUDIO_ENCODER_TYPE:-wavlm}"
export AUDIO_MODEL_ID="${AUDIO_MODEL_ID:-microsoft/wavlm-large-plus}"
export AUDIO_FEATURE_EXTRACTOR_ID="${AUDIO_FEATURE_EXTRACTOR_ID:-microsoft/wavlm-large-plus}"
export AUDIO_INPUT_DTYPE="${AUDIO_INPUT_DTYPE:-auto}"
export TARGET_MODULES="${TARGET_MODULES:-q_proj k_proj v_proj out_proj intermediate_dense output_dense}"
export VARIANT_TAG="${VARIANT_TAG:-hn1024_varctx576_v3_aud_wavlmlp}"
export VERSION="${VERSION:-3var_gsv2full_gsdedup_varctx576_aud_wavlmlp_gc${GRAD_CACHE_CHUNK_SIZE:-512}_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep${EPOCHS:-6}_v3_bs12k_smallest_dense_normAGGR_8gpu_aries}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-variantE_hn1024_varctx576_v3_aud_wavlmlp_gc${GRAD_CACHE_CHUNK_SIZE:-512}_tcmoff_ep${EPOCHS:-6}_8gpu_aries}"
export NOTES_FILE="${NOTES_FILE:-/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hn1024_gsv2full_gsdedup_varctx_lmlb_v3_encoder_ablation.md}"
export DATA_TAG="${DATA_TAG:-3variant_gsv2full_gsdedup_varctx576_aud_wavlmlp}"
export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:varctx576_aud_wavlmlp compute:aries-8gpu}"
export MASTER_PORT="${MASTER_PORT:-30014}"

exec bash "${BASE_DIR}/run_mfa_smallest_dense_hn1024_gsv2full_gsdedup_varctx_lmlb_v3_tcmoff_ep6_8gpu_aries.sh"
