#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export INPUT_JSONL="${INPUT_JSONL:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup.jsonl}"
export OUTPUT_JSONL="${OUTPUT_JSONL:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx2p88_3p84_4p80_5p76.jsonl}"
export AUDIO_OUTPUT_DIR="${AUDIO_OUTPUT_DIR:-/mnt/gemini/home/jiaxuanluo/term_train_audio_chunks_gsv2full_gsdedup_varctx2p88_3p84_4p80_5p76}"
export WIKI_AUDIO_OUTPUT_DIR="${WIKI_AUDIO_OUTPUT_DIR:-${AUDIO_OUTPUT_DIR}/wiki_synth}"
export STATS_JSON="${STATS_JSON:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx2p88_3p84_4p80_5p76_stats.json}"

export DURATION_SECS="${DURATION_SECS:-2.88 3.84 4.80 5.76}"
export DURATION_ASSIGNMENT="${DURATION_ASSIGNMENT:-balance_rows}"

exec bash "${SCRIPT_DIR}/run_build_gsv2full_gsdedup_varctx_0p96_1p92_2p88_3p84.sh"
