#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260526__medicine_ja_serial_promptfix_vllmaudio128_max40lm_aries.sh}"

RUN_STAMP="${RUN_STAMP:-20260526T0636_medicine_ja_lm1_tailfirst_temp_promptfix_vllmaudio128_aries01}"
OUT_ROOT="${OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/medicine_ja_lm1_tailfirst_temp_promptfix_vllmaudio128_20260526T0636_aries01}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/medicine_ja_lm1_tailfirst_temp_promptfix_vllmaudio128_20260526T0636_aries01}"
INPUT_ROOT="${INPUT_ROOT:-${OUT_ROOT}/inputs_tailfirst}"

exec env \
  ROOT_DIR="${ROOT_DIR}" \
  RUN_STAMP="${RUN_STAMP}" \
  OUT_ROOT="${OUT_ROOT}" \
  LOG_ROOT="${LOG_ROOT}" \
  PID_FILE="${LOG_ROOT}/launcher.pid" \
  GPU_PAIR="${GPU_PAIR:-0,1}" \
  LM=1 \
  VLLM_LIMIT_AUDIO=128 \
  VLLM_MAX_MODEL_LEN=12288 \
  MAX_CACHE_CHUNKS=30 \
  KEEP_CACHE_CHUNKS=30 \
  MAX_NEW_TOKENS=40 \
  SRC_LIST="${INPUT_ROOT}/medicine.source__medicine5_tailfirst_605000_606_first.txt" \
  TGT_LIST="${INPUT_ROOT}/medicine.target.ja__medicine5_tailfirst_605000_606_first.txt" \
  SOURCE_TEXT="${INPUT_ROOT}/medicine.source_text.en__medicine5_tailfirst_605000_606_first.txt" \
  REF_FILE="${INPUT_ROOT}/medicine.ref.ja__medicine5_tailfirst_605000_606_first.txt" \
  AUDIO_YAML="${INPUT_ROOT}/medicine.audio__medicine5_tailfirst_605000_606_first.yaml" \
  DENSITY_TAG="medhard_ja_lm1_tailfirst_temp_promptfix_vllmaudio128" \
  EVAL_TMPDIR="/tmp/jxmjtail1" \
  bash "${BASE_LAUNCHER}"
