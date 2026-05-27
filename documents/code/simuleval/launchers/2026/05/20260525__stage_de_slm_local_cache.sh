#!/usr/bin/env bash
set -euo pipefail

HOST_LABEL="${HOST_LABEL:?set HOST_LABEL}"
DEST_BASE="${DEST_BASE:?set DEST_BASE}"
BW_LIMIT_KB="${BW_LIMIT_KB:-80000}"

SRC_CAP16="/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_de_retriever_cap16_exactboundary_r32a32_ep4_taurus8/keep1.0_r32/v1-20260525-141908-hf"
SRC_DENOISE="/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_de_cap16_denoise_budget_ttag_r32a32_ep1_taurus6/keep1.0_r32/v0-20260525-203735-hf"

echo "[INFO] host=${HOST_LABEL}"
echo "[INFO] dest_base=${DEST_BASE}"
echo "[INFO] bwlimit_kb=${BW_LIMIT_KB}"
date -u +"[INFO] start_utc=%Y-%m-%dT%H:%M:%SZ"

stage_model() {
  local label="$1"
  local src="$2"
  local dst="$3"
  local tmp="${dst}.tmp"
  local complete="${dst}/.stage_complete"

  echo "[INFO] staging label=${label}"
  echo "[INFO] src=${src}"
  echo "[INFO] dst=${dst}"

  if [[ ! -d "${src}" ]]; then
    echo "[ERROR] missing source: ${src}" >&2
    exit 2
  fi

  if [[ -f "${complete}" ]]; then
    local n
    n="$(find "${dst}" -maxdepth 1 -type f -name '*.safetensors' | wc -l)"
    if [[ "${n}" == "15" ]]; then
      echo "[INFO] already complete label=${label} safetensors=${n}"
      du -sh "${dst}"
      return 0
    fi
    echo "[ERROR] complete marker exists but safetensors=${n}: ${dst}" >&2
    exit 3
  fi

  if [[ -e "${dst}" && ! -d "${dst}" ]]; then
    echo "[ERROR] destination exists and is not a directory: ${dst}" >&2
    exit 4
  fi
  if [[ -d "${dst}" && ! -f "${complete}" ]]; then
    echo "[ERROR] destination directory exists without complete marker: ${dst}" >&2
    echo "[ERROR] refusing to silently reuse a possibly partial cache" >&2
    exit 5
  fi

  mkdir -p "$(dirname "${dst}")"
  mkdir -p "${tmp}"

  rsync -a \
    --delete \
    --partial \
    --info=progress2,stats2,name1 \
    --bwlimit="${BW_LIMIT_KB}" \
    "${src}/" "${tmp}/"

  local shard_count
  shard_count="$(find "${tmp}" -maxdepth 1 -type f -name '*.safetensors' | wc -l)"
  if [[ "${shard_count}" != "15" ]]; then
    echo "[ERROR] staged shard count mismatch label=${label} count=${shard_count}" >&2
    exit 6
  fi

  date -u +"%Y-%m-%dT%H:%M:%SZ" > "${tmp}/.stage_complete"
  mv "${tmp}" "${dst}"
  du -sh "${dst}"
  echo "[INFO] staged complete label=${label} dst=${dst}"
}

stage_model \
  "cap16_exactboundary" \
  "${SRC_CAP16}" \
  "${DEST_BASE}/cap16_exactboundary/v1-20260525-141908-hf"

stage_model \
  "cap16_denoise_ttag" \
  "${SRC_DENOISE}" \
  "${DEST_BASE}/cap16_denoise_ttag/v0-20260525-203735-hf"

date -u +"[INFO] done_utc=%Y-%m-%dT%H:%M:%SZ"
