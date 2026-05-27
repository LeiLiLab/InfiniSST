#!/usr/bin/env bash
set -euo pipefail

# Stage the JA cap16-denoise HF export onto Aries local NVMe for faster vLLM load.
# If the HF export is still being produced, this waits until the source is complete.

HOST_LABEL="${HOST_LABEL:-$(hostname -s)}"
SRC_MODEL="${SRC_MODEL:-/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_ja_cap16_denoise_budget_ttag_r32a32_ep1_taurus4/keep1.0_r32/v2-20260525-235251-hf}"
DST_MODEL="${DST_MODEL:-/mnt/data3/jiaxuanluo/slm_local_cache/ja_tagged_acl_20260525/cap16_denoise_ttag/v2-20260525-235251-hf}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/stage_ja_cap16_denoise_local_cache_aries_20260525}"
BW_LIMIT_KB="${BW_LIMIT_KB:-120000}"
POLL_SECS="${POLL_SECS:-60}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-21600}"

mkdir -p "${LOG_ROOT}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

fail() {
  log "[ERROR] $*" >&2
  exit 3
}

validate_hf_dir() {
  local path="$1"
  python3 - "${path}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
required = [root / "config.json", root / "generation_config.json", root / "model.safetensors.index.json"]
missing = [str(p) for p in required if not p.is_file() or p.stat().st_size <= 0]
if missing:
    raise SystemExit(f"missing required files: {missing}")
index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
expected = sorted(set(index.get("weight_map", {}).values()))
present = sorted(p.name for p in root.glob("*.safetensors"))
missing_weights = [name for name in expected if name not in present]
zero = [p.name for p in root.glob("*.safetensors") if p.stat().st_size <= 0]
if len(present) != 15 or missing_weights or zero:
    raise SystemExit(
        f"bad weights present={len(present)} missing={missing_weights[:5]} zero={zero[:5]}"
    )
print(f"ok weights={len(present)}")
PY
}

wait_for_source() {
  local waited=0
  while true; do
    if [[ -d "${SRC_MODEL}" ]] && validate_hf_dir "${SRC_MODEL}" >/dev/null 2>&1; then
      validate_hf_dir "${SRC_MODEL}" | sed 's/^/[SRC] /'
      return 0
    fi
    if (( waited >= MAX_WAIT_SECS )); then
      fail "Timed out waiting for complete HF source: ${SRC_MODEL}"
    fi
    log "[WAIT] HF source not ready yet: ${SRC_MODEL}"
    sleep "${POLL_SECS}"
    waited=$((waited + POLL_SECS))
  done
}

stage_model() {
  local complete="${DST_MODEL}/.stage_complete"
  if [[ -f "${complete}" ]]; then
    validate_hf_dir "${DST_MODEL}" | sed 's/^/[DST] /'
    log "[DONE] local cache already complete: ${DST_MODEL}"
    return 0
  fi
  if [[ -e "${DST_MODEL}" ]]; then
    fail "Destination exists without .stage_complete; refusing partial cache: ${DST_MODEL}"
  fi

  local tmp="${DST_MODEL}.tmp.$(date -u +%Y%m%dT%H%M%S).$$"
  if [[ -e "${tmp}" ]]; then
    fail "Temporary destination already exists: ${tmp}"
  fi
  mkdir -p "$(dirname "${DST_MODEL}")" "${tmp}"

  log "[COPY] host=${HOST_LABEL}"
  log "[COPY] src=${SRC_MODEL}"
  log "[COPY] tmp=${tmp}"
  log "[COPY] dst=${DST_MODEL}"
  rsync -a --delete --partial --info=progress2,stats2,name1 --bwlimit="${BW_LIMIT_KB}" \
    "${SRC_MODEL}/" "${tmp}/"

  validate_hf_dir "${tmp}" | sed 's/^/[TMP] /'
  date -u +%Y-%m-%dT%H:%M:%SZ > "${tmp}/.stage_complete"
  mv "${tmp}" "${DST_MODEL}"
  validate_hf_dir "${DST_MODEL}" | sed 's/^/[DST] /'
  du -sh "${DST_MODEL}"
  log "[DONE] staged JA cap16-denoise local cache: ${DST_MODEL}"
}

log "[START] stage JA cap16-denoise local cache"
wait_for_source
stage_model
