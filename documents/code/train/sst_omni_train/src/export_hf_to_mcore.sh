#!/usr/bin/env bash
set -euo pipefail

: "${HF_MODEL_DIR:?Set HF_MODEL_DIR to the HF checkpoint directory}"
: "${MCORE_OUTPUT_DIR:?Set MCORE_OUTPUT_DIR to the desired Megatron-core output directory}"

TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
CONVERT_OVERWRITE="${CONVERT_OVERWRITE:-0}"

if [[ ! -d "${HF_MODEL_DIR}" ]]; then
  echo "[ERROR] Missing HF_MODEL_DIR: ${HF_MODEL_DIR}" >&2
  exit 3
fi

if [[ ! -f "${HF_MODEL_DIR}/config.json" ]]; then
  echo "[ERROR] HF_MODEL_DIR is missing config.json: ${HF_MODEL_DIR}" >&2
  exit 3
fi

if ! compgen -G "${HF_MODEL_DIR}/*.safetensors" >/dev/null && \
   ! compgen -G "${HF_MODEL_DIR}/pytorch_model*.bin" >/dev/null; then
  echo "[ERROR] HF_MODEL_DIR has no safetensors or pytorch_model*.bin weights: ${HF_MODEL_DIR}" >&2
  exit 3
fi

if [[ -e "${MCORE_OUTPUT_DIR}" ]]; then
  if [[ "${CONVERT_OVERWRITE}" != "1" ]]; then
    echo "[ERROR] MCORE_OUTPUT_DIR already exists. Set CONVERT_OVERWRITE=1 to replace it: ${MCORE_OUTPUT_DIR}" >&2
    exit 2
  fi
  rm -rf "${MCORE_OUTPUT_DIR}"
fi

mkdir -p "$(dirname "${MCORE_OUTPUT_DIR}")"

echo "[INFO] HF_MODEL_DIR=${HF_MODEL_DIR}"
echo "[INFO] MCORE_OUTPUT_DIR=${MCORE_OUTPUT_DIR}"
echo "[INFO] TORCH_DTYPE=${TORCH_DTYPE}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

swift export \
  --model "${HF_MODEL_DIR}" \
  --to_mcore true \
  --torch_dtype "${TORCH_DTYPE}" \
  --output_dir "${MCORE_OUTPUT_DIR}"

python3 - "${MCORE_OUTPUT_DIR}" <<'PY'
import sys
from pathlib import Path

out = Path(sys.argv[1])
required = [out / "args.json", out / "latest_checkpointed_iteration.txt"]
missing = [str(p) for p in required if not p.exists()]
iters = [p for p in out.iterdir() if p.is_dir() and p.name.startswith("iter_")]
if missing or not iters:
    raise SystemExit(f"MCore export incomplete: missing={missing} iter_dirs={len(iters)} dir={out}")
print(f"[OK] MCore export complete: iter_dirs={len(iters)} dir={out}", flush=True)
PY
