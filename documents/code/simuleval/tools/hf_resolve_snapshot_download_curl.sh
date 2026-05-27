#!/usr/bin/env bash
set -euo pipefail

REPO_ID=""
REPO_TYPE="model"
REVISION="main"
MANIFEST=""
OUT_DIR=""
TOKEN_FILE="${HOME}/.cache/huggingface/token"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_ID="$2"; shift 2 ;;
    --repo-type) REPO_TYPE="$2"; shift 2 ;;
    --revision) REVISION="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --token-file) TOKEN_FILE="$2"; shift 2 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${REPO_ID}" || -z "${MANIFEST}" || -z "${OUT_DIR}" ]]; then
  echo "[ERROR] Required: --repo <repo> --manifest <tsv> --out-dir <dir>" >&2
  exit 2
fi
case "${REPO_TYPE}" in
  model|dataset) ;;
  *) echo "[ERROR] Unsupported --repo-type ${REPO_TYPE}; expected model or dataset" >&2; exit 2 ;;
esac
if [[ ! -s "${TOKEN_FILE}" ]]; then
  echo "[ERROR] Missing token file: ${TOKEN_FILE}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
cfg="$(mktemp)"
trap 'rm -f "${cfg}"' EXIT
chmod 600 "${cfg}"
printf 'header = "Authorization: Bearer %s"\n' "$(cat "${TOKEN_FILE}")" > "${cfg}"

urlencode_path() {
  python3 - "$1" <<'PY'
import sys
from urllib.parse import quote
print("/".join(quote(part) for part in sys.argv[1].split("/")))
PY
}

while IFS=$'\t' read -r name expected_size; do
  [[ -n "${name}" ]] || continue
  [[ "${name}" == \#* ]] && continue
  out="${OUT_DIR}/${name}"
  part="${out}.part"
  mkdir -p "$(dirname "${out}")"
  if [[ -f "${out}" ]]; then
    got="$(stat -c '%s' "${out}")"
    if [[ "${got}" == "${expected_size}" ]]; then
      echo "[SKIP] ${name} size=${expected_size}"
      continue
    fi
    echo "[ERROR] Existing output has wrong size: ${out} got=${got} expected=${expected_size}" >&2
    exit 3
  fi

  encoded="$(urlencode_path "${name}")"
  if [[ "${REPO_TYPE}" == "dataset" ]]; then
    url="https://huggingface.co/datasets/${REPO_ID}/resolve/${REVISION}/${encoded}"
  else
    url="https://huggingface.co/${REPO_ID}/resolve/${REVISION}/${encoded}"
  fi
  echo "[GET] ${name} expected=${expected_size} partial=$([[ -f "${part}" ]] && stat -c '%s' "${part}" || echo 0)"
  curl \
    --ipv4 \
    --fail \
    --location \
    --retry 8 \
    --retry-delay 5 \
    --connect-timeout 60 \
    --speed-time 120 \
    --speed-limit 1024 \
    --continue-at - \
    --output "${part}" \
    --config "${cfg}" \
    "${url}"
  got="$(stat -c '%s' "${part}")"
  if [[ "${got}" != "${expected_size}" ]]; then
    echo "[ERROR] Incomplete download: ${name} got=${got} expected=${expected_size}" >&2
    exit 3
  fi
  mv "${part}" "${out}"
  echo "[DONE] ${name}"
done < "${MANIFEST}"

echo "[ALL DONE] ${OUT_DIR}"
