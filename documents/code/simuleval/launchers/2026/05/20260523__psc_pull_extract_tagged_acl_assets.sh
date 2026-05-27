#!/usr/bin/env bash
set -euo pipefail

PSC_BASE="${PSC_BASE:-/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval}"
ROOT_DIR="${ROOT_DIR:-${PSC_BASE}/src/InfiniSST}"
ASSET_DIR="${ASSET_DIR:-${PSC_BASE}/assets/tagged_acl_newv5_r32}"
MANIFEST_TSV="${MANIFEST_TSV:-${PSC_BASE}/assets/psc_tagged_acl_newv5_r32_assets_files.tsv}"
DOWNLOADER="${DOWNLOADER:-${ROOT_DIR}/documents/code/simuleval/tools/hf_resolve_snapshot_download_curl.sh}"
HF_REPO_ID="${HF_REPO_ID:-gavinlaw/infinisst-psc-tagged-acl-newv5-r32-assets}"

CKPT_NAME="${CKPT_NAME:-lh1b88kw_best_eval_acl6060_recallat10.pt}"
ENV_TAR_NAME="${ENV_TAR_NAME:-spaCyEnv_20260518.tar.gz}"
ACL_TAR_NAME="${ACL_TAR_NAME:-acl6060_20260523.tar.gz}"
MWER_TAR_NAME="${MWER_TAR_NAME:-mwerSegmenter_20260523.tar.gz}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PSC_BASE}/checkpoints}"
ENV_DIR="${ENV_DIR:-${PSC_BASE}/envs/spaCyEnv_20260518}"
DATA_DIR="${DATA_DIR:-${PSC_BASE}/data}"
TOOLS_DIR="${TOOLS_DIR:-${PSC_BASE}/tools}"

for p in "${DOWNLOADER}" "${MANIFEST_TSV}"; do
  if [[ ! -f "${p}" ]]; then
    echo "[ERROR] Missing required control file: ${p}" >&2
    exit 3
  fi
done

mkdir -p "${ASSET_DIR}" "${CHECKPOINT_DIR}" "${PSC_BASE}/envs" "${DATA_DIR}" "${TOOLS_DIR}" "${PSC_BASE}/cache/hf"

echo "[INFO] Pulling HF assets into ${ASSET_DIR}"
bash "${DOWNLOADER}" \
  --repo "${HF_REPO_ID}" \
  --repo-type dataset \
  --manifest "${MANIFEST_TSV}" \
  --out-dir "${ASSET_DIR}"

ln -sfn "${ASSET_DIR}/checkpoints/${CKPT_NAME}" "${CHECKPOINT_DIR}/${CKPT_NAME}"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  echo "[EXTRACT] env -> ${ENV_DIR}"
  rm -rf "${ENV_DIR}.tmp"
  mkdir -p "${ENV_DIR}.tmp"
  tar -xzf "${ASSET_DIR}/envs/${ENV_TAR_NAME}" -C "${ENV_DIR}.tmp"
  rm -rf "${ENV_DIR}"
  mv "${ENV_DIR}.tmp" "${ENV_DIR}"
  if [[ -x "${ENV_DIR}/bin/conda-unpack" ]]; then
    "${ENV_DIR}/bin/conda-unpack"
  fi
else
  echo "[SKIP] existing env: ${ENV_DIR}"
fi

if [[ ! -f "${DATA_DIR}/acl6060/dev.yaml" ]]; then
  echo "[EXTRACT] ACL6060 data -> ${DATA_DIR}"
  tar -xzf "${ASSET_DIR}/data/${ACL_TAR_NAME}" -C "${DATA_DIR}"
else
  echo "[SKIP] existing ACL6060 data: ${DATA_DIR}/acl6060"
fi

if [[ ! -x "${TOOLS_DIR}/mwerSegmenter/mwerSegmenter" && ! -f "${TOOLS_DIR}/mwerSegmenter/mwerSegmenter" ]]; then
  echo "[EXTRACT] mwerSegmenter -> ${TOOLS_DIR}"
  tar -xzf "${ASSET_DIR}/tools/${MWER_TAR_NAME}" -C "${TOOLS_DIR}"
else
  echo "[SKIP] existing mwerSegmenter: ${TOOLS_DIR}/mwerSegmenter"
fi

for p in \
  "${CHECKPOINT_DIR}/${CKPT_NAME}" \
  "${ENV_DIR}/bin/python" \
  "${DATA_DIR}/acl6060/dev.yaml" \
  "${TOOLS_DIR}/mwerSegmenter"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Post-extract validation missing: ${p}" >&2
    exit 3
  fi
done

"${ENV_DIR}/bin/python" - <<'PY'
import importlib
mods = ["torch", "transformers", "vllm", "simuleval", "yaml", "soundfile", "wandb"]
for mod in mods:
    importlib.import_module(mod)
print("[INFO] Python import validation passed:", ",".join(mods))
PY

echo "[ALL DONE] PSC assets are ready under ${PSC_BASE}"
