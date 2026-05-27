#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
RUN_STAMP="${RUN_STAMP:-20260523T0410}"

HF_REPO_ID="${HF_REPO_ID:-gavinlaw/infinisst-psc-tagged-acl-newv5-r32-assets}"
HF_REPO_TYPE="${HF_REPO_TYPE:-dataset}"
STAGE_ROOT="${STAGE_ROOT:-/mnt/aries/data6/jiaxuanluo/hf_upload_staging/psc_tagged_acl_newv5_r32_assets}"
FILES_DIR="${FILES_DIR:-${STAGE_ROOT}/files}"
MANIFEST_TSV="${MANIFEST_TSV:-${STAGE_ROOT}/psc_tagged_acl_newv5_r32_assets_files.tsv}"

ACL_DATA_ROOT="${ACL_DATA_ROOT:-/mnt/taurus/data/siqiouyang/datasets/acl6060}"
RETRIEVER_CKPT="${RETRIEVER_CKPT:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
ENV_TAR="${ENV_TAR:-/mnt/gemini/home/jiaxuanluo/transfer_packages/spaCyEnv_20260518.tar.gz}"
MWERSEGMENTER_ROOT="${MWERSEGMENTER_ROOT:-/mnt/taurus/home/jiaxuanluo/mwerSegmenter}"

ACL_TAR_NAME="${ACL_TAR_NAME:-acl6060_20260523.tar.gz}"
MWER_TAR_NAME="${MWER_TAR_NAME:-mwerSegmenter_20260523.tar.gz}"
CKPT_NAME="${CKPT_NAME:-lh1b88kw_best_eval_acl6060_recallat10.pt}"
ENV_TAR_NAME="${ENV_TAR_NAME:-spaCyEnv_20260518.tar.gz}"

LOG_DIR="${LOG_DIR:-/mnt/gemini/data1/jiaxuanluo/logs}"
mkdir -p "${FILES_DIR}/data" "${FILES_DIR}/checkpoints" "${FILES_DIR}/envs" "${FILES_DIR}/tools" "${LOG_DIR}"

for p in "${ACL_DATA_ROOT}" "${RETRIEVER_CKPT}" "${ENV_TAR}" "${MWERSEGMENTER_ROOT}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

if [[ "${HF_REPO_TYPE}" != "dataset" ]]; then
  echo "[ERROR] This launcher expects HF_REPO_TYPE=dataset, got ${HF_REPO_TYPE}" >&2
  exit 2
fi

echo "[INFO] stage_root=${STAGE_ROOT}"
echo "[INFO] hf_repo=${HF_REPO_ID} repo_type=${HF_REPO_TYPE}"

if [[ ! -f "${FILES_DIR}/data/${ACL_TAR_NAME}" ]]; then
  echo "[PACK] ACL6060 data -> ${FILES_DIR}/data/${ACL_TAR_NAME}"
  tar -C "$(dirname "${ACL_DATA_ROOT}")" -czf "${FILES_DIR}/data/${ACL_TAR_NAME}" "$(basename "${ACL_DATA_ROOT}")"
else
  echo "[SKIP] existing ACL data tar: ${FILES_DIR}/data/${ACL_TAR_NAME}"
fi

if [[ ! -f "${FILES_DIR}/tools/${MWER_TAR_NAME}" ]]; then
  echo "[PACK] mwerSegmenter -> ${FILES_DIR}/tools/${MWER_TAR_NAME}"
  tar -C "$(dirname "${MWERSEGMENTER_ROOT}")" -czf "${FILES_DIR}/tools/${MWER_TAR_NAME}" "$(basename "${MWERSEGMENTER_ROOT}")"
else
  echo "[SKIP] existing mwerSegmenter tar: ${FILES_DIR}/tools/${MWER_TAR_NAME}"
fi

ln -sfn "${RETRIEVER_CKPT}" "${FILES_DIR}/checkpoints/${CKPT_NAME}"
ln -sfn "${ENV_TAR}" "${FILES_DIR}/envs/${ENV_TAR_NAME}"

find -L "${FILES_DIR}" -type f -printf '%P\t%s\n' | sort > "${MANIFEST_TSV}"
echo "[INFO] asset manifest=${MANIFEST_TSV}"
cat "${MANIFEST_TSV}"

hf repo create "${HF_REPO_ID}" --repo-type "${HF_REPO_TYPE}" --private --exist-ok
hf upload-large-folder "${HF_REPO_ID}" "${FILES_DIR}" \
  --repo-type "${HF_REPO_TYPE}" \
  --num-workers "${HF_UPLOAD_WORKERS:-8}"

echo "[INFO] Upload finished: https://huggingface.co/datasets/${HF_REPO_ID}"
