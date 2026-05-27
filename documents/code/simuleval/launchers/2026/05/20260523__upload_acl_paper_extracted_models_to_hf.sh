#!/usr/bin/env bash
set -euo pipefail

# Upload the two ACL paper-extracted main-result model families to private HF
# model repos.  This stages symlinks so upload metadata is written to a writable
# directory instead of the model export directory.

MODEL_ROOT_OWASKI="${MODEL_ROOT_OWASKI:-/mnt/gemini/data/jiaxuanluo/owaski}"
STAGE_ROOT="${STAGE_ROOT:-/mnt/aries/data6/jiaxuanluo/hf_upload_staging/acl_paper_extracted_models_20260523}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ONLY_MODELS="${ONLY_MODELS:-}"
HF_PRIVATE="${HF_PRIVATE:-1}"

mkdir -p "${STAGE_ROOT}"

model_specs() {
  cat <<'EOF'
no_tmsft	zh	gigaspeech-zh-s_origin-bsz4	gavinlaw/infinisst-no-tmsft-origin-bsz4-zh
no_tmsft	de	gigaspeech-de-s_origin-bsz4	gavinlaw/infinisst-no-tmsft-origin-bsz4-de
no_tmsft	ja	gigaspeech-ja-s_origin-bsz4	gavinlaw/infinisst-no-tmsft-origin-bsz4-ja
rasst	zh	gigaspeech-zh-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4	gavinlaw/infinisst-llmgen-rasst-bsz4-zh
rasst	de	gigaspeech-de-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4	gavinlaw/infinisst-llmgen-rasst-bsz4-de
rasst	ja	gigaspeech-ja-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4	gavinlaw/infinisst-llmgen-rasst-bsz4-ja
EOF
}

should_upload() {
  local key="$1"
  [[ -z "${ONLY_MODELS}" ]] && return 0
  local requested
  for requested in ${ONLY_MODELS}; do
    [[ "${requested}" == "${key}" ]] && return 0
  done
  return 1
}

validate_model_dir() {
  local model_dir="$1"
  if [[ ! -f "${model_dir}/config.json" ]]; then
    echo "[ERROR] Missing config.json: ${model_dir}" >&2
    return 3
  fi
  local shard_count
  shard_count="$(find "${model_dir}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
  if [[ "${shard_count}" != "15" ]]; then
    echo "[ERROR] Expected 15 safetensor shards, found ${shard_count}: ${model_dir}" >&2
    return 3
  fi
}

hf auth whoami
repo_create_args=(--repo-type model --exist-ok)
upload_args=(--repo-type model --num-workers "${NUM_WORKERS}" --no-bars)
if [[ "${HF_PRIVATE}" == "1" ]]; then
  repo_create_args+=(--private)
  upload_args+=(--private)
fi

while IFS=$'\t' read -r method lang dirname repo_id; do
  key="${method}_${lang}"
  if ! should_upload "${key}"; then
    echo "[SKIP] ${key}: not requested by ONLY_MODELS=${ONLY_MODELS}"
    continue
  fi
  model_dir="${MODEL_ROOT_OWASKI}/${dirname}"
  stage_dir="${STAGE_ROOT}/${dirname}"
  validate_model_dir "${model_dir}"

  echo "[INFO] Uploading ${key}"
  echo "[INFO] model_dir=${model_dir}"
  echo "[INFO] repo_id=${repo_id}"
  du -sh "${model_dir}"

  mkdir -p "${stage_dir}"
  find "${model_dir}" -maxdepth 1 -type f -print0 | while IFS= read -r -d '' f; do
    ln -sfn "${f}" "${stage_dir}/$(basename "${f}")"
  done
  find -L "${stage_dir}" -maxdepth 1 -type f -printf '%f\t%s\n' | sort > "${stage_dir}/_files.tsv"

  hf repo create "${repo_id}" "${repo_create_args[@]}"
  hf upload-large-folder \
    "${upload_args[@]}" \
    "${repo_id}" \
    "${stage_dir}"
  echo "[DONE] ${key}: https://huggingface.co/${repo_id}"
done < <(model_specs)

echo "[ALL DONE] ACL paper-extracted model upload loop complete."
