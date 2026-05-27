#!/usr/bin/env bash
set -euo pipefail

# Pull ACL paper-extracted main-result HF model repos onto PSC.

PSC_BASE="${PSC_BASE:-/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval}"
MODEL_ROOT_OWASKI="${MODEL_ROOT_OWASKI:-${PSC_BASE}/models/owaski}"
HF_HOME="${HF_HOME:-${PSC_BASE}/cache/hf}"
ONLY_MODELS="${ONLY_MODELS:-}"

export HF_HOME
mkdir -p "${MODEL_ROOT_OWASKI}" "${HF_HOME}"

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

should_pull() {
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
    echo "[ERROR] Missing config.json after pull: ${model_dir}" >&2
    return 3
  fi
  local shard_count
  shard_count="$(find "${model_dir}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
  if [[ "${shard_count}" != "15" ]]; then
    echo "[ERROR] Expected 15 safetensor shards after pull, found ${shard_count}: ${model_dir}" >&2
    return 3
  fi
}

hf auth whoami

while IFS=$'\t' read -r method lang dirname repo_id; do
  key="${method}_${lang}"
  if ! should_pull "${key}"; then
    echo "[SKIP] ${key}: not requested by ONLY_MODELS=${ONLY_MODELS}"
    continue
  fi
  target_dir="${MODEL_ROOT_OWASKI}/${dirname}"
  if [[ -f "${target_dir}/config.json" ]]; then
    shard_count="$(find "${target_dir}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
    if [[ "${shard_count}" == "15" ]]; then
      echo "[SKIP] existing complete model ${key}: ${target_dir}"
      continue
    fi
  fi

  echo "[GET] ${key} repo=${repo_id} -> ${target_dir}"
  mkdir -p "${target_dir}"
  hf download "${repo_id}" \
    --repo-type model \
    --local-dir "${target_dir}" \
    --local-dir-use-symlinks False
  validate_model_dir "${target_dir}"
  du -sh "${target_dir}"
  echo "[DONE] ${key}: ${target_dir}"
done < <(model_specs)

echo "[ALL DONE] ACL paper-extracted model pull loop complete."
