#!/usr/bin/env bash
# Shared helpers for Megatron-core adapter -> HF export.
#
# Optional acceleration:
#   HF_EXPORT_STAGE_ROOT=/tmp/jx_hf_export
#
# When set, swift writes the HF export to a local/near-node staging directory
# first, validates it, then syncs into HF_OUTPUT_DIR via a temporary sibling and
# atomically renames it into place. This keeps large shard writes off slow NFS
# during the export itself and avoids publishing partial HF directories.

if [[ -n "${_INFINSST_HF_EXPORT_STAGING_SH:-}" ]]; then
  return 0
fi
_INFINSST_HF_EXPORT_STAGING_SH=1

hf_export_validate_dir() {
  local out_dir="$1"
  python3 - "${out_dir}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
required = [out / "config.json", out / "generation_config.json"]
missing = [str(p) for p in required if not p.exists()]
weights = sorted(out.glob("*.safetensors")) + sorted(out.glob("pytorch_model*.bin"))
zero_size = [str(p) for p in weights if p.stat().st_size <= 0]
index = out / "model.safetensors.index.json"
if index.exists():
    payload = json.loads(index.read_text(encoding="utf-8"))
    expected = sorted(set(payload.get("weight_map", {}).values()))
    present = sorted(p.name for p in out.glob("*.safetensors"))
    missing_weights = [name for name in expected if name not in present]
else:
    missing_weights = []
if missing or not weights or zero_size or missing_weights:
    raise SystemExit(
        "HF export incomplete: "
        f"dir={out} missing={missing} weights={len(weights)} "
        f"zero_size={zero_size[:5]} missing_weights={missing_weights[:5]}"
    )
print(f"[OK] HF export complete: weights={len(weights)} dir={out}", flush=True)
PY
}

hf_export_check_free_space() {
  local path="$1"
  local min_free_gb="${2:-90}"
  mkdir -p "${path}"
  python3 - "${path}" "${min_free_gb}" <<'PY'
import shutil
import sys
from pathlib import Path

path = Path(sys.argv[1])
min_free_gb = float(sys.argv[2])
free_gb = shutil.disk_usage(path).free / (1024 ** 3)
print(f"[INFO] free_space path={path} free_gb={free_gb:.1f} required_gb={min_free_gb:.1f}", flush=True)
if free_gb < min_free_gb:
    raise SystemExit(f"Not enough free space for staged HF export: {path} has {free_gb:.1f} GB")
PY
}

hf_export_sync_dir() {
  local src_dir="$1"
  local dst_dir="$2"
  mkdir -p "${dst_dir}"
  if command -v rsync >/dev/null 2>&1; then
    rsync ${HF_EXPORT_RSYNC_FLAGS:--a --delete --info=progress2} "${src_dir%/}/" "${dst_dir%/}/"
  else
    echo "[WARN] rsync not found; falling back to cp -a" >&2
    rm -rf "${dst_dir:?}/"*
    cp -a "${src_dir%/}/." "${dst_dir%/}/"
  fi
}

hf_export_publish_local_cache() {
  local src_dir="$1"
  local final_base="$2"
  local cache_root="${HF_EXPORT_LOCAL_CACHE_ROOT:-}"
  local latest_link="${HF_EXPORT_LOCAL_LATEST_LINK:-}"
  if [[ -z "${cache_root}" || "${cache_root}" == "0" || "${cache_root}" == "false" || "${cache_root}" == "none" ]]; then
    return 0
  fi

  local min_free_gb="${HF_EXPORT_LOCAL_CACHE_MIN_FREE_GB:-90}"
  hf_export_check_free_space "${cache_root}" "${min_free_gb}"

  local cache_dir="${cache_root%/}/${final_base}"
  local cache_tmp="${cache_dir}.tmp.$$"
  echo "[INFO] Publishing local HF cache:"
  echo "[INFO]   src=${src_dir}"
  echo "[INFO]   cache_tmp=${cache_tmp}"
  echo "[INFO]   cache_dir=${cache_dir}"

  rm -rf "${cache_tmp}"
  hf_export_sync_dir "${src_dir}" "${cache_tmp}"
  hf_export_validate_dir "${cache_tmp}"
  rm -rf "${cache_dir}"
  mv "${cache_tmp}" "${cache_dir}"
  hf_export_validate_dir "${cache_dir}"

  if [[ -n "${latest_link}" && "${latest_link}" != "0" && "${latest_link}" != "false" && "${latest_link}" != "none" ]]; then
    mkdir -p "$(dirname "${latest_link}")"
    ln -sfn "${cache_dir}" "${latest_link}"
    echo "[INFO] Updated local latest HF symlink: ${latest_link} -> ${cache_dir}"
  fi
}

export_mcore_checkpoint_to_hf_staged() {
  local mcore_adapters="$1"
  local hf_output_dir="$2"
  local swift_cmd="${SWIFT_CMD:-swift}"
  local torch_dtype="${SWIFT_TORCH_DTYPE:-${TORCH_DTYPE:-bfloat16}}"
  local stage_root="${HF_EXPORT_STAGE_ROOT:-}"
  local min_free_gb="${HF_EXPORT_MIN_FREE_GB:-90}"
  local cuda_env=()
  local swift_extra_args=()
  if [[ -n "${CUDA_VISIBLE_DEVICES-}" ]]; then
    cuda_env=(env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
  fi
  if [[ -n "${HF_EXPORT_SWIFT_EXTRA_ARGS:-}" ]]; then
    read -r -a swift_extra_args <<< "${HF_EXPORT_SWIFT_EXTRA_ARGS}"
  fi

  if [[ ! -d "${mcore_adapters}" ]]; then
    echo "[ERROR] Missing mcore adapters: ${mcore_adapters}" >&2
    return 3
  fi
  if ! command -v "${swift_cmd}" >/dev/null 2>&1; then
    echo "[ERROR] SWIFT_CMD is not available in PATH: ${swift_cmd}" >&2
    return 3
  fi

  mkdir -p "$(dirname "${hf_output_dir}")"

  if [[ -z "${stage_root}" || "${stage_root}" == "0" || "${stage_root}" == "false" || "${stage_root}" == "none" ]]; then
    echo "[INFO] HF export mode=direct output_dir=${hf_output_dir}"
    "${cuda_env[@]}" "${swift_cmd}" export \
      --mcore_adapters "${mcore_adapters}" \
      --to_hf true \
      --torch_dtype "${torch_dtype}" \
      --output_dir "${hf_output_dir}" \
      "${swift_extra_args[@]}"
    hf_export_validate_dir "${hf_output_dir}"
    hf_export_publish_local_cache "${hf_output_dir}" "$(basename "${hf_output_dir}")"
    return 0
  fi

  hf_export_check_free_space "${stage_root}" "${min_free_gb}"
  hf_export_check_free_space "$(dirname "${hf_output_dir}")" "${min_free_gb}"

  local final_base
  final_base="$(basename "${hf_output_dir}")"
  local stage_dir="${stage_root%/}/${final_base}.stage.$$"
  local final_tmp="${hf_output_dir}.tmp.$$"

  if [[ -e "${hf_output_dir}" && "${HF_EXPORT_OVERWRITE:-0}" != "1" ]]; then
    echo "[ERROR] HF output already exists. Set HF_EXPORT_OVERWRITE=1 to replace: ${hf_output_dir}" >&2
    return 3
  fi

  echo "[INFO] HF export mode=staged"
  echo "[INFO] mcore_adapters=${mcore_adapters}"
  echo "[INFO] stage_dir=${stage_dir}"
  echo "[INFO] final_tmp=${final_tmp}"
  echo "[INFO] final_dir=${hf_output_dir}"

  rm -rf "${stage_dir}" "${final_tmp}"
  mkdir -p "$(dirname "${stage_dir}")"

  "${cuda_env[@]}" "${swift_cmd}" export \
    --mcore_adapters "${mcore_adapters}" \
    --to_hf true \
    --torch_dtype "${torch_dtype}" \
    --output_dir "${stage_dir}" \
    "${swift_extra_args[@]}"

  hf_export_validate_dir "${stage_dir}"
  hf_export_sync_dir "${stage_dir}" "${final_tmp}"
  hf_export_validate_dir "${final_tmp}"

  if [[ -e "${hf_output_dir}" ]]; then
    if [[ "${HF_EXPORT_OVERWRITE:-0}" == "1" ]]; then
      rm -rf "${hf_output_dir}"
    else
      echo "[ERROR] HF output appeared during export: ${hf_output_dir}" >&2
      return 3
    fi
  fi
  mv "${final_tmp}" "${hf_output_dir}"
  hf_export_validate_dir "${hf_output_dir}"
  hf_export_publish_local_cache "${stage_dir}" "${final_base}"

  if [[ "${HF_EXPORT_KEEP_STAGE:-0}" != "1" ]]; then
    rm -rf "${stage_dir}"
  else
    echo "[INFO] Keeping staged export dir: ${stage_dir}"
  fi
}
