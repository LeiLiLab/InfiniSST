#!/usr/bin/env bash
set -euo pipefail

# Fast PSC runtime smoke for vLLM MoE native ops. This intentionally does not
# load any model; it only verifies the Python env/userspace can register the
# kernels needed before spending queue time on a full vLLM engine startup.

PSC_BASE="${PSC_BASE:-/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval}"
ROOT_DIR="${ROOT_DIR:-${PSC_BASE}/src/InfiniSST}"
ENV_DIR="${ENV_DIR:-${PSC_BASE}/envs/spaCyEnv_20260518}"
SELF_SCRIPT="${SELF_SCRIPT:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__psc_vllm_moe_ops_smoke.sh}"
USE_APPTAINER="${USE_APPTAINER:-1}"
APPTAINER_SIF="${APPTAINER_SIF:-${PSC_BASE}/containers/ubuntu_22_04_gcc.sif}"
IN_APPTAINER="${IN_APPTAINER:-0}"

if [[ "${USE_APPTAINER}" == "1" && "${IN_APPTAINER}" != "1" ]]; then
  if [[ ! -f "${APPTAINER_SIF}" ]]; then
    echo "[ERROR] APPTAINER_SIF not found: ${APPTAINER_SIF}" >&2
    exit 3
  fi
  export IN_APPTAINER=1
  export APPTAINERENV_IN_APPTAINER=1
  export APPTAINERENV_PSC_BASE="${PSC_BASE}"
  export APPTAINERENV_ROOT_DIR="${ROOT_DIR}"
  export APPTAINERENV_ENV_DIR="${ENV_DIR}"
  exec apptainer exec --nv -B /ocean,/jet "${APPTAINER_SIF}" bash "${SELF_SCRIPT}" "$@"
fi

export PATH="${ENV_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export CONDA_PREFIX="${ENV_DIR}"
export CONDA_DEFAULT_ENV="spaCyEnv_20260518"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

"${ENV_DIR}/bin/python" - <<'PY'
import platform
import socket
import subprocess
import sys
import traceback


def ldd_version() -> str:
    try:
        out = subprocess.check_output(["ldd", "--version"], text=True, stderr=subprocess.STDOUT)
        return out.splitlines()[0]
    except Exception as exc:
        return f"ldd unavailable: {exc!r}"


print("hostname", socket.gethostname(), flush=True)
print("platform", platform.platform(), flush=True)
print("ldd", ldd_version(), flush=True)

try:
    import torch

    print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
    print("torch_cuda_available", torch.cuda.is_available(), flush=True)
    print("torch_cuda_device_count", torch.cuda.device_count(), flush=True)
    if torch.cuda.is_available():
        print("gpu0_name", torch.cuda.get_device_name(0), flush=True)
        print("gpu0_capability", torch.cuda.get_device_capability(0), flush=True)

    import vllm
    print("vllm", vllm.__version__, vllm.__file__, flush=True)

    import vllm._C  # noqa: F401
    print("import_vllm_C OK", flush=True)
    import vllm._moe_C  # noqa: F401
    print("import_vllm_moe_C OK", flush=True)
    import vllm._custom_ops  # noqa: F401
    print("import_vllm_custom_ops OK", flush=True)

    has_namespace = hasattr(torch.ops, "_moe_C")
    print("has_torch_ops_moe_C", has_namespace, flush=True)
    required = [
        "topk_softmax",
        "moe_align_block_size",
        "batched_moe_align_block_size",
        "moe_sum",
    ]
    missing = []
    for name in required:
        ok = bool(has_namespace and hasattr(torch.ops._moe_C, name))
        print(f"op_{name}", ok, flush=True)
        if not ok:
            missing.append(name)

    if missing:
        print("missing_ops", ",".join(missing), flush=True)
        raise SystemExit(20)

    print("vllm_moe_ops_smoke OK", flush=True)
except SystemExit:
    raise
except Exception:
    traceback.print_exc()
    raise SystemExit(10)
PY
