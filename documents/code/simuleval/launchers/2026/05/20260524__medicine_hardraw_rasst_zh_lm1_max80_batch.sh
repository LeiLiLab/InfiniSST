#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
BATCH_LAUNCHER="${BATCH_LAUNCHER_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__medicine_hardraw_lm1to4_5samples_hn1024_tau078_new_v9_batch.sh}"

RUN_STAMP="${RUN_STAMP:-20260524T1742_medicine_rasst_zh_lm1_max80_batch}"
MODEL_LABEL="${MODEL_LABEL:-new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078_max80}"
OUT_ROOT="${OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/medicine_hardraw_hn1024_tau078_new_v9_batch_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/medicine_hardraw_hn1024_tau078_new_v9_batch_${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/medicine_hardraw_batch_${RUN_STAMP}}"
EVAL_TMPDIR="${EVAL_TMPDIR:-/tmp/jx_med_zh_lm1_m80}"

OLD_INPUTS="${OLD_INPUTS:-/mnt/gemini/data1/jiaxuanluo/medicine_hardraw_hn1024_tau078_new_v9_batch_20260524T0242/zh/__medicine_inputs__/lists}"
NEW_INPUTS="${OUT_ROOT}/zh/__medicine_inputs__/lists"
SHARED_AUDIO_ROOT="${SHARED_AUDIO_ROOT:-/mnt/gemini/data1/jiaxuanluo/medicine_hardraw_shared_audio_20260524}"

mkdir -p "${LOG_ROOT}" "${INDEX_CACHE_DIR}" "${EVAL_TMPDIR}" "${NEW_INPUTS}"
if [[ ! -s "${OLD_INPUTS}/medicine_inputs_manifest__medicine5_hardraw.json" ]]; then
  echo "[ERROR] Missing old prepared medicine inputs: ${OLD_INPUTS}" >&2
  exit 3
fi
if command -v rsync >/dev/null 2>&1; then
  rsync -a "${OLD_INPUTS}/" "${NEW_INPUTS}/"
else
  cp -a "${OLD_INPUTS}/." "${NEW_INPUTS}/"
fi

python3 - <<PY
from pathlib import Path

samples = "404 545006 596001 605000 606".split()
shared = Path("${SHARED_AUDIO_ROOT}")
new_inputs = Path("${NEW_INPUTS}")
old_root = "/home/jiaxingxu/rag-sst/eso-dataset/outputs_v2/test"

for sample in samples:
    dst = shared / f"{sample}_v2.wav"
    if not dst.is_file() or dst.stat().st_size == 0:
        raise SystemExit(f"[ERROR] shared audio missing: {dst}")

replacements = {
    f"{old_root}/sample_{sample}_v2/{sample}_v2.wav": str(shared / f"{sample}_v2.wav")
    for sample in samples
}

changed = 0
for path in new_inputs.iterdir():
    if path.suffix not in {".txt", ".yaml", ".json"}:
        continue
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new in replacements.items():
        text = text.replace(old, new)
    if text != original:
        path.write_text(text, encoding="utf-8")
        changed += 1

combined = new_inputs / "medicine.source__medicine5_hardraw.txt"
missing = []
for line in combined.read_text(encoding="utf-8").splitlines():
    if line.strip() and not Path(line.strip()).is_file():
        missing.append(line.strip())
if missing:
    raise SystemExit("[ERROR] rewritten source list has missing wavs: " + repr(missing[:5]))
print(f"[INFO] rewrote medicine input audio paths to {shared} in {changed} files")
PY

echo "[INFO] RUN_STAMP=${RUN_STAMP}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo "[INFO] LOG_ROOT=${LOG_ROOT}"
echo "[INFO] copied prepared inputs: ${OLD_INPUTS} -> ${NEW_INPUTS}"
echo "[INFO] shared audio root: ${SHARED_AUDIO_ROOT}"
echo "[INFO] max_new_tokens=80"

ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
CONDA_BASE="${CONDA_BASE:-/mnt/taurus/home/jiaxuanluo/miniconda3}" \
CONDA_ENV_NAME="${CONDA_ENV_NAME:-spaCyEnv}" \
RUN_STAMP="${RUN_STAMP}" \
LANG_CODE_OVERRIDE="zh" \
TARGET_LMS_OVERRIDE="1" \
TARGET_SAMPLES_OVERRIDE="404 545006 596001 605000 606" \
GPU_PAIR="${GPU_PAIR:-3,4}" \
RAG_GPU_OVERRIDE="${RAG_GPU_OVERRIDE:-cuda:1}" \
OUTPUT_BASE_OVERRIDE="${OUT_ROOT}" \
LOG_ROOT_OVERRIDE="${LOG_ROOT}" \
INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}" \
EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR}" \
DENSITY_TAG_OVERRIDE="medhard5_${MODEL_LABEL}_raw" \
COMBINED_PREFIX_OVERRIDE="medicine5_hardraw" \
COMBINED_GLOSSARY_TAG_OVERRIDE="hard_medicine_raw__medicine5" \
MAX_NEW_TOKENS_OVERRIDE="80" \
VLLM_TP_SIZE_OVERRIDE="${VLLM_TP_SIZE_OVERRIDE:-2}" \
GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}" \
FORCE_PREPARE_OVERRIDE="0" \
FORCE_RERUN_OVERRIDE="${FORCE_RERUN_OVERRIDE:-0}" \
bash "${BATCH_LAUNCHER}"

echo "[ALL DONE] medicine zh lm1 max80 RASST batch: ${OUT_ROOT}"
