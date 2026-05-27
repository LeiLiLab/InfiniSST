#!/usr/bin/env bash
set -euo pipefail

# Compute per-paper TERM_ACC from a SINGLE full-run instances.log:
# - Run SimulEval once on the full SRC/TGT lists (stable, deterministic inputs).
# - Post-eval: split the full instances.log by paper wav name and compute TERM_ACC using
#   the per-paper extracted glossary (or any glossary path provided in the mapping).
#
# This avoids re-running SimulEval per paper while still allowing per-paper term accuracy.
#
# All user-facing strings are in English.

# ======Configuration=====
# Exit codes
EXIT_CONFIG_ERROR="2"
EXIT_DATA_ERROR="3"

# Repo
ROOT_DIR="/home/jiaxuanluo/InfiniSST"

# Full-run outputs (either set OUTPUT_DIR_OVERRIDE directly or let the script locate it)
OUTPUT_BASE=""
LANG_CODE="de"
MODEL_NAME_OVERRIDE=""
LATENCY_MULTIPLIER="4"
RAG_K2="10"
RAG_K1_FIXED="10"
RAG_HOP_SIZE="0.48"
RAG_CONFIDENCE_THRESHOLD="0.0"
OUTPUT_DIR_OVERRIDE="${OUTPUT_DIR_OVERRIDE:-}"

# Instances log filename under output dir
INSTANCES_FILE_NAME="instances.log"

# Mapping JSON (from prepare_extracted_glossary_by_paper_inputs.py)
# It provides paper_id -> glossary_path.
PAPER_INPUTS_MAP_JSON="${PAPER_INPUTS_MAP_JSON:-}"
PAPER_IDS_OVERRIDE="${PAPER_IDS_OVERRIDE:-}"

# Dataset / references
DATA_ROOT="/mnt/taurus/data/siqiouyang/datasets/acl6060"
AUDIO_YAML_FULL="${DATA_ROOT}/dev.yaml"
REF_FULL="${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.${LANG_CODE}.txt"

# StreamLAAL tool
FBK_FAIRSEQ_ROOT="/mnt/taurus/home/jiaxuanluo/FBK-fairseq"
STREAM_LAAL_TOOL_REL="examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py"
TERM_MISMATCH_EXAMPLES="0"

# Tokenizer/latency unit by language (baseline-aligned)
SACREBLEU_TOKENIZER="13a"
LATENCY_UNIT="word"
TERM_LANG="${LANG_CODE}"

# Optional: compute overall TERM_ACC too (if set).
OVERALL_GLOSSARY_PATH_OVERRIDE="${OVERALL_GLOSSARY_PATH_OVERRIDE:-}"

# Conda env
CONDA_BASE="/mnt/taurus/home/jiaxuanluo/miniconda3"
CONDA_ENV_NAME="spaCyEnv"

# MWER segmenter
MWERSEGMENTER_ROOT="/mnt/taurus/home/jiaxuanluo/mwerSegmenter"
# ======Configuration=====

if [[ -z "${OUTPUT_DIR_OVERRIDE}" ]]; then
  if [[ -z "${OUTPUT_BASE}" ]]; then
    echo "[ERROR] Set OUTPUT_DIR_OVERRIDE or OUTPUT_BASE." >&2
    exit "${EXIT_CONFIG_ERROR}"
  fi
  if [[ -z "${MODEL_NAME_OVERRIDE}" ]]; then
    echo "[ERROR] When using OUTPUT_BASE, set MODEL_NAME_OVERRIDE to locate output dir." >&2
    exit "${EXIT_CONFIG_ERROR}"
  fi
  MODEL_SHORT="$(basename "${MODEL_NAME_OVERRIDE}")"
  THRESHOLD_TAG="${RAG_CONFIDENCE_THRESHOLD//./p}"
  # Best-effort locate output dir (rank16 naming)
  set +e
  OUTPUT_DIR_OVERRIDE="$(
    ls -dt \
      "${OUTPUT_BASE}/${LANG_CODE}/${MODEL_SHORT}"*_hs${RAG_HOP_SIZE}_lm${LATENCY_MULTIPLIER}_k2${RAG_K2}_k1${RAG_K1_FIXED}_th${THRESHOLD_TAG} \
      2>/dev/null | head -n 1
  )"
  set -e
  if [[ -z "${OUTPUT_DIR_OVERRIDE}" ]]; then
    echo "[ERROR] Failed to locate output dir under OUTPUT_BASE=${OUTPUT_BASE}" >&2
    exit "${EXIT_DATA_ERROR}"
  fi
fi

OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE}"
INSTANCES_PATH="${OUTPUT_DIR}/${INSTANCES_FILE_NAME}"

if [[ ! -f "${INSTANCES_PATH}" ]] || [[ ! -s "${INSTANCES_PATH}" ]]; then
  echo "[ERROR] Missing/empty instances log: ${INSTANCES_PATH}" >&2
  exit "${EXIT_DATA_ERROR}"
fi

if [[ -z "${PAPER_INPUTS_MAP_JSON}" ]] || [[ ! -f "${PAPER_INPUTS_MAP_JSON}" ]]; then
  echo "[ERROR] PAPER_INPUTS_MAP_JSON is required and must exist (got: ${PAPER_INPUTS_MAP_JSON:-<empty>})" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi

STREAM_LAAL_TOOL="${FBK_FAIRSEQ_ROOT}/${STREAM_LAAL_TOOL_REL}"
if [[ ! -f "${STREAM_LAAL_TOOL}" ]]; then
  echo "[ERROR] stream_laal_term.py not found: ${STREAM_LAAL_TOOL}" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi
if [[ ! -f "${AUDIO_YAML_FULL}" ]]; then
  echo "[ERROR] AUDIO_YAML_FULL not found: ${AUDIO_YAML_FULL}" >&2
  exit "${EXIT_DATA_ERROR}"
fi
if [[ ! -f "${REF_FULL}" ]]; then
  echo "[ERROR] REF_FULL not found: ${REF_FULL}" >&2
  exit "${EXIT_DATA_ERROR}"
fi
if [[ -n "${OVERALL_GLOSSARY_PATH_OVERRIDE}" ]] && [[ ! -f "${OVERALL_GLOSSARY_PATH_OVERRIDE}" ]]; then
  echo "[ERROR] OVERALL_GLOSSARY_PATH_OVERRIDE not found: ${OVERALL_GLOSSARY_PATH_OVERRIDE}" >&2
  exit "${EXIT_DATA_ERROR}"
fi

# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
echo "[INFO] Activated conda env: ${CONDA_ENV_NAME}"

export MWERSEGMENTER_ROOT
export PATH="${MWERSEGMENTER_ROOT}:${PATH}"

PAPERS="$(
python3 - <<PY
import json
mp="${PAPER_INPUTS_MAP_JSON}"
with open(mp,"r",encoding="utf-8") as f:
    obj=json.load(f)
print(" ".join(sorted(obj.get("papers", {}).keys())))
PY
)"
if [[ -z "${PAPERS}" ]]; then
  echo "[ERROR] No papers found in mapping: ${PAPER_INPUTS_MAP_JSON}" >&2
  exit "${EXIT_DATA_ERROR}"
fi
if [[ -n "${PAPER_IDS_OVERRIDE}" ]]; then
  PAPERS="${PAPER_IDS_OVERRIDE}"
fi

echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] INSTANCES_PATH=${INSTANCES_PATH}"
echo "[INFO] PAPER_INPUTS_MAP_JSON=${PAPER_INPUTS_MAP_JSON}"
echo "[INFO] Papers: ${PAPERS}"

# Overall metrics from the full instances.log.
echo "[INFO] ============================================================"
echo "[INFO] Overall metrics (full instances.log)"
set +e
if [[ -n "${OVERALL_GLOSSARY_PATH_OVERRIDE}" ]]; then
  OVERALL_OUT="$(python "${STREAM_LAAL_TOOL}" \
    --simuleval-instances "${INSTANCES_PATH}" \
    --reference "${REF_FULL}" \
    --audio-yaml "${AUDIO_YAML_FULL}" \
    --sacrebleu-tokenizer "${SACREBLEU_TOKENIZER}" \
    --latency-unit "${LATENCY_UNIT}" \
    --glossary "${OVERALL_GLOSSARY_PATH_OVERRIDE}" \
    --term-lang "${TERM_LANG}" \
    --term-mismatch-examples "${TERM_MISMATCH_EXAMPLES}" 2>&1)"
else
  OVERALL_OUT="$(python "${STREAM_LAAL_TOOL}" \
    --simuleval-instances "${INSTANCES_PATH}" \
    --reference "${REF_FULL}" \
    --audio-yaml "${AUDIO_YAML_FULL}" \
    --sacrebleu-tokenizer "${SACREBLEU_TOKENIZER}" \
    --latency-unit "${LATENCY_UNIT}" 2>&1)"
fi
OVERALL_RC="$?"
set -e
if [[ "${OVERALL_RC}" != "0" ]]; then
  echo "[WARN] Overall stream_laal_term.py failed (rc=${OVERALL_RC})" >&2
  echo "${OVERALL_OUT}" >&2
else
  OVERALL_VALUES_LINE="$(echo "${OVERALL_OUT}" | awk 'BEGIN{found=0} /^BLEU[[:space:]]/{found=1; next} found {print; exit}')"
  OVERALL_BLEU="$(echo "${OVERALL_VALUES_LINE}" | awk '{print $1}')"
  OVERALL_STREAM_LAAL="$(echo "${OVERALL_VALUES_LINE}" | awk '{print $2}')"
  OVERALL_STREAM_LAAL_CA="$(echo "${OVERALL_VALUES_LINE}" | awk '{print $3}')"
  echo "[INFO] Overall BLEU/StreamLAAL/StreamLAAL_CA = ${OVERALL_BLEU}\t${OVERALL_STREAM_LAAL}\t${OVERALL_STREAM_LAAL_CA}"
  if [[ -n "${OVERALL_GLOSSARY_PATH_OVERRIDE}" ]]; then
    OVERALL_TERM_LINE="$(echo "${OVERALL_OUT}" | awk '/^TERM_ACC[[:space:]]/{print; exit}')"
    if [[ -n "${OVERALL_TERM_LINE}" ]]; then
      echo "[INFO] ${OVERALL_TERM_LINE}"
    fi
  fi
fi

echo "[INFO] ============================================================"
echo "[INFO] Per-paper TERM_ACC (from full instances.log)"
printf '%s\t%s\t%s\t%s\t%s\n' "paper_id" "TERM_ACC" "TERM_CORRECT" "TERM_TOTAL" "glossary_path"

for PAPER_ID in ${PAPERS}; do
  GLOSSARY_PATH="$(
  python3 - <<PY
import json
pid="${PAPER_ID}"
mp="${PAPER_INPUTS_MAP_JSON}"
with open(mp,"r",encoding="utf-8") as f:
    obj=json.load(f)
print(obj.get("papers", {}).get(pid, {}).get("glossary_path", ""))
PY
  )"
  if [[ -z "${GLOSSARY_PATH}" ]] || [[ ! -f "${GLOSSARY_PATH}" ]]; then
    echo "[WARN] Skip paper_id=${PAPER_ID}: glossary_path missing: ${GLOSSARY_PATH}" >&2
    continue
  fi

  PAPER_WAV="${PAPER_ID}.wav"
  TMP_INSTANCES="$(mktemp -t instances_${PAPER_ID}_XXXXXX.log)"
  TMP_AUDIO="$(mktemp -t audio_${PAPER_ID}_XXXXXX.yaml)"
  TMP_REF="$(mktemp -t ref_${PAPER_ID}_XXXXXX.txt)"

  # Filter instances.log to this paper wav (by source[0] path).
  python3 - <<PY
import json
from pathlib import Path

src = Path("${INSTANCES_PATH}")
dst = Path("${TMP_INSTANCES}")
paper_wav = "${PAPER_WAV}"

out_lines = []
with src.open("r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line=line.strip()
        if not line:
            continue
        try:
            obj=json.loads(line)
        except Exception:
            continue
        s=obj.get("source") or []
        s0=s[0] if isinstance(s,list) and s else ""
        if paper_wav in str(s0):
            out_lines.append(json.dumps(obj, ensure_ascii=False))

dst.write_text("\\n".join(out_lines) + ("\\n" if out_lines else ""), encoding="utf-8")
PY

  if [[ ! -s "${TMP_INSTANCES}" ]]; then
    echo "[WARN] Skip paper_id=${PAPER_ID}: no instances found in full log for ${PAPER_WAV}" >&2
    rm -f "${TMP_INSTANCES}" "${TMP_AUDIO}" "${TMP_REF}"
    continue
  fi

  # Build per-paper audio/ref aligned with full dev.yaml and full reference.
  python3 - <<PY
from pathlib import Path
import yaml

audio_yaml = Path("${AUDIO_YAML_FULL}")
ref_full = Path("${REF_FULL}")
paper_wav = "${PAPER_WAV}"

data = yaml.safe_load(audio_yaml.read_text(encoding="utf-8", errors="replace"))
refs = ref_full.read_text(encoding="utf-8", errors="replace").splitlines()

if len(data) != len(refs):
    raise RuntimeError(f"dev.yaml entries {len(data)} != reference lines {len(refs)}")

indices = [i for i, x in enumerate(data) if isinstance(x, dict) and x.get("wav") == paper_wav]
subset_audio = [data[i] for i in indices]
subset_refs = [refs[i] for i in indices]

Path("${TMP_AUDIO}").write_text(yaml.safe_dump(subset_audio, allow_unicode=True), encoding="utf-8")
Path("${TMP_REF}").write_text("\\n".join(subset_refs) + "\\n", encoding="utf-8")
PY

  set +e
  EVAL_OUT="$(python "${STREAM_LAAL_TOOL}" \
    --simuleval-instances "${TMP_INSTANCES}" \
    --reference "${TMP_REF}" \
    --audio-yaml "${TMP_AUDIO}" \
    --sacrebleu-tokenizer "${SACREBLEU_TOKENIZER}" \
    --latency-unit "${LATENCY_UNIT}" \
    --glossary "${GLOSSARY_PATH}" \
    --term-lang "${TERM_LANG}" \
    --term-mismatch-examples "${TERM_MISMATCH_EXAMPLES}" 2>&1)"
  EVAL_RC="$?"
  set -e

  if [[ "${EVAL_RC}" != "0" ]]; then
    echo "[WARN] stream_laal_term.py failed for paper_id=${PAPER_ID} (rc=${EVAL_RC})" >&2
    echo "${EVAL_OUT}" >&2
    rm -f "${TMP_INSTANCES}" "${TMP_AUDIO}" "${TMP_REF}"
    continue
  fi

  TERM_LINE="$(echo "${EVAL_OUT}" | awk '/^TERM_ACC[[:space:]]/{print; exit}')"
  TERM_ACC="$(echo "${TERM_LINE}" | awk '{print $2}')"
  TERM_CORRECT="$(echo "${TERM_LINE}" | awk '{print $4}')"
  TERM_TOTAL="$(echo "${TERM_LINE}" | awk '{print $6}')"

  printf '%s\t%s\t%s\t%s\t%s\n' "${PAPER_ID}" "${TERM_ACC}" "${TERM_CORRECT}" "${TERM_TOTAL}" "${GLOSSARY_PATH}"

  rm -f "${TMP_INSTANCES}" "${TMP_AUDIO}" "${TMP_REF}"
done

