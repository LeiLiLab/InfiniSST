set -euo pipefail

# ======Configuration=====
CONDA_BASE="/mnt/taurus/home/jiaxuanluo/miniconda3"
CONDA_ENV_NAME="spaCyEnv"
OUTPUT_ZH_BASE="/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_rank32_iter_0000452_hf_zh_bypass_slurm/zh"
DATA_ROOT="/mnt/taurus/data/siqiouyang/datasets/acl6060"
REF_FILE="${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.zh.txt"
AUDIO_YAML="${DATA_ROOT}/dev.yaml"
GLOSSARY_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary_with_translations.json"
FBK_FAIRSEQ_ROOT="/mnt/taurus/home/jiaxuanluo/FBK-fairseq"
MWERSEGMENTER_ROOT="/mnt/taurus/home/jiaxuanluo/mwerSegmenter"
SUMMARY_TSV="${OUTPUT_ZH_BASE}/rank32_iter_0000452_streamlaal_summary.tsv"
# =======================

# Activate env
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export MWERSEGMENTER_ROOT
export PATH="${MWERSEGMENTER_ROOT}:${PATH}"

if [[ ! -f "${FBK_FAIRSEQ_ROOT}/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py" ]]; then
  echo "[ERROR] stream_laal_term.py not found under FBK_FAIRSEQ_ROOT=${FBK_FAIRSEQ_ROOT}" >&2
  exit 2
fi

if [[ ! -f "${REF_FILE}" ]]; then
  echo "[ERROR] REF_FILE missing: ${REF_FILE}" >&2
  exit 2
fi
if [[ ! -f "${AUDIO_YAML}" ]]; then
  echo "[ERROR] AUDIO_YAML missing: ${AUDIO_YAML}" >&2
  exit 2
fi
if [[ ! -f "${GLOSSARY_PATH}" ]]; then
  echo "[ERROR] GLOSSARY_PATH missing: ${GLOSSARY_PATH}" >&2
  exit 2
fi

# Header
{
  echo -e "timestamp\tchunk_size\tBLEU\tStreamLAAL\tStreamLAAL_CA\tTERM_ACC\tTERM_CORRECT\tTERM_TOTAL\tRTF\toutput_path\tpost_eval_raw_json"
} > "${SUMMARY_TSV}"

for out in "${OUTPUT_ZH_BASE}"/iter_0000452-hf_cs*_hs0.48_rk5_vk10; do
  [[ -d "${out}" ]] || continue
  if [[ ! -f "${out}/instances.log" ]]; then
    echo "[WARN] Missing instances.log (skip): ${out}" >&2
    continue
  fi

  echo "[INFO] Post-eval: ${out}"

  EVAL_OUT="$(
    python "${FBK_FAIRSEQ_ROOT}/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py" \
      --simuleval-instances "${out}/instances.log" \
      --reference "${REF_FILE}" \
      --audio-yaml "${AUDIO_YAML}" \
      --sacrebleu-tokenizer zh \
      --latency-unit char \
      --glossary "${GLOSSARY_PATH}" \
      --term-lang zh \
      --term-mismatch-examples 0 2>&1
  )"

  echo "${EVAL_OUT}" > "${out}/post_eval.log"

  # JSON-escape raw output into a single TSV field
  EVAL_OUT_JSON="$(printf '%s' "${EVAL_OUT}" | python3 -c 'import sys, json; print(json.dumps(sys.stdin.read(), ensure_ascii=False))')"

  # Parse BLEU/StreamLAAL/StreamLAAL_CA from the first line with 3 numeric fields
  METRIC_LINE="$(
    echo "${EVAL_OUT}" | awk '
    function isnum(x){ return (x ~ /^[0-9]+(\.[0-9]+)?$/) }
    NF>=3 && isnum($1) && isnum($2) && isnum($3) { print $1"\t"$2"\t"$3; exit }
    '
  )"
  BLEU="$(echo "${METRIC_LINE}" | awk -F'\t' '{print $1}')"
  STREAM_LAAL="$(echo "${METRIC_LINE}" | awk -F'\t' '{print $2}')"
  STREAM_LAAL_CA="$(echo "${METRIC_LINE}" | awk -F'\t' '{print $3}')"

  # TERM_ACC line
  TERM_LINE="$(echo "${EVAL_OUT}" | awk '/^TERM_ACC[[:space:]]/{print; exit}')"
  TERM_ACC="$(echo "${TERM_LINE}" | awk '{print $2}')"
  TERM_CORRECT="$(echo "${TERM_LINE}" | awk '{print $4}')"
  TERM_TOTAL="$(echo "${TERM_LINE}" | awk '{print $6}')"

  # RTF from simuleval.log (if present)
  RTF_TOTAL="$(grep -oP 'rtf_total=\K[0-9.]+' "${out}/simuleval.log" 2>/dev/null | tail -n 1 || true)"

  # chunk_size from dirname (csX)
  CHUNK_SIZE="$(basename "${out}" | sed -n 's/.*_cs\([0-9.]*\)_.*/\1/p')"

  echo -e "$(date +'%Y-%m-%d %H:%M:%S')\t${CHUNK_SIZE}\t${BLEU}\t${STREAM_LAAL}\t${STREAM_LAAL_CA}\t${TERM_ACC}\t${TERM_CORRECT}\t${TERM_TOTAL}\t${RTF_TOTAL}\t${out}\t${EVAL_OUT_JSON}" >> "${SUMMARY_TSV}"

done

echo "[INFO] Summary written: ${SUMMARY_TSV}"
column -t -s $'\t' "${SUMMARY_TSV}" | head -n 20