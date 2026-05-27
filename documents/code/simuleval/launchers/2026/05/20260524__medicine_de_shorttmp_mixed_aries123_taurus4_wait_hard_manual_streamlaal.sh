#!/usr/bin/env bash
set -euo pipefail

CONDA_PREFIX="${CONDA_PREFIX_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv}"
FBK_FAIRSEQ_ROOT="${FBK_FAIRSEQ_ROOT_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/FBK-fairseq}"
MWERSEGMENTER_ROOT="${MWERSEGMENTER_ROOT_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/mwerSegmenter}"
ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
GLOSSARY="${GLOSSARY_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_hard_manual_glossary_streamlaal_20260524.json}"
WAIT_SECONDS="${WAIT_SECONDS_OVERRIDE:-60}"
MAX_POLLS="${MAX_POLLS_OVERRIDE:-960}"

export CONDA_PREFIX
export FBK_FAIRSEQ_ROOT
export MWERSEGMENTER_ROOT
export STREAM_LAAL_TOOL="${FBK_FAIRSEQ_ROOT}/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py"
export PATH="${CONDA_PREFIX}/bin:${MWERSEGMENTER_ROOT}:${PATH}"
export TMPDIR="${TMPDIR_OVERRIDE:-/dev/shm/jxdepost}"
mkdir -p "${TMPDIR}"

run_post_one() {
  local lm="$1"
  local label="$2"
  local cs="$3"
  local out="$4"
  local setting="${out}/de/gigaspeech-de-s_origin-bsz4_gstrict_fixed_medicine_glossary_abbrev_restored__medicine5_cs${cs}_hs0.48_lm${lm}_k210_k110_th0p0"
  local combined="${out}/de/__medicine_inputs__/combined"

  for _ in $(seq 1 "${MAX_POLLS}"); do
    local ts
    local inst
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    inst=missing
    [[ -f "${setting}/instances.log" ]] && inst=$(wc -l < "${setting}/instances.log")
    echo "[${ts}] wait de lm=${lm} label=${label} timing instances=${inst}"
    if [[ -s "${out}/timing.tsv" ]] && awk -F '\t' 'NR>1 && $5 == "success" {found=1} END {exit found?0:1}' "${out}/timing.tsv"; then
      echo "[${ts}] de lm=${lm} success detected"
      break
    fi
    if [[ -s "${out}/timing.tsv" ]] && awk -F '\t' 'NR>1 && $5 == "failed" {found=1} END {exit found?0:1}' "${out}/timing.tsv"; then
      echo "[ERROR] de lm=${lm} failed; not running post-eval: ${out}/timing.tsv" >&2
      return 2
    fi
    sleep "${WAIT_SECONDS}"
  done

  if [[ ! -s "${out}/timing.tsv" ]] || ! awk -F '\t' 'NR>1 && $5 == "success" {found=1} END {exit found?0:1}' "${out}/timing.tsv"; then
    echo "[ERROR] de lm=${lm} did not finish successfully before timeout" >&2
    return 2
  fi

  if [[ -s "${setting}/eval_results_streamlaal_term.hard_llm_manual_check.tsv" ]]; then
    echo "[INFO] de lm=${lm} hard-term eval already exists; skip"
    return 0
  fi

  cd "${ROOT_DIR}"
  "${CONDA_PREFIX}/bin/python" documents/code/offline_sst_eval/offline_streamlaal_eval.py \
    --mode acl6060 \
    --instances-log "${setting}/instances.log" \
    --lang-code de \
    --source-file "${combined}/medicine5.source_text.en.sentences.txt" \
    --ref-file "${combined}/medicine5.ref.de.sentences.txt" \
    --audio-yaml "${combined}/medicine5.audio.yaml" \
    --glossary-acl6060 "${GLOSSARY}" \
    --fbk-fairseq-root "${FBK_FAIRSEQ_ROOT}" \
    --term-fcr-policy source_ref_negative_sentence \
    --output-tsv "${setting}/eval_results_streamlaal_term.hard_llm_manual_check.tsv" \
    --output-log "${setting}/post_eval_streamlaal_term.hard_llm_manual_check.log" \
    --work-dir "${setting}/work_streamlaal_term.hard_llm_manual_check" \
    --term-mismatch-examples 20

  "${CONDA_PREFIX}/bin/python" documents/code/simuleval/export_streamlaal_term_misses.py \
    --instances-log "${setting}/instances.log" \
    --reference "${combined}/medicine5.ref.de.sentences.txt" \
    --source-reference "${combined}/medicine5.source_text.en.sentences.txt" \
    --audio-yaml "${combined}/medicine5.audio.yaml" \
    --glossary "${GLOSSARY}" \
    --lang-code de \
    --stream-laal-tool "${STREAM_LAAL_TOOL}" \
    --mwersegmenter-root "${MWERSEGMENTER_ROOT}" \
    --output-misses "${setting}/term_misses.hard_llm_manual_check.de_lm${lm}.tsv" \
    --output-summary "${setting}/term_miss_summary.hard_llm_manual_check.de_lm${lm}.tsv" \
    --output-normalized-glossary "${setting}/hard_medicine_glossary.streamlaal_dict.hard_llm_manual_check.json"

  echo "[DONE] de lm=${lm} hard-term post-eval"
}

run_post_one 1 aries23 0.96 /mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260524_de_rerun_shorttmp_lm1_aries23 &
run_post_one 2 aries45 1.92 /mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260524_de_rerun_shorttmp_lm2_aries45 &
run_post_one 3 aries67 2.88 /mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260524_de_rerun_shorttmp_lm3_aries67 &
run_post_one 4 taurus45 3.84 /mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260524_de_rerun_shorttmp_lm4_taurus45 &
wait
