#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
CONDA_PREFIX="${CONDA_PREFIX_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv}"
FBK_FAIRSEQ_ROOT="${FBK_FAIRSEQ_ROOT_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/FBK-fairseq}"
MWERSEGMENTER_ROOT="${MWERSEGMENTER_ROOT_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/mwerSegmenter}"
GLOSSARY="${GLOSSARY_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_hard_manual_glossary_streamlaal_20260524.json}"

LM2_BASE="${LM2_BASE_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260522}"
LM2_SETTING="${LM2_SETTING_OVERRIDE:-${LM2_BASE}/zh/gigaspeech-zh-s_origin-bsz4_gmedicine_gt571_abbrev_restored__medicine5_cs1.92_hs0.48_lm2_k210_k110_th0p0}"
LM2_COMBINED="${LM2_COMBINED_OVERRIDE:-${LM2_BASE}/zh/__medicine_inputs__/combined}"

LM4_4_BASE="${LM4_4_BASE_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260524_zh_lm4_no605000_aries23}"
LM4_4_SETTING="${LM4_4_SETTING_OVERRIDE:-${LM4_4_BASE}/zh/gigaspeech-zh-s_origin-bsz4_gstrict_fixed_medicine_glossary_abbrev_restored__medicine5_cs3.84_hs0.48_lm4_k210_k110_th0p0}"
LM4_4_COMBINED="${LM4_4_COMBINED_OVERRIDE:-${LM4_4_BASE}/zh/__medicine_inputs__/combined}"

LM4_605_BASE="${LM4_605_BASE_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_norag_vllm_override_probe_20260523/orig80}"
LM4_605_SETTING="${LM4_605_SETTING_OVERRIDE:-${LM4_605_BASE}/zh/gigaspeech-zh-s_origin-bsz4_gstrict_fixed_medicine_glossary_abbrev_restored__medicine5_cs3.84_hs0.48_lm4_k210_k110_th0p0}"
LM4_605_COMBINED="${LM4_605_COMBINED_OVERRIDE:-${LM4_605_BASE}/zh/__medicine_inputs__/combined}"

LM4_FULL_BASE="${LM4_FULL_BASE_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260524_zh_lm4_with605000_from_taurus_orig80}"
LM4_FULL_SETTING="${LM4_FULL_SETTING_OVERRIDE:-${LM4_FULL_BASE}/zh/gigaspeech-zh-s_origin-bsz4_gstrict_fixed_medicine_glossary_abbrev_restored__medicine5_cs3.84_hs0.48_lm4_k210_k110_th0p0}"
LM4_FULL_COMBINED="${LM4_FULL_COMBINED_OVERRIDE:-${LM4_FULL_BASE}/zh/__medicine_inputs__/combined}"

export CONDA_PREFIX
export FBK_FAIRSEQ_ROOT
export MWERSEGMENTER_ROOT
export STREAM_LAAL_TOOL="${FBK_FAIRSEQ_ROOT}/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py"
export PATH="${CONDA_PREFIX}/bin:${MWERSEGMENTER_ROOT}:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

require_file() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    echo "[ERROR] missing required file: ${path}" >&2
    exit 3
  fi
}

run_post_eval_one() {
  local label="$1"
  local lm="$2"
  local setting="$3"
  local combined="$4"
  local miss_suffix="$5"

  require_file "${setting}/instances.log"
  require_file "${combined}/medicine5.source_text.en.sentences.txt"
  require_file "${combined}/medicine5.ref.zh.sentences.txt"
  require_file "${combined}/medicine5.audio.yaml"
  require_file "${GLOSSARY}"

  local eval_tsv="${setting}/eval_results_streamlaal_term.hard_llm_manual_check.tsv"
  local eval_log="${setting}/post_eval_streamlaal_term.hard_llm_manual_check.log"
  local work_dir="${setting}/work_streamlaal_term.hard_llm_manual_check"
  local miss_tsv="${setting}/term_misses.hard_llm_manual_check.zh_lm${miss_suffix}.tsv"
  local miss_summary="${setting}/term_miss_summary.hard_llm_manual_check.zh_lm${miss_suffix}.tsv"
  local norm_glossary="${setting}/hard_medicine_glossary.streamlaal_dict.hard_llm_manual_check.json"

  echo "[POST START] ${label} setting=${setting}"
  cd "${ROOT_DIR}"

  if [[ ! -s "${eval_tsv}" ]]; then
    "${CONDA_PREFIX}/bin/python" documents/code/offline_sst_eval/offline_streamlaal_eval.py \
      --mode acl6060 \
      --instances-log "${setting}/instances.log" \
      --lang-code zh \
      --source-file "${combined}/medicine5.source_text.en.sentences.txt" \
      --ref-file "${combined}/medicine5.ref.zh.sentences.txt" \
      --audio-yaml "${combined}/medicine5.audio.yaml" \
      --glossary-acl6060 "${GLOSSARY}" \
      --fbk-fairseq-root "${FBK_FAIRSEQ_ROOT}" \
      --term-fcr-policy source_ref_negative_sentence \
      --output-tsv "${eval_tsv}" \
      --output-log "${eval_log}" \
      --work-dir "${work_dir}" \
      --term-mismatch-examples 20
  else
    echo "[SKIP] eval exists: ${eval_tsv}"
  fi

  if [[ ! -s "${miss_tsv}" || ! -s "${miss_summary}" ]]; then
    "${CONDA_PREFIX}/bin/python" documents/code/simuleval/export_streamlaal_term_misses.py \
      --instances-log "${setting}/instances.log" \
      --reference "${combined}/medicine5.ref.zh.sentences.txt" \
      --source-reference "${combined}/medicine5.source_text.en.sentences.txt" \
      --audio-yaml "${combined}/medicine5.audio.yaml" \
      --glossary "${GLOSSARY}" \
      --lang-code zh \
      --stream-laal-tool "${STREAM_LAAL_TOOL}" \
      --mwersegmenter-root "${MWERSEGMENTER_ROOT}" \
      --output-misses "${miss_tsv}" \
      --output-summary "${miss_summary}" \
      --output-normalized-glossary "${norm_glossary}"
  else
    echo "[SKIP] misses exist: ${miss_tsv}"
  fi

  echo "[POST DONE] ${label}"
}

build_lm4_full_combined() {
  require_file "${LM4_4_SETTING}/instances.log"
  require_file "${LM4_605_SETTING}/instances.log"
  require_file "${LM4_4_COMBINED}/medicine5.source_text.en.sentences.txt"
  require_file "${LM4_4_COMBINED}/medicine5.ref.zh.sentences.txt"
  require_file "${LM4_4_COMBINED}/medicine5.audio.yaml"
  require_file "${LM4_605_COMBINED}/medicine5.source_text.en.sentences.txt"
  require_file "${LM4_605_COMBINED}/medicine5.ref.zh.sentences.txt"
  require_file "${LM4_605_COMBINED}/medicine5.audio.yaml"

  mkdir -p "${LM4_FULL_SETTING}" "${LM4_FULL_COMBINED}"
  python - "$LM4_4_SETTING" "$LM4_605_SETTING" "$LM4_4_COMBINED" "$LM4_605_COMBINED" "$LM4_FULL_SETTING" "$LM4_FULL_COMBINED" <<'PY'
import json
import shutil
import sys
from pathlib import Path

lm4_4_setting, lm4_605_setting, comb4, comb605, out_setting, out_comb = map(Path, sys.argv[1:])
out_setting.mkdir(parents=True, exist_ok=True)
out_comb.mkdir(parents=True, exist_ok=True)

with (out_setting / "instances.log").open("w", encoding="utf-8") as out:
    next_idx = 0
    for src in [lm4_4_setting / "instances.log", lm4_605_setting / "instances.log"]:
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                obj["index"] = next_idx
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                next_idx += 1

for name in [
    "medicine5.source_text.en.sentences.txt",
    "medicine5.ref.zh.sentences.txt",
    "medicine5.audio.yaml",
    "medicine5.source.txt",
    "medicine5.target.zh.txt",
]:
    with (out_comb / name).open("w", encoding="utf-8") as out:
        for src_dir in [comb4, comb605]:
            path = src_dir / name
            if path.is_file():
                text = path.read_text(encoding="utf-8")
                out.write(text)
                if text and not text.endswith("\n"):
                    out.write("\n")

sample_map = out_comb / "medicine5.sample_map.tsv"
with sample_map.open("w", encoding="utf-8") as out:
    out.write("instance_index\tsample_id\tsource\n")
    for idx, sample in enumerate(["404", "545006", "596001", "606", "605000"]):
        src = "aries_lm4_no605000" if sample != "605000" else "taurus_orig80"
        out.write(f"{idx}\t{sample}\t{src}\n")

manifest = {
    "description": "Combined lm4 no-RAG zh instances: Aries four-sample run plus Taurus orig80 sample605000.",
    "instances": str(out_setting / "instances.log"),
    "combined_input_dir": str(out_comb),
    "components": [
        {"samples": ["404", "545006", "596001", "606"], "setting": str(lm4_4_setting)},
        {"samples": ["605000"], "setting": str(lm4_605_setting), "variant": "taurus_orig80"},
    ],
}
(out_comb / "medicine5.combined_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

main() {
  echo "[START] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  run_post_eval_one "zh_lm2_hard_manual" "2" "${LM2_SETTING}" "${LM2_COMBINED}" "2"
  run_post_eval_one "zh_lm4_sample605000_taurus_orig80_hard_manual" "4" "${LM4_605_SETTING}" "${LM4_605_COMBINED}" "4_sample605000"
  build_lm4_full_combined
  run_post_eval_one "zh_lm4_full5_aries4_plus_taurus605000_hard_manual" "4" "${LM4_FULL_SETTING}" "${LM4_FULL_COMBINED}" "4_full5"
  echo "[DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

main "$@"
