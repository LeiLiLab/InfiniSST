#!/usr/bin/env bash
set -euo pipefail

# Post-hoc CPU/offline rescore for PSC zh tagged ACL gs1k/gs10k outputs.
# It does not rerun generation and does not overwrite original eval_results.tsv.

PSC_BASE="${PSC_BASE:-/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval}"
ROOT_DIR="${ROOT_DIR:-${PSC_BASE}/src/InfiniSST}"
ENV_DIR="${ENV_DIR:-${PSC_BASE}/envs/spaCyEnv_20260518}"
RUN_STAMP_BASE="${RUN_STAMP_BASE:-20260524T0520_psc_tagacl_newv9_hn1024_tau078_gs1k_gs10k_fixedraw_zh}"
MODEL_LABEL="${MODEL_LABEL:-new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078}"
OUT_ROOT="${OUT_ROOT:-${PSC_BASE}/outputs/tagged_acl_new_v9_hn1024_tau078_gs_fixedraw/${RUN_STAMP_BASE}}"
SUMMARY_DIR="${SUMMARY_DIR:-${OUT_ROOT}/__summary__}"
RAW_GLOSSARY="${RAW_GLOSSARY:-${PSC_BASE}/glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
FBK_FAIRSEQ_ROOT="${FBK_FAIRSEQ_ROOT:-${PSC_BASE}/tools/FBK-fairseq}"
MWERSEGMENTER_ROOT="${MWERSEGMENTER_ROOT:-${PSC_BASE}/tools/mwerSegmenter}"
STREAM_LAAL_TOOL_REL="${STREAM_LAAL_TOOL_REL:-examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py}"
LANG_CODE="${LANG_CODE:-zh}"
FORCE="${FORCE:-0}"

mkdir -p "${SUMMARY_DIR}"

export PATH="${ENV_DIR}/bin:${MWERSEGMENTER_ROOT}:${PATH}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export INFINISST_ROOT="${ROOT_DIR}"
export FBK_FAIRSEQ_ROOT
export MWERSEGMENTER_ROOT
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Missing required path: ${path}" >&2
    exit 3
  fi
}

for p in \
  "${ROOT_DIR}/documents/code/offline_sst_eval/offline_streamlaal_eval.py" \
  "${ENV_DIR}/bin/python" \
  "${RAW_GLOSSARY}" \
  "${FBK_FAIRSEQ_ROOT}/${STREAM_LAAL_TOOL_REL}" \
  "${MWERSEGMENTER_ROOT}" \
  "${OUT_ROOT}"; do
  require_path "${p}"
done

tsv_for_setting() {
  local kind="$1" lm="$2"
  local glossary_tag
  case "${kind}" in
    gs1k) glossary_tag="acl6060_tagged_gt_union_gs1000_min_norm2_backfill" ;;
    gs10k) glossary_tag="acl6060_tagged_gt_union_gs10000_min_norm2_backfill" ;;
    *) echo "[ERROR] unsupported kind=${kind}" >&2; return 2 ;;
  esac
  printf '%s/%s/lm%s/%s/%s/dtagacl_new_v9_hn1024_tau078_gs_fixedraw_lm%s_k10_th0.78_g%s' \
    "${OUT_ROOT}" "${kind}" "${lm}" "${MODEL_LABEL}" "${LANG_CODE}" "${lm}" "${glossary_tag}"
}

input_dir_for_setting() {
  local kind="$1" lm="$2"
  printf '%s/%s/lm%s/__inputs__/%s/full_all/%s/all' \
    "${OUT_ROOT}" "${kind}" "${lm}" "${MODEL_LABEL}" "${LANG_CODE}"
}

run_one() {
  local kind="$1" lm="$2"
  local out_dir in_dir out_tsv out_log
  out_dir="$(tsv_for_setting "${kind}" "${lm}")"
  in_dir="$(input_dir_for_setting "${kind}" "${lm}")"
  out_tsv="${out_dir}/eval_results.strip_term_recheck.tsv"
  out_log="${out_dir}/eval_results.strip_term_recheck.log"

  for p in \
    "${out_dir}/instances.log" \
    "${in_dir}/ref.txt" \
    "${in_dir}/source_text.txt" \
    "${in_dir}/audio.yaml"; do
    require_path "${p}"
  done

  if [[ "${FORCE}" != "1" && -s "${out_tsv}" ]]; then
    echo "[SKIP] ${kind} lm=${lm}: ${out_tsv}"
    return 0
  fi

  echo "[RUN] strip recheck kind=${kind} lm=${lm}"
  "${ENV_DIR}/bin/python" "${ROOT_DIR}/documents/code/offline_sst_eval/offline_streamlaal_eval.py" \
    --mode acl6060 \
    --instances-log "${out_dir}/instances.log" \
    --lang-code "${LANG_CODE}" \
    --ref-file "${in_dir}/ref.txt" \
    --source-file "${in_dir}/source_text.txt" \
    --audio-yaml "${in_dir}/audio.yaml" \
    --glossary-acl6060 "${RAW_GLOSSARY}" \
    --fbk-fairseq-root "${FBK_FAIRSEQ_ROOT}" \
    --stream-laal-tool-rel "${STREAM_LAAL_TOOL_REL}" \
    --strip-output-tags term \
    --term-fcr-policy term_map_source_ref_negative_sentence \
    --output-tsv "${out_tsv}" \
    --output-log "${out_log}"
}

for kind in gs1k gs10k; do
  for lm in 1 2 3 4; do
    run_one "${kind}" "${lm}"
  done
done

"${ENV_DIR}/bin/python" - "${OUT_ROOT}" "${SUMMARY_DIR}/strip_recheck_summary.tsv" <<'PY'
import csv
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
rows = []
for kind, tag in [
    ("gs1k", "acl6060_tagged_gt_union_gs1000_min_norm2_backfill"),
    ("gs10k", "acl6060_tagged_gt_union_gs10000_min_norm2_backfill"),
]:
    for lm in [1, 2, 3, 4]:
        tsv = next((out_root / kind / f"lm{lm}").glob(
            f"*/zh/dtagacl_new_v9_hn1024_tau078_gs_fixedraw_lm{lm}_k10_th0.78_g{tag}/eval_results.strip_term_recheck.tsv"
        ), None)
        if tsv is None or not tsv.is_file():
            raise SystemExit(f"missing strip recheck TSV for {kind} lm{lm}")
        with tsv.open("r", encoding="utf-8", newline="") as f:
            data = list(csv.DictReader(f, delimiter="\t"))
        if len(data) != 1:
            raise SystemExit(f"expected one row in {tsv}, got {len(data)}")
        row = dict(data[0])
        row.update({"runtime_glossary": kind, "lm": str(lm), "strip_recheck_tsv": str(tsv)})
        rows.append(row)

fieldnames = [
    "runtime_glossary", "lm", "mode", "lang_code", "BLEU", "StreamLAAL",
    "StreamLAAL_CA", "TERM_ACC", "TERM_CORRECT", "TERM_TOTAL",
    "REAL_TERM_ADOPT", "REAL_TERM_ADOPTED", "REAL_TERM_ADOPT_TOTAL",
    "TERM_FCR", "FALSE_COPY", "NEG_TOTAL", "SOURCE_TERM_SENT_FCR",
    "SOURCE_FALSE_COPY", "SOURCE_NEG_TOTAL", "instances_log",
    "strip_recheck_tsv",
]
summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(f"[SUMMARY] wrote {summary_path}")
PY

echo "[DONE] strip recheck summary: ${SUMMARY_DIR}/strip_recheck_summary.tsv"
