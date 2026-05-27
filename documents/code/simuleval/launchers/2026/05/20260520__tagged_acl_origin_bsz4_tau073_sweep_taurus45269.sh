#!/usr/bin/env bash
set -euo pipefail

# Tagged ACL baseline sweep for existing streaming SLM checkpoints.
#
# Modes:
#   MODE=prepare  create derived glossaries and per-paper inputs only
#   MODE=smoke    run one small clipped sentence by default
#   MODE=full     run lang x lm x glossary settings, max 4 workers by default
#
# Each setting worker uses one 2-GPU pair and runs all selected ACL talks
# sequentially, then writes one aggregate eval_results.tsv for WandB logging.

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
LOCAL_ROOT_DIR="${LOCAL_ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
EVAL_SCRIPT="${ROOT_DIR}/documents/code/simuleval/eval_density_unified.sh"
LAUNCHER_SELF="${LAUNCHER_SELF:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260520__tagged_acl_origin_bsz4_tau073_sweep_taurus45269.sh}"
WANDB_LOGGER="${ROOT_DIR}/documents/code/offline_evaluation/wandb_eval_logger.py"
WANDB_PYTHON="${WANDB_PYTHON:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python}"

MODE="${MODE:-smoke}"
RUN_GRANULARITY="${RUN_GRANULARITY:-full_corpus}"
HOLD_JOB_ID="${HOLD_JOB_ID:-45269}"
INSIDE_HOLD_STEP="${INSIDE_HOLD_STEP:-0}"
MAX_PARALLEL="${MAX_PARALLEL_OVERRIDE:-4}"
SMOKE_MAX_SENTENCES="${SMOKE_MAX_SENTENCES_OVERRIDE:-1}"
LANGS="${LANGS_OVERRIDE:-zh ja de}"
LMS="${LMS_OVERRIDE:-1 2 3 4}"
GLOSSARY_KINDS="${GLOSSARY_KINDS_OVERRIDE:-raw gs1k gs10k}"
PAPERS="${PAPERS_OVERRIDE:-2022.acl-long.268 2022.acl-long.367 2022.acl-long.590 2022.acl-long.110 2022.acl-long.117}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)}"

DATA_ROOT="${DATA_ROOT:-/mnt/taurus/data/siqiouyang/datasets/acl6060}"
DEV_SOURCE="${DATA_ROOT}/dev.source"
DEV_AUDIO_YAML="${DATA_ROOT}/dev.yaml"
DEV_SOURCE_TEXT="${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.en.txt"
REF_TEMPLATE="${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.%s.txt"

OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/tagged_acl_origin_bsz4_tau073_baseline_${RUN_STAMP}/${MODE}}"
INPUT_ROOT="${INPUT_ROOT_OVERRIDE:-${OUTPUT_BASE}/__inputs__}"
LOG_DIR="${LOG_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_origin_bsz4_tau073_${RUN_STAMP}}"
SUMMARY_DIR="${SUMMARY_DIR_OVERRIDE:-${OUTPUT_BASE}/__summary__}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/maxsim_index_cache}"

GS10K_GLOSSARY="${GS10K_GLOSSARY_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_union_gs10000_min_norm2_backfill.json}"
RAW_GLOSSARY="${RAW_GLOSSARY_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
GS1K_GLOSSARY="${GS1K_GLOSSARY_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_union_gs1000_min_norm2_backfill.json}"
EXTRACTED_GLOSSARY="${EXTRACTED_GLOSSARY_OVERRIDE:-${ROOT_DIR}/documents/data/data_pre/extracted_glossaries_by_paper/extracted_glossary__2022.acl-long.110.json}"
EXTRACTED_GS10K_GLOSSARY="${EXTRACTED_GS10K_GLOSSARY_OVERRIDE:-${ROOT_DIR}/documents/data/data_pre/expanded_glossaries_by_paper/expanded_glossary__2022.acl-long.110_gs10000.json}"
EVAL_GLOSSARY_PATH_GLOBAL="${EVAL_GLOSSARY_PATH_GLOBAL_OVERRIDE:-${RAW_GLOSSARY}}"
EVAL_GLOSSARY_FOLLOWS_KIND="${EVAL_GLOSSARY_FOLLOWS_KIND_OVERRIDE:-0}"

RAG_MODEL_PATH="${RAG_MODEL_PATH_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
RAG_TOP_K="${RAG_TOP_K_OVERRIDE:-10}"
RAG_SCORE_THRESHOLD="${RAG_SCORE_THRESHOLD_OVERRIDE:-0.73}"
RAG_LORA_R="${RAG_LORA_R_OVERRIDE:-128}"
RAG_TEXT_LORA_R="${RAG_TEXT_LORA_R_OVERRIDE:-128}"
RAG_TIMELINE_LOOKBACK_SEC="${RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE:-1.92}"
RAG_STREAMING_MODE="${RAG_STREAMING_MODE_OVERRIDE:-timeline}"
TERM_MAP_FORMAT="${TERM_MAP_FORMAT_OVERRIDE:-plain}"
DENSITY_TAG="${DENSITY_TAG_OVERRIDE:-tagacl_origin_bsz4_tau073}"
NOTES_FILE="${NOTES_FILE_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260520__tagged_acl_origin_bsz4_tau073_baseline_sweep.md}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX_OVERRIDE:-origin_bsz4}"
WANDB_EXPERIMENT_FAMILY="${WANDB_EXPERIMENT_FAMILY_OVERRIDE:-tagged_acl_origin_bsz4_tau073}"
WANDB_VARIANT_PREFIX="${WANDB_VARIANT_PREFIX_OVERRIDE:-origin}"
WANDB_COMPUTE_TAG="${WANDB_COMPUTE_TAG_OVERRIDE:-compute:taurus45269}"

TERM_FCR_POLICY="${TERM_FCR_POLICY_OVERRIDE:-term_map_source_ref_negative_sentence}"
GPU_PAIRS_CSV="${GPU_PAIRS_CSV_OVERRIDE:-0,1;2,3;4,5;6,7}"
WANDB_HOME="${WANDB_HOME:-/mnt/taurus/home/jiaxuanluo}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN_OVERRIDE:-16384}"

model_for_lang() {
  if [[ -n "${MODEL_NAME_OVERRIDE:-}" ]]; then
    printf '%s\n' "${MODEL_NAME_OVERRIDE}"
    return 0
  fi
  case "$1" in
    zh) printf '%s\n' "/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_origin-bsz4" ;;
    ja) printf '%s\n' "/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-ja-s_origin-bsz4" ;;
    de) printf '%s\n' "/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_origin-bsz4" ;;
    *) echo "[ERROR] Unsupported language: $1" >&2; return 2 ;;
  esac
}

glossary_for_kind() {
  case "$1" in
    raw) printf '%s\n' "${RAW_GLOSSARY}" ;;
    gs1k) printf '%s\n' "${GS1K_GLOSSARY}" ;;
    gs10k) printf '%s\n' "${GS10K_GLOSSARY}" ;;
    extracted|extracted110) printf '%s\n' "${EXTRACTED_GLOSSARY}" ;;
    extracted_gs10k|extracted110_gs10k) printf '%s\n' "${EXTRACTED_GS10K_GLOSSARY}" ;;
    *) echo "[ERROR] Unsupported glossary kind: $1" >&2; return 2 ;;
  esac
}

glossary_tag_for_kind() {
  basename "$(glossary_for_kind "$1")" .json
}

prepare_glossaries() {
  mkdir -p "$(dirname "${RAW_GLOSSARY}")"
  "${PYTHON_BIN}" - "${GS10K_GLOSSARY}" "${RAW_GLOSSARY}" "${GS1K_GLOSSARY}" <<'PY'
import json
import sys
from pathlib import Path

gs10k_path, raw_path, gs1k_path = map(Path, sys.argv[1:4])
data = json.loads(gs10k_path.read_text(encoding="utf-8"))
if not isinstance(data, list):
    raise SystemExit(f"expected list glossary: {gs10k_path}")
raw = [x for x in data if isinstance(x, dict) and x.get("source") == "acl_tagged_gt"]
if len(raw) != 238:
    raise SystemExit(f"expected 238 tagged raw terms, found {len(raw)}")
gs1k = data[:1000]
if len(gs1k) != 1000:
    raise SystemExit(f"expected first 1000 entries for gs1k, found {len(gs1k)}")
raw_terms = {x.get("term") for x in raw}
gs1k_terms = {x.get("term") for x in gs1k if isinstance(x, dict)}
if not raw_terms.issubset(gs1k_terms):
    missing = sorted(raw_terms - gs1k_terms)[:10]
    raise SystemExit(f"gs1k does not contain all raw tagged terms; examples={missing}")
raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
gs1k_path.write_text(json.dumps(gs1k, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[PREP] raw={raw_path} terms={len(raw)}")
print(f"[PREP] gs1k={gs1k_path} terms={len(gs1k)}")
print(f"[PREP] gs10k={gs10k_path} terms={len(data)}")
PY
}

prepare_inputs() {
  mkdir -p "${INPUT_ROOT}/full" "${INPUT_ROOT}/smoke_wavs"
  "${PYTHON_BIN}" - \
    "${INPUT_ROOT}" "${DATA_ROOT}" "${DEV_SOURCE}" "${DEV_AUDIO_YAML}" \
    "${DEV_SOURCE_TEXT}" "${RAW_GLOSSARY}" "${SMOKE_MAX_SENTENCES}" <<'PY'
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import yaml

input_root = Path(sys.argv[1])
data_root = Path(sys.argv[2])
dev_source = Path(sys.argv[3])
dev_audio_yaml = Path(sys.argv[4])
dev_source_text = Path(sys.argv[5])
raw_glossary = Path(sys.argv[6])
smoke_n = int(sys.argv[7])

audio_entries = yaml.safe_load(dev_audio_yaml.read_text(encoding="utf-8"))
if not isinstance(audio_entries, list):
    raise SystemExit(f"invalid yaml list: {dev_audio_yaml}")
source_lines = dev_source.read_text(encoding="utf-8").splitlines()
source_texts = dev_source_text.read_text(encoding="utf-8").splitlines()
if len(audio_entries) != len(source_texts):
    raise SystemExit(f"yaml/source_text mismatch: {len(audio_entries)} vs {len(source_texts)}")

remapped_audio_paths = 0

def normalize_audio_path(path_s: str) -> str:
    global remapped_audio_paths
    path_s = str(path_s or "").strip()
    if not path_s:
        return path_s
    if Path(path_s).exists():
        return path_s
    local_wav = data_root / "dev" / "full_wavs" / Path(path_s).name
    if local_wav.exists():
        remapped_audio_paths += 1
        return str(local_wav)
    if path_s.startswith("/mnt/data/"):
        alt = "/mnt/taurus/data/" + path_s[len("/mnt/data/"):]
        if Path(alt).exists():
            remapped_audio_paths += 1
            return alt
    return path_s

paper_source = {}
normalized_source_lines = []
for line in source_lines:
    normalized = normalize_audio_path(line)
    normalized_source_lines.append(normalized)
    p = Path(normalized)
    paper_source[p.stem] = normalized

terms = []
for item in json.loads(raw_glossary.read_text(encoding="utf-8")):
    term = str(item.get("term") or "").strip()
    if term:
        terms.append(term)

def source_contains(text: str, term: str) -> bool:
    text_norm = re.sub(r"\s+", " ", text or "").strip().casefold()
    term_norm = re.sub(r"\s+", " ", term or "").strip().casefold()
    if not text_norm or not term_norm:
        return False
    if re.fullmatch(r"[a-z0-9][a-z0-9 ._+/#-]*", term_norm):
        return re.search(r"(?<![a-z0-9])" + re.escape(term_norm) + r"(?![a-z0-9])", text_norm) is not None
    return term_norm in text_norm

by_paper_indices = defaultdict(list)
for idx, item in enumerate(audio_entries):
    if isinstance(item, dict) and "wav" in item:
        item["wav"] = normalize_audio_path(str(item.get("wav", "")))
    paper = Path(str(item.get("wav", ""))).stem
    by_paper_indices[paper].append(idx)

langs = ["zh", "ja", "de"]
refs_by_lang = {}
for lang in langs:
    ref_path = data_root / "dev" / "text" / "txt" / f"ACL.6060.dev.en-xx.{lang}.txt"
    refs = ref_path.read_text(encoding="utf-8").splitlines()
    if len(refs) != len(audio_entries):
        raise SystemExit(f"ref/yaml mismatch for {lang}: {len(refs)} vs {len(audio_entries)}")
    refs_by_lang[lang] = refs

full_root = input_root / "full"
full_root.mkdir(parents=True, exist_ok=True)
full_all_root = input_root / "full_all"
full_all_root.mkdir(parents=True, exist_ok=True)
paper_order = [Path(line.strip()).stem for line in normalized_source_lines]
for lang, refs in refs_by_lang.items():
    all_dir = full_all_root / lang / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    (all_dir / "source.list").write_text("\n".join(paper_source[p] for p in paper_order) + "\n", encoding="utf-8")
    (all_dir / "target.list").write_text(
        "\n".join(" ".join(refs[i].strip() for i in by_paper_indices[p]) for p in paper_order) + "\n",
        encoding="utf-8",
    )
    (all_dir / "source_text.txt").write_text("\n".join(source_texts) + "\n", encoding="utf-8")
    (all_dir / "ref.txt").write_text("\n".join(refs) + "\n", encoding="utf-8")
    (all_dir / "audio.yaml").write_text(
        yaml.safe_dump(audio_entries, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    for paper, indices in by_paper_indices.items():
        if paper not in paper_source:
            raise SystemExit(f"paper missing from dev.source: {paper}")
        out_dir = full_root / lang / paper
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "source.list").write_text(paper_source[paper] + "\n", encoding="utf-8")
        (out_dir / "target.list").write_text(" ".join(refs[i].strip() for i in indices) + "\n", encoding="utf-8")
        (out_dir / "source_text.txt").write_text("\n".join(source_texts[i] for i in indices) + "\n", encoding="utf-8")
        (out_dir / "ref.txt").write_text("\n".join(refs[i] for i in indices) + "\n", encoding="utf-8")
        (out_dir / "audio.yaml").write_text(
            yaml.safe_dump([audio_entries[i] for i in indices], allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

selected = []
for idx, text in enumerate(source_texts):
    if any(source_contains(text, term) for term in terms):
        selected.append(idx)
    if len(selected) >= smoke_n:
        break
if len(selected) != smoke_n:
    raise SystemExit(f"could only find {len(selected)} term-bearing smoke rows, wanted {smoke_n}")

smoke_wav_dir = input_root / "smoke_wavs"
smoke_wav_dir.mkdir(parents=True, exist_ok=True)
for lang, refs in refs_by_lang.items():
    out_dir = input_root / "smoke" / lang / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_paths = []
    smoke_yaml = []
    for out_idx, src_idx in enumerate(selected):
        item = dict(audio_entries[src_idx])
        wav_name = str(item["wav"])
        src_wav = Path(wav_name) if Path(wav_name).is_absolute() else data_root / "dev" / "full_wavs" / wav_name
        if not src_wav.is_file():
            raise SystemExit(f"source wav not found: {src_wav}")
        audio, sr = sf.read(str(src_wav), dtype="float32", always_2d=False)
        start = int(round(float(item["offset"]) * sr))
        stop = int(round((float(item["offset"]) + float(item["duration"])) * sr))
        clip = audio[start:stop]
        clip_path = smoke_wav_dir / f"smoke_{out_idx:03d}_{src_idx:04d}_{Path(wav_name).stem}.wav"
        sf.write(str(clip_path), clip, sr)
        source_paths.append(str(clip_path))
        smoke_yaml.append({"duration": float(item["duration"]), "offset": 0.0, "wav": clip_path.name})
    (out_dir / "source.list").write_text("\n".join(source_paths) + "\n", encoding="utf-8")
    (out_dir / "target.list").write_text("\n".join(refs[i] for i in selected) + "\n", encoding="utf-8")
    (out_dir / "source_text.txt").write_text("\n".join(source_texts[i] for i in selected) + "\n", encoding="utf-8")
    (out_dir / "ref.txt").write_text("\n".join(refs[i] for i in selected) + "\n", encoding="utf-8")
    (out_dir / "audio.yaml").write_text(yaml.safe_dump(smoke_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")

print(f"[PREP] full paper inputs under {full_root}")
print(f"[PREP] remapped_audio_paths={remapped_audio_paths}")
print(f"[PREP] smoke rows={len(selected)} selected_indices={selected}")
PY
}

index_path_for_glossary() {
  local glossary_path="$1"
  local model_hash glossary_hash glossary_tag
  model_hash="$(printf '%s' "${RAG_MODEL_PATH}" | sha1sum | awk '{print substr($1,1,10)}')"
  glossary_hash="$(printf '%s' "${glossary_path}" | sha1sum | awk '{print substr($1,1,10)}')"
  glossary_tag="$(basename "${glossary_path}" .json)"
  printf '%s/lh1b88kw_%s__%s__%s__maxsim.pt\n' \
    "${INDEX_CACHE_DIR}" "${model_hash}" "${glossary_tag}" "${glossary_hash}"
}

run_srun_or_direct() {
  local gpu_pair="$1"
  local log_prefix="$2"
  shift 2
  mkdir -p "${LOG_DIR}"
  if [[ -n "${HOLD_JOB_ID}" && "${HOLD_JOB_ID}" != "0" && "${INSIDE_HOLD_STEP}" != "1" ]]; then
    srun --jobid="${HOLD_JOB_ID}" --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 \
      --chdir="${ROOT_DIR}" \
      env CUDA_VISIBLE_DEVICES="${gpu_pair}" CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${gpu_pair}" "$@" \
      > "${log_prefix}.out" 2> "${log_prefix}.err"
  else
    CUDA_VISIBLE_DEVICES="${gpu_pair}" CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${gpu_pair}" "$@" \
      > "${log_prefix}.out" 2> "${log_prefix}.err"
  fi
}

prebuild_indices() {
  local gpu_pair="$1"
  local kind glossary_path index_path
  for kind in ${GLOSSARY_KINDS}; do
    glossary_path="$(glossary_for_kind "${kind}")"
    index_path="$(index_path_for_glossary "${glossary_path}")"
    if [[ -f "${index_path}" ]]; then
      echo "[INDEX] exists kind=${kind}: ${index_path}"
      continue
    fi
    echo "[INDEX] build kind=${kind}: ${index_path}"
    local log_prefix="${LOG_DIR}/index_${kind}_${RUN_STAMP}"
    run_srun_or_direct "${gpu_pair}" "${log_prefix}" \
      "${PYTHON_BIN}" "${ROOT_DIR}/retriever/gigaspeech/build_maxsim_index.py" \
        --model-path "${RAG_MODEL_PATH}" \
        --glossary-path "${glossary_path}" \
        --output-path "${index_path}" \
        --device cuda:0 \
        --text-lora-rank "${RAG_TEXT_LORA_R}"
  done
}

input_dir_for() {
  local mode="$1" lang="$2" paper="$3"
  if [[ "${mode}" == "smoke" ]]; then
    printf '%s/smoke/%s/smoke\n' "${INPUT_ROOT}" "${lang}"
  else
    if [[ "${paper}" == "all" ]]; then
      printf '%s/full_all/%s/all\n' "${INPUT_ROOT}" "${lang}"
    else
      printf '%s/full/%s/%s\n' "${INPUT_ROOT}" "${lang}" "${paper}"
    fi
  fi
}

aggregate_setting() {
  local lang="$1" lm="$2" kind="$3" glossary_tag="$4" papers="$5"
  local out_dir="${OUTPUT_BASE}/${lang}/d${DENSITY_TAG}_lm${lm}_k${RAG_TOP_K}_th${RAG_SCORE_THRESHOLD}_g${glossary_tag}"
  mkdir -p "${out_dir}" "${SUMMARY_DIR}"
  "${PYTHON_BIN}" - "${OUTPUT_BASE}" "${lang}" "${DENSITY_TAG}" "${lm}" "${RAG_TOP_K}" "${RAG_SCORE_THRESHOLD}" "${glossary_tag}" "${papers}" "${out_dir}/eval_results.tsv" <<'PY'
import csv
import sys
from pathlib import Path

base, lang, density, lm, topk, tau, glossary_tag, papers_s, out_tsv = sys.argv[1:10]
papers = [p for p in papers_s.split() if p]
rows = []
for paper in papers:
    p = Path(base) / lang / f"d{density}_lm{lm}_k{topk}_th{tau}_g{glossary_tag}_pp{paper}" / "eval_results.tsv"
    if not p.is_file():
        raise SystemExit(f"missing per-paper eval TSV: {p}")
    with p.open("r", encoding="utf-8", newline="") as f:
        data = list(csv.DictReader(f, delimiter="\t"))
    if len(data) != 1:
        raise SystemExit(f"expected one row in {p}, got {len(data)}")
    rows.append(data[0])

def f(row, key):
    try:
        return float(row.get(key) or 0)
    except ValueError:
        return 0.0

def i(row, key):
    try:
        return int(float(row.get(key) or 0))
    except ValueError:
        return 0

term_correct = sum(i(r, "TERM_CORRECT") for r in rows)
term_total = sum(i(r, "TERM_TOTAL") for r in rows)
adopted = sum(i(r, "TERM_ADOPTED") for r in rows)
adopt_total = sum(i(r, "TERM_ADOPTION_TOTAL") for r in rows)
real_adopted = sum(i(r, "REAL_TERM_ADOPTED") for r in rows)
real_total = sum(i(r, "REAL_TERM_ADOPT_TOTAL") for r in rows)
false_copy = sum(i(r, "FALSE_COPY") for r in rows)
neg_total = sum(i(r, "NEG_TOTAL") for r in rows)
false_copy_terms = sum(i(r, "FALSE_COPY_TERMS") for r in rows)
source_false_copy = sum(i(r, "SOURCE_FALSE_COPY") for r in rows)
source_neg_total = sum(i(r, "SOURCE_NEG_TOTAL") for r in rows)
source_false_copy_terms = sum(i(r, "SOURCE_FALSE_COPY_TERMS") for r in rows)
fcr_modes = sorted({r.get("TERM_FCR_MODE", "") for r in rows if r.get("TERM_FCR_MODE")})
header = [
    "mode", "lang_code", "BLEU", "StreamLAAL", "StreamLAAL_CA",
    "TERM_ACC", "TERM_CORRECT", "TERM_TOTAL",
    "TERM_ADOPTION", "TERM_ADOPTED", "TERM_ADOPTION_TOTAL", "TERM_ADOPTION_SENTENCES",
    "TERM_ADOPTION_MICRO", "REAL_TERM_ADOPT", "REAL_TERM_ADOPTED",
    "REAL_TERM_ADOPT_TOTAL", "REAL_TERM_ADOPT_SENTENCES", "REAL_TERM_ADOPT_MICRO",
    "TERM_FCR", "FALSE_COPY", "NEG_TOTAL", "FALSE_COPY_TERMS",
    "instances_log", "TERM_FCR_MODE", "SOURCE_TERM_SENT_FCR",
    "SOURCE_FALSE_COPY", "SOURCE_NEG_TOTAL", "SOURCE_FALSE_COPY_TERMS",
]
out = {
    "mode": "acl6060_tagged_perpaper_aggregate",
    "lang_code": lang,
    "BLEU": sum(f(r, "BLEU") for r in rows) / len(rows),
    "StreamLAAL": sum(f(r, "StreamLAAL") for r in rows) / len(rows),
    "StreamLAAL_CA": sum(f(r, "StreamLAAL_CA") for r in rows) / len(rows),
    "TERM_ACC": term_correct / term_total if term_total else 0.0,
    "TERM_CORRECT": term_correct,
    "TERM_TOTAL": term_total,
    "TERM_ADOPTION": adopted / adopt_total if adopt_total else 0.0,
    "TERM_ADOPTED": adopted,
    "TERM_ADOPTION_TOTAL": adopt_total,
    "TERM_ADOPTION_SENTENCES": sum(i(r, "TERM_ADOPTION_SENTENCES") for r in rows),
    "TERM_ADOPTION_MICRO": adopted / adopt_total if adopt_total else 0.0,
    "REAL_TERM_ADOPT": real_adopted / real_total if real_total else 0.0,
    "REAL_TERM_ADOPTED": real_adopted,
    "REAL_TERM_ADOPT_TOTAL": real_total,
    "REAL_TERM_ADOPT_SENTENCES": sum(i(r, "REAL_TERM_ADOPT_SENTENCES") for r in rows),
    "REAL_TERM_ADOPT_MICRO": real_adopted / real_total if real_total else 0.0,
    "TERM_FCR": false_copy / neg_total if neg_total else 0.0,
    "FALSE_COPY": false_copy,
    "NEG_TOTAL": neg_total,
    "FALSE_COPY_TERMS": false_copy_terms,
    "instances_log": "per-paper aggregate",
    "TERM_FCR_MODE": ",".join(fcr_modes),
    "SOURCE_TERM_SENT_FCR": source_false_copy / source_neg_total if source_neg_total else 0.0,
    "SOURCE_FALSE_COPY": source_false_copy,
    "SOURCE_NEG_TOTAL": source_neg_total,
    "SOURCE_FALSE_COPY_TERMS": source_false_copy_terms,
}
out_path = Path(out_tsv)
with out_path.open("w", encoding="utf-8", newline="") as fobj:
    writer = csv.DictWriter(fobj, fieldnames=header, delimiter="\t")
    writer.writeheader()
    writer.writerow(out)
print(f"[AGG] wrote {out_path}")
PY
}

log_setting_to_wandb() {
  local lang="$1" lm="$2" kind="$3" glossary_tag="$4"
  local tau_tag="tau${RAG_SCORE_THRESHOLD/./}"
  [[ "${WANDB_LOG_OVERRIDE:-1}" == "1" ]] || return 0
  HOME="${WANDB_HOME}" \
  WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_HOME}/.config/wandb}" \
  "${WANDB_PYTHON}" "${WANDB_LOGGER}" \
    --project simuleval_eval \
    --run-name "${WANDB_RUN_PREFIX}__tagged_acl__${lang}__lm${lm}__${tau_tag}__${kind}__${MODE}" \
    --experiment-family "${WANDB_EXPERIMENT_FAMILY}" \
    --data-tag "tagged_acl_strict_raw_${lang}" \
    --task-tag "$([[ "${MODE}" == "smoke" ]] && echo smoke || echo eval)" \
    --notes-file "${NOTES_FILE}" \
    --extra-tags "variant:${WANDB_VARIANT_PREFIX}_${lang}_${kind}_lm${lm}" "${WANDB_COMPUTE_TAG}" "tau:${tau_tag}" "glossary:${kind}" "lang:${lang}" \
    --density "${DENSITY_TAG}" \
    --rag-top-k "${RAG_TOP_K}" \
    --rag-score-threshold "${RAG_SCORE_THRESHOLD}" \
    --output-base "${OUTPUT_BASE}" \
    --lang-code "${lang}" \
    --latency-multipliers "${lm}" \
    --glossary-tag "${glossary_tag}" \
    --model-name "$(model_for_lang "${lang}")" \
    --rag-model-path "${RAG_MODEL_PATH}" \
    --verdict "Tagged ACL ${MODE}: ${WANDB_RUN_PREFIX} ${lang}, lm=${lm}, glossary=${kind}; retrieval tau=${RAG_SCORE_THRESHOLD}, timeline lookback=${RAG_TIMELINE_LOOKBACK_SEC}s, metric glossary policy follows_kind=${EVAL_GLOSSARY_FOLLOWS_KIND}, default fixed raw tagged denominator."
}

run_paper_eval() {
  local lang="$1" lm="$2" kind="$3" paper="$4" gpu_pair="$5"
  local glossary_path eval_glossary_path glossary_tag index_path in_dir model_path log_prefix paper_id_tag
  local port_seed vllm_port
  glossary_path="$(glossary_for_kind "${kind}")"
  if [[ "${EVAL_GLOSSARY_FOLLOWS_KIND}" == "1" ]]; then
    eval_glossary_path="${glossary_path}"
  else
    eval_glossary_path="${EVAL_GLOSSARY_PATH_GLOBAL}"
  fi
  glossary_tag="$(basename "${glossary_path}" .json)"
  index_path="$(index_path_for_glossary "${glossary_path}")"
  in_dir="$(input_dir_for "${MODE}" "${lang}" "${paper}")"
  model_path="$(model_for_lang "${lang}")"
  log_prefix="${LOG_DIR}/${MODE}_${lang}_lm${lm}_${kind}_${paper}_${RUN_STAMP}"
  if [[ "${paper}" == "all" ]]; then
    paper_id_tag=""
  else
    paper_id_tag="${paper}"
  fi
  if [[ ! -d "${in_dir}" ]]; then
    echo "[ERROR] input dir missing: ${in_dir}" >&2
    return 3
  fi
  port_seed="$(printf '%s' "${MODE}_${lang}_${lm}_${kind}_${paper}_${RUN_STAMP}" | cksum | awk '{print $1}')"
  vllm_port="$((24000 + (port_seed % 20000)))"
  echo "[RUN] mode=${MODE} lang=${lang} lm=${lm} glossary=${kind} paper=${paper} gpu=${gpu_pair} vllm_port=${vllm_port}"
  run_srun_or_direct "${gpu_pair}" "${log_prefix}" \
    env \
      VLLM_PORT="${vllm_port}" \
      MODEL_NAME_OVERRIDE="${model_path}" \
      RAG_MODEL_PATH_OVERRIDE="${RAG_MODEL_PATH}" \
      RAG_LORA_R_OVERRIDE="${RAG_LORA_R}" \
      RAG_TEXT_LORA_R_OVERRIDE="${RAG_TEXT_LORA_R}" \
      OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
      EVAL_MODE_OVERRIDE="acl6060" \
      LANG_CODE_OVERRIDE="${lang}" \
      GLOSSARY_PATH_OVERRIDE="${glossary_path}" \
      INDEX_PATH_OVERRIDE="${index_path}" \
      EVAL_GLOSSARY_PATH_OVERRIDE="${eval_glossary_path}" \
      SRC_LIST_OVERRIDE="${in_dir}/source.list" \
      TGT_LIST_OVERRIDE="${in_dir}/target.list" \
      REF_FILE_OVERRIDE="${in_dir}/ref.txt" \
      SOURCE_TEXT_FILE_OVERRIDE="${in_dir}/source_text.txt" \
      AUDIO_YAML_OVERRIDE="${in_dir}/audio.yaml" \
      LATENCY_MULTIPLIER_OVERRIDE="${lm}" \
      RAG_TOP_K_OVERRIDE="${RAG_TOP_K}" \
      RAG_SCORE_THRESHOLD_OVERRIDE="${RAG_SCORE_THRESHOLD}" \
      RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE="${RAG_TIMELINE_LOOKBACK_SEC}" \
      RAG_STREAMING_MODE_OVERRIDE="${RAG_STREAMING_MODE}" \
      TERM_MAP_FORMAT_OVERRIDE="${TERM_MAP_FORMAT}" \
      TERM_FCR_POLICY="${TERM_FCR_POLICY}" \
      INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}" \
      EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR_OVERRIDE:-${TMPDIR:-/mnt/gemini/data1/jiaxuanluo/tmp}}" \
      HF_HOME="${HF_HOME:-}" \
      TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-}" \
      HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-}" \
      TORCH_HOME="${TORCH_HOME:-}" \
      WANDB_HOME="${WANDB_HOME:-}" \
      WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-}" \
      GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}" \
      MAX_CACHE_SECONDS_OVERRIDE="${MAX_CACHE_SECONDS_OVERRIDE:-}" \
      KEEP_CACHE_SECONDS_OVERRIDE="${KEEP_CACHE_SECONDS_OVERRIDE:-}" \
      MAX_CACHE_CHUNKS_OVERRIDE="${MAX_CACHE_CHUNKS_OVERRIDE:-}" \
      KEEP_CACHE_CHUNKS_OVERRIDE="${KEEP_CACHE_CHUNKS_OVERRIDE:-}" \
      VLLM_LIMIT_AUDIO_OVERRIDE="${VLLM_LIMIT_AUDIO_OVERRIDE:-}" \
      VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN}" \
      STRIP_OUTPUT_TAGS_OVERRIDE="${STRIP_OUTPUT_TAGS_OVERRIDE:-none}" \
      DENSITY_TAG="${DENSITY_TAG}" \
      PAPER_ID_TAG="${paper_id_tag}" \
      VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME="VLLM_OBJ_${MODE}_${lang}_${lm}_${kind}_${paper}_${RUN_STAMP}_${RANDOM}" \
      bash "${EVAL_SCRIPT}"
}

run_setting() {
  local lang="$1" lm="$2" kind="$3" gpu_pair="$4"
  local glossary_tag papers_for_setting
  glossary_tag="$(glossary_tag_for_kind "${kind}")"
  if [[ "${MODE}" == "smoke" ]]; then
    papers_for_setting="smoke"
  elif [[ "${RUN_GRANULARITY}" == "full_corpus" ]]; then
    papers_for_setting="all"
  else
    papers_for_setting="${PAPERS}"
  fi
  for paper in ${papers_for_setting}; do
    run_paper_eval "${lang}" "${lm}" "${kind}" "${paper}" "${gpu_pair}"
  done
  if [[ "${papers_for_setting}" != "all" ]]; then
    aggregate_setting "${lang}" "${lm}" "${kind}" "${glossary_tag}" "${papers_for_setting}"
  fi
  log_setting_to_wandb "${lang}" "${lm}" "${kind}" "${glossary_tag}"
  echo "[DONE] setting mode=${MODE} lang=${lang} lm=${lm} glossary=${kind}"
}

wait_for_slot() {
  local max_parallel="$1"
  while (( $(jobs -pr | wc -l) >= max_parallel )); do
    wait -n
  done
}

main() {
  if [[ "${MODE}" != "prepare" && -n "${HOLD_JOB_ID}" && "${HOLD_JOB_ID}" != "0" && "${INSIDE_HOLD_STEP}" != "1" ]]; then
    echo "[HOLD] entering one 8-GPU step in job ${HOLD_JOB_ID}; inner workers will split ${GPU_PAIRS_CSV}"
    exec srun --jobid="${HOLD_JOB_ID}" --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 \
      --chdir="${ROOT_DIR}" \
      env INSIDE_HOLD_STEP=1 HOLD_JOB_ID=0 \
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
        CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="0,1,2,3,4,5,6,7" \
        bash "${LAUNCHER_SELF}"
  fi

  mkdir -p "${OUTPUT_BASE}" "${INPUT_ROOT}" "${LOG_DIR}" "${SUMMARY_DIR}"
  for p in "${EVAL_SCRIPT}" "${WANDB_LOGGER}" "${NOTES_FILE}" "${RAG_MODEL_PATH}" "${GS10K_GLOSSARY}"; do
    if [[ ! -e "${p}" ]]; then
      echo "[ERROR] Missing required path: ${p}" >&2
      exit 3
    fi
  done
  prepare_glossaries
  prepare_inputs
  if [[ "${MODE}" == "prepare" ]]; then
    echo "[DONE] prepare only: ${INPUT_ROOT}"
    exit 0
  fi

  IFS=';' read -r -a GPU_PAIRS <<< "${GPU_PAIRS_CSV}"
  if (( ${#GPU_PAIRS[@]} < 1 )); then
    echo "[ERROR] no GPU pairs configured" >&2
    exit 2
  fi
  if [[ "${MODE}" == "smoke" ]]; then
    local old_glossary_kinds="${GLOSSARY_KINDS}"
    GLOSSARY_KINDS="${SMOKE_GLOSSARY_KIND_OVERRIDE:-raw}"
    prebuild_indices "${GPU_PAIRS[$((${#GPU_PAIRS[@]} - 1))]}"
    GLOSSARY_KINDS="${old_glossary_kinds}"
  else
    prebuild_indices "${GPU_PAIRS[$((${#GPU_PAIRS[@]} - 1))]}"
  fi

  local setting_idx=0
  if [[ "${MODE}" == "smoke" ]]; then
    run_setting "${SMOKE_LANG_OVERRIDE:-zh}" "${SMOKE_LM_OVERRIDE:-2}" "${SMOKE_GLOSSARY_KIND_OVERRIDE:-raw}" "${GPU_PAIRS[0]}"
  elif [[ "${MODE}" == "full" ]]; then
    for lang in ${LANGS}; do
      for lm in ${LMS}; do
        for kind in ${GLOSSARY_KINDS}; do
          wait_for_slot "${MAX_PARALLEL}"
          local gpu_pair="${GPU_PAIRS[$((setting_idx % ${#GPU_PAIRS[@]}))]}"
          run_setting "${lang}" "${lm}" "${kind}" "${gpu_pair}" &
          setting_idx=$((setting_idx + 1))
        done
      done
    done
    local failed=0
    for pid in $(jobs -pr); do
      if ! wait "${pid}"; then
        failed=1
      fi
    done
    if (( failed )); then
      echo "[ERROR] one or more settings failed" >&2
      exit 1
    fi
  else
    echo "[ERROR] Unsupported MODE=${MODE}" >&2
    exit 2
  fi

  echo "[ALL DONE] mode=${MODE} output_base=${OUTPUT_BASE} logs=${LOG_DIR}"
}

main "$@"
