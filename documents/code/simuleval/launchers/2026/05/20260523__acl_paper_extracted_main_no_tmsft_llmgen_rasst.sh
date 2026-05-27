#!/usr/bin/env bash
set -euo pipefail

# Portable ACL paper-extracted main-result launcher.
#
# Runs single-paper SimulEval jobs and aggregates the 5 paper rows into one
# setting-level eval_results.tsv.  TERM metrics are always scored against the
# paper's raw extracted glossary, even when the runtime glossary is gs1k/gs10k.

PSC_BASE="${PSC_BASE:-}"
USE_APPTAINER="${USE_APPTAINER:-0}"
APPTAINER_SIF="${APPTAINER_SIF:-${PSC_BASE:+${PSC_BASE}/containers/ubuntu_22_04_gcc.sif}}"
IN_APPTAINER="${IN_APPTAINER:-0}"

if [[ "${USE_APPTAINER}" == "1" && "${IN_APPTAINER}" != "1" ]]; then
  if [[ -z "${APPTAINER_SIF}" || ! -f "${APPTAINER_SIF}" ]]; then
    echo "[ERROR] APPTAINER_SIF not found: ${APPTAINER_SIF:-<empty>}" >&2
    exit 3
  fi
  export IN_APPTAINER=1
  export APPTAINERENV_IN_APPTAINER=1
  export APPTAINERENV_PSC_BASE="${PSC_BASE}"
  exec apptainer exec --nv -B /ocean,/jet "${APPTAINER_SIF}" bash "$0" "$@"
fi

if [[ -n "${PSC_BASE}" ]]; then
  ROOT_DIR="${ROOT_DIR:-${PSC_BASE}/src/InfiniSST}"
  ENV_DIR="${ENV_DIR:-${PSC_BASE}/envs/spaCyEnv_20260518}"
  DATA_ROOT="${DATA_ROOT:-${PSC_BASE}/data/acl6060}"
  MODEL_ROOT_OWASKI="${MODEL_ROOT_OWASKI:-${PSC_BASE}/models/owaski}"
  RAG_MODEL_PATH="${RAG_MODEL_PATH:-${PSC_BASE}/checkpoints/lh1b88kw_best_eval_acl6060_recallat10.pt}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-${PSC_BASE}/outputs/acl_paper_extracted_main/${RUN_STAMP:-manual}}"
  LOG_ROOT="${LOG_ROOT:-${PSC_BASE}/logs/acl_paper_extracted_main/${RUN_STAMP:-manual}}"
  INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-${PSC_BASE}/cache/maxsim_index_cache}"
  EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR_OVERRIDE:-/tmp/${USER:-jluo7}/infinisst_${SLURM_JOB_ID:-manual}}"
  MWERSEGMENTER_ROOT="${MWERSEGMENTER_ROOT:-${PSC_BASE}/tools/mwerSegmenter}"
  FBK_FAIRSEQ_ROOT="${FBK_FAIRSEQ_ROOT:-${PSC_BASE}/tools/FBK-fairseq}"
else
  ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
  ENV_DIR="${ENV_DIR:-}"
  DATA_ROOT="${DATA_ROOT:-/mnt/taurus/data/siqiouyang/datasets/acl6060}"
  MODEL_ROOT_OWASKI="${MODEL_ROOT_OWASKI:-/mnt/gemini/data/jiaxuanluo/owaski}"
  RAG_MODEL_PATH="${RAG_MODEL_PATH:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/gemini/data2/jiaxuanluo/acl_paper_extracted_main_${RUN_STAMP:-manual}}"
  LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/acl_paper_extracted_main_${RUN_STAMP:-manual}}"
  INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/gemini/data2/jiaxuanluo/maxsim_index_cache}"
  EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR_OVERRIDE:-${TMPDIR:-/mnt/gemini/data1/jiaxuanluo/tmp}}"
  MWERSEGMENTER_ROOT="${MWERSEGMENTER_ROOT:-/mnt/taurus/home/jiaxuanluo/mwerSegmenter}"
  FBK_FAIRSEQ_ROOT="${FBK_FAIRSEQ_ROOT:-/mnt/taurus/home/jiaxuanluo/FBK-fairseq}"
fi

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${ROOT_DIR}/documents/code/simuleval/eval_density_unified.sh}"
WANDB_LOGGER="${WANDB_LOGGER:-${ROOT_DIR}/documents/code/offline_evaluation/wandb_eval_logger.py}"
if [[ -n "${PSC_BASE}" && -x "${ENV_DIR}/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
WANDB_PYTHON="${WANDB_PYTHON:-${PYTHON_BIN}}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260523__acl_paper_extracted_main_no_tmsft_llmgen_rasst.md}"

METHODS="${METHODS:-no_tmsft llmgen_rasst}"
LANGS="${LANGS:-zh de ja}"
LMS="${LMS:-1 2 3 4}"
GLOSSARY_KINDS="${GLOSSARY_KINDS:-raw gs1k gs10k}"
PAPERS="${PAPERS:-2022.acl-long.268 2022.acl-long.367 2022.acl-long.590 2022.acl-long.110 2022.acl-long.117}"
MODE="${MODE:-full}"
SMOKE_MAX_SENTENCES="${SMOKE_MAX_SENTENCES_OVERRIDE:-1}"
MAX_PARALLEL="${MAX_PARALLEL_OVERRIDE:-1}"
GPU_GROUPS_CSV="${GPU_GROUPS_CSV:-0,1,2}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
WANDB_LOG_OVERRIDE="${WANDB_LOG_OVERRIDE:-1}"

RAW_PER_PAPER_GLOSSARY_DIR="${RAW_PER_PAPER_GLOSSARY_DIR:-${ROOT_DIR}/documents/data/data_pre/extracted_glossaries_by_paper}"
EXPANDED_PER_PAPER_GLOSSARY_DIR="${EXPANDED_PER_PAPER_GLOSSARY_DIR:-${ROOT_DIR}/documents/data/data_pre/expanded_glossaries_by_paper}"
GLOBAL_GS1K_GLOSSARY="${GLOBAL_GS1K_GLOSSARY:-${ROOT_DIR}/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs1000.json}"
GLOBAL_GS10K_GLOSSARY="${GLOBAL_GS10K_GLOSSARY:-${ROOT_DIR}/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs10000.json}"
GLOSSARY_SIZE_SOURCE="${GLOSSARY_SIZE_SOURCE:-global_acl_gt_union}"

RAG_TOP_K="${RAG_TOP_K:-10}"
RAG_SCORE_THRESHOLD="${RAG_SCORE_THRESHOLD:-0.73}"
RAG_LORA_R="${RAG_LORA_R:-128}"
RAG_TEXT_LORA_R="${RAG_TEXT_LORA_R:-128}"
RAG_TIMELINE_LOOKBACK_SEC="${RAG_TIMELINE_LOOKBACK_SEC:-1.92}"
RAG_STREAMING_MODE="${RAG_STREAMING_MODE:-timeline}"
TERM_FCR_POLICY="${TERM_FCR_POLICY:-term_map_source_ref_negative_sentence}"
TERM_MAP_FORMAT="${TERM_MAP_FORMAT:-plain}"
GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.70}"
VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN_OVERRIDE:-16384}"
WANDB_COMPUTE_TAG_OVERRIDE="${WANDB_COMPUTE_TAG_OVERRIDE:-compute:portable}"

DEV_SOURCE="${DATA_ROOT}/dev.source"
DEV_AUDIO_YAML="${DATA_ROOT}/dev.yaml"
DEV_SOURCE_TEXT="${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.en.txt"
REF_DIR="${DATA_ROOT}/dev/text/txt"
INPUT_ROOT="${INPUT_ROOT:-${OUTPUT_ROOT}/__inputs__}"
TASK_FILE="${TASK_FILE:-${OUTPUT_ROOT}/__control__/tasks.tsv}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${INDEX_CACHE_DIR}" "${INPUT_ROOT}" "$(dirname "${TASK_FILE}")" "${EVAL_TMPDIR_OVERRIDE}"

if [[ -n "${PSC_BASE}" ]]; then
  if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    echo "[ERROR] PSC env python not found: ${ENV_DIR}/bin/python" >&2
    exit 3
  fi
  export PATH="${ENV_DIR}/bin:${MWERSEGMENTER_ROOT}:${PATH}"
  export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
  export CONDA_PREFIX="${ENV_DIR}"
  export CONDA_DEFAULT_ENV="$(basename "${ENV_DIR}")"
  export HF_HOME="${HF_HOME:-${PSC_BASE}/cache/hf}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${PSC_BASE}/cache/hf/transformers}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PSC_BASE}/cache/hf/datasets}"
  export TORCH_HOME="${TORCH_HOME:-${PSC_BASE}/cache/torch}"
fi
export MWERSEGMENTER_ROOT
export FBK_FAIRSEQ_ROOT

for p in "${EVAL_SCRIPT}" "${WANDB_LOGGER}" "${NOTES_FILE}" "${DATA_ROOT}" "${DEV_SOURCE}" "${DEV_AUDIO_YAML}" "${DEV_SOURCE_TEXT}" "${RAG_MODEL_PATH}" "${MWERSEGMENTER_ROOT}" "${FBK_FAIRSEQ_ROOT}/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

method_label() {
  case "$1" in
    no_tmsft) printf '%s\n' "no_tmsft" ;;
    rasst|llmgen_rasst) printf '%s\n' "llmgen_rasst" ;;
    *) echo "[ERROR] Unsupported method: $1" >&2; return 2 ;;
  esac
}

method_key() {
  case "$1" in
    no_tmsft) printf '%s\n' "no_tmsft" ;;
    rasst|llmgen_rasst) printf '%s\n' "rasst" ;;
    *) echo "[ERROR] Unsupported method: $1" >&2; return 2 ;;
  esac
}

density_for_method() {
  case "$(method_key "$1")" in
    no_tmsft) printf '%s\n' "aclpp_no_tmsft_tau073" ;;
    rasst) printf '%s\n' "aclpp_llmgen_rasst_tau073" ;;
    *) echo "[ERROR] Unsupported method: $1" >&2; return 2 ;;
  esac
}

model_for() {
  local method="$1" lang="$2"
  method="$(method_key "${method}")"
  case "${method}:${lang}" in
    no_tmsft:zh) printf '%s\n' "${NO_TMSFT_ZH_MODEL:-${MODEL_ROOT_OWASKI}/gigaspeech-zh-s_origin-bsz4}" ;;
    no_tmsft:de) printf '%s\n' "${NO_TMSFT_DE_MODEL:-${MODEL_ROOT_OWASKI}/gigaspeech-de-s_origin-bsz4}" ;;
    no_tmsft:ja) printf '%s\n' "${NO_TMSFT_JA_MODEL:-${MODEL_ROOT_OWASKI}/gigaspeech-ja-s_origin-bsz4}" ;;
    rasst:zh) printf '%s\n' "${RASST_ZH_MODEL:-${MODEL_ROOT_OWASKI}/gigaspeech-zh-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4}" ;;
    rasst:de) printf '%s\n' "${RASST_DE_MODEL:-${MODEL_ROOT_OWASKI}/gigaspeech-de-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4}" ;;
    rasst:ja) printf '%s\n' "${RASST_JA_MODEL:-${MODEL_ROOT_OWASKI}/gigaspeech-ja-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4}" ;;
    *) echo "[ERROR] Unsupported method/lang: ${method}/${lang}" >&2; return 2 ;;
  esac
}

raw_glossary_for_paper() {
  printf '%s/extracted_glossary__%s.json\n' "${RAW_PER_PAPER_GLOSSARY_DIR}" "$1"
}

runtime_glossary_for() {
  local kind="$1" paper="$2"
  case "${kind}" in
    raw)
      raw_glossary_for_paper "${paper}"
      ;;
    gs1k)
      if [[ "${GLOSSARY_SIZE_SOURCE}" == "per_paper_expanded" ]]; then
        printf '%s/expanded_glossary__%s_gs1000.json\n' "${EXPANDED_PER_PAPER_GLOSSARY_DIR}" "${paper}"
      else
        printf '%s\n' "${GLOBAL_GS1K_GLOSSARY}"
      fi
      ;;
    gs10k)
      if [[ "${GLOSSARY_SIZE_SOURCE}" == "per_paper_expanded" ]]; then
        printf '%s/expanded_glossary__%s_gs10000.json\n' "${EXPANDED_PER_PAPER_GLOSSARY_DIR}" "${paper}"
      else
        printf '%s\n' "${GLOBAL_GS10K_GLOSSARY}"
      fi
      ;;
    *) echo "[ERROR] Unsupported glossary kind: ${kind}" >&2; return 2 ;;
  esac
}

glossary_tag_for() {
  local kind="$1" paper="$2"
  basename "$(runtime_glossary_for "${kind}" "${paper}")" .json
}

aggregate_glossary_tag_for() {
  local kind="$1"
  case "${kind}" in
    raw)
      printf '%s\n' "paper_extracted_raw"
      ;;
    gs1k|gs10k)
      if [[ "${GLOSSARY_SIZE_SOURCE}" == "global_acl_gt_union" ]]; then
        glossary_tag_for "${kind}" "${PAPERS%% *}"
      else
        printf '%s_%s\n' "${kind}" "${GLOSSARY_SIZE_SOURCE}" | tr -c 'A-Za-z0-9_.-' '_'
      fi
      ;;
    *) echo "[ERROR] Unsupported glossary kind: ${kind}" >&2; return 2 ;;
  esac
}

output_base_for_method() {
  printf '%s/%s\n' "${OUTPUT_ROOT}" "$(method_label "$1")"
}

index_path_for_glossary() {
  local glossary_path="$1"
  local model_hash glossary_hash glossary_tag
  model_hash="$(printf '%s' "${RAG_MODEL_PATH}" | sha1sum | awk '{print substr($1,1,10)}')"
  glossary_hash="$(printf '%s' "${glossary_path}" | sha1sum | awk '{print substr($1,1,10)}')"
  glossary_tag="$(basename "${glossary_path}" .json)"
  printf '%s/lh1b88kw_%s__%s__%s__maxsim.pt\n' "${INDEX_CACHE_DIR}" "${model_hash}" "${glossary_tag}" "${glossary_hash}"
}

check_hf_model() {
  local model_dir="$1"
  if [[ ! -f "${model_dir}/config.json" ]]; then
    echo "[ERROR] Missing HF model config: ${model_dir}" >&2
    return 3
  fi
  local shard_count
  shard_count="$(find "${model_dir}" -maxdepth 1 -name 'model-*.safetensors' | wc -l | tr -d ' ')"
  if [[ "${shard_count}" != "15" ]]; then
    echo "[ERROR] Expected 15 safetensor shards under ${model_dir}, found ${shard_count}" >&2
    return 3
  fi
}

prepare_inputs() {
  "${PYTHON_BIN}" - "${INPUT_ROOT}" "${DATA_ROOT}" "${DEV_SOURCE}" "${DEV_AUDIO_YAML}" "${DEV_SOURCE_TEXT}" "${REF_DIR}" "${LANGS}" "${PAPERS}" "${SMOKE_MAX_SENTENCES}" <<'PY'
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
ref_dir = Path(sys.argv[6])
langs = [x for x in sys.argv[7].split() if x]
papers = [x for x in sys.argv[8].split() if x]
smoke_n = int(sys.argv[9])

audio_entries = yaml.safe_load(dev_audio_yaml.read_text(encoding="utf-8"))
if not isinstance(audio_entries, list):
    raise SystemExit(f"invalid yaml list: {dev_audio_yaml}")
source_lines = dev_source.read_text(encoding="utf-8").splitlines()
source_texts = dev_source_text.read_text(encoding="utf-8").splitlines()
if len(audio_entries) != len(source_texts):
    raise SystemExit(f"yaml/source_text mismatch: {len(audio_entries)} vs {len(source_texts)}")

remapped = 0

def normalize_audio_path(path_s: str) -> str:
    global remapped
    path_s = str(path_s or "").strip()
    if not path_s:
        return path_s
    if Path(path_s).exists():
        return path_s
    local_wav = data_root / "dev" / "full_wavs" / Path(path_s).name
    if local_wav.exists():
        remapped += 1
        return str(local_wav)
    if path_s.startswith("/mnt/data/"):
        alt = "/mnt/taurus/data/" + path_s[len("/mnt/data/"):]
        if Path(alt).exists():
            remapped += 1
            return alt
    return path_s

paper_source = {}
for line in source_lines:
    normalized = normalize_audio_path(line)
    paper_source[Path(normalized).stem] = normalized

by_paper_indices = defaultdict(list)
for idx, item in enumerate(audio_entries):
    if isinstance(item, dict) and "wav" in item:
        item["wav"] = normalize_audio_path(str(item.get("wav", "")))
    paper = Path(str(item.get("wav", ""))).stem
    by_paper_indices[paper].append(idx)

for paper in papers:
    if paper not in paper_source:
        raise SystemExit(f"paper missing from dev.source: {paper}")
    if paper not in by_paper_indices:
        raise SystemExit(f"paper missing from dev.yaml: {paper}")

for lang in langs:
    ref_path = ref_dir / f"ACL.6060.dev.en-xx.{lang}.txt"
    refs = ref_path.read_text(encoding="utf-8").splitlines()
    if len(refs) != len(audio_entries):
        raise SystemExit(f"ref/yaml mismatch for {lang}: {len(refs)} vs {len(audio_entries)}")
    for paper in papers:
        indices = by_paper_indices[paper]
        out_dir = input_root / "full" / lang / paper
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "source.list").write_text(paper_source[paper] + "\n", encoding="utf-8")
        (out_dir / "target.list").write_text(" ".join(refs[i].strip() for i in indices) + "\n", encoding="utf-8")
        (out_dir / "source_text.txt").write_text("\n".join(source_texts[i] for i in indices) + "\n", encoding="utf-8")
        (out_dir / "ref.txt").write_text("\n".join(refs[i] for i in indices) + "\n", encoding="utf-8")
        (out_dir / "audio.yaml").write_text(
            yaml.safe_dump([audio_entries[i] for i in indices], allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

smoke_indices = list(range(min(smoke_n, len(audio_entries))))
smoke_wav_dir = input_root / "smoke_wavs"
smoke_wav_dir.mkdir(parents=True, exist_ok=True)
for lang in langs:
    ref_path = ref_dir / f"ACL.6060.dev.en-xx.{lang}.txt"
    refs = ref_path.read_text(encoding="utf-8").splitlines()
    out_dir = input_root / "smoke" / lang / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_paths = []
    smoke_yaml = []
    for out_idx, src_idx in enumerate(smoke_indices):
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
    (out_dir / "target.list").write_text("\n".join(refs[i] for i in smoke_indices) + "\n", encoding="utf-8")
    (out_dir / "source_text.txt").write_text("\n".join(source_texts[i] for i in smoke_indices) + "\n", encoding="utf-8")
    (out_dir / "ref.txt").write_text("\n".join(refs[i] for i in smoke_indices) + "\n", encoding="utf-8")
    (out_dir / "audio.yaml").write_text(yaml.safe_dump(smoke_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")

print(f"[PREP] inputs={input_root} remapped_audio_paths={remapped}")
PY
}

prebuild_index() {
  local glossary_path="$1" gpu_group="$2"
  local index_path log_prefix
  index_path="$(index_path_for_glossary "${glossary_path}")"
  if [[ -s "${index_path}" ]]; then
    echo "[INDEX] exists ${index_path}"
    return 0
  fi
  if [[ ! -f "${glossary_path}" ]]; then
    echo "[ERROR] Missing runtime glossary for index build: ${glossary_path}" >&2
    return 3
  fi
  log_prefix="${LOG_ROOT}/index_$(basename "${glossary_path}" .json)_${RUN_STAMP}"
  echo "[INDEX] build ${index_path}"
  CUDA_VISIBLE_DEVICES="${gpu_group}" CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${gpu_group}" \
    "${PYTHON_BIN}" "${ROOT_DIR}/retriever/gigaspeech/build_maxsim_index.py" \
      --model-path "${RAG_MODEL_PATH}" \
      --glossary-path "${glossary_path}" \
      --output-path "${index_path}" \
      --device cuda:0 \
      --text-lora-rank "${RAG_TEXT_LORA_R}" \
      > "${log_prefix}.out" 2> "${log_prefix}.err"
}

prebuild_indices() {
  local gpu_group="$1" method lang paper kind glossary_path key
  declare -A seen=()
  for paper in ${PAPERS}; do
    for kind in ${GLOSSARY_KINDS}; do
      glossary_path="$(runtime_glossary_for "${kind}" "${paper}")"
      key="${glossary_path}"
      if [[ -z "${seen[$key]+x}" ]]; then
        seen[$key]=1
        prebuild_index "${glossary_path}" "${gpu_group}"
      fi
    done
  done
  for method in ${METHODS}; do
    for lang in ${LANGS}; do
      check_hf_model "$(model_for "${method}" "${lang}")"
    done
  done
}

write_tasks() {
  : > "${TASK_FILE}"
  local method lang lm kind paper
  if [[ "${MODE}" == "smoke" ]]; then
    method="${SMOKE_METHOD_OVERRIDE:-no_tmsft}"
    lang="${SMOKE_LANG_OVERRIDE:-zh}"
    lm="${SMOKE_LM_OVERRIDE:-2}"
    kind="${SMOKE_GLOSSARY_KIND_OVERRIDE:-raw}"
    paper="${SMOKE_PAPER_OVERRIDE:-smoke}"
    printf "%s\t%s\t%s\t%s\t%s\n" "${method}" "${lang}" "${lm}" "${kind}" "${paper}" >> "${TASK_FILE}"
    return 0
  fi
  for method in ${METHODS}; do
    for lang in ${LANGS}; do
      for lm in ${LMS}; do
        for kind in ${GLOSSARY_KINDS}; do
          for paper in ${PAPERS}; do
            printf "%s\t%s\t%s\t%s\t%s\n" "${method}" "${lang}" "${lm}" "${kind}" "${paper}" >> "${TASK_FILE}"
          done
        done
      done
    done
  done
}

run_one_task() {
  local method="$1" lang="$2" lm="$3" kind="$4" paper="$5" gpu_group="$6"
  local output_base density model_path runtime_glossary raw_glossary glossary_tag index_path in_dir output_dir log_prefix port_seed vllm_port paper_tag

  output_base="$(output_base_for_method "${method}")"
  density="$(density_for_method "${method}")"
  model_path="$(model_for "${method}" "${lang}")"
  if [[ "${paper}" == "smoke" ]]; then
    runtime_glossary="$(runtime_glossary_for "${kind}" "2022.acl-long.268")"
    raw_glossary="$(raw_glossary_for_paper "2022.acl-long.268")"
    in_dir="${INPUT_ROOT}/smoke/${lang}/smoke"
    paper_tag="smoke"
  else
    runtime_glossary="$(runtime_glossary_for "${kind}" "${paper}")"
    raw_glossary="$(raw_glossary_for_paper "${paper}")"
    in_dir="${INPUT_ROOT}/full/${lang}/${paper}"
    paper_tag="${paper}"
  fi
  glossary_tag="$(basename "${runtime_glossary}" .json)"
  index_path="$(index_path_for_glossary "${runtime_glossary}")"
  output_dir="${output_base}/${lang}/d${density}_lm${lm}_k${RAG_TOP_K}_th${RAG_SCORE_THRESHOLD}_g${glossary_tag}_pp${paper_tag}"
  log_prefix="${LOG_ROOT}/$(method_label "${method}")_${lang}_lm${lm}_${kind}_${paper_tag}_${RUN_STAMP}"
  port_seed="$(printf '%s' "${method}_${lang}_${lm}_${kind}_${paper_tag}_${RUN_STAMP}" | cksum | awk '{print $1}')"
  vllm_port="$((24000 + (port_seed % 20000)))"

  if [[ "${SKIP_COMPLETED}" == "1" && -s "${output_dir}/eval_results.tsv" && -s "${output_dir}/instances.log" ]]; then
    echo "[SKIP] ${method} ${lang} lm=${lm} ${kind} ${paper_tag}"
    return 0
  fi
  for p in "${model_path}" "${runtime_glossary}" "${raw_glossary}" "${index_path}" "${in_dir}/source.list" "${in_dir}/target.list" "${in_dir}/ref.txt" "${in_dir}/source_text.txt" "${in_dir}/audio.yaml"; do
    if [[ ! -e "${p}" ]]; then
      echo "[ERROR] Missing task input: ${p}" >&2
      return 3
    fi
  done

  echo "[RUN] method=${method} lang=${lang} lm=${lm} kind=${kind} paper=${paper_tag} gpu=${gpu_group}"
  CUDA_VISIBLE_DEVICES="${gpu_group}" CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${gpu_group}" \
    VLLM_PORT="${vllm_port}" \
    MODEL_NAME_OVERRIDE="${model_path}" \
    RAG_MODEL_PATH_OVERRIDE="${RAG_MODEL_PATH}" \
    RAG_LORA_R_OVERRIDE="${RAG_LORA_R}" \
    RAG_TEXT_LORA_R_OVERRIDE="${RAG_TEXT_LORA_R}" \
    OUTPUT_BASE_OVERRIDE="${output_base}" \
    EVAL_MODE_OVERRIDE="acl6060" \
    LANG_CODE_OVERRIDE="${lang}" \
    GLOSSARY_PATH_OVERRIDE="${runtime_glossary}" \
    INDEX_PATH_OVERRIDE="${index_path}" \
    EVAL_GLOSSARY_PATH_OVERRIDE="${raw_glossary}" \
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
    EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR_OVERRIDE}" \
    GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE}" \
    VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN_OVERRIDE}" \
    DENSITY_TAG="${density}" \
    PAPER_ID_TAG="${paper_tag}" \
    VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME="VLLM_OBJ_aclpp_${method}_${lang}_${lm}_${kind}_${paper_tag}_${RUN_STAMP}_${RANDOM}" \
    bash "${EVAL_SCRIPT}" > "${log_prefix}.out" 2> "${log_prefix}.err"
}

aggregate_one_setting() {
  local method="$1" lang="$2" lm="$3" kind="$4"
  local output_base density aggregate_glossary_tag out_dir paper paper_tag_pairs
  output_base="$(output_base_for_method "${method}")"
  density="$(density_for_method "${method}")"
  aggregate_glossary_tag="$(aggregate_glossary_tag_for "${kind}")"
  out_dir="${output_base}/${lang}/d${density}_lm${lm}_k${RAG_TOP_K}_th${RAG_SCORE_THRESHOLD}_g${aggregate_glossary_tag}"
  mkdir -p "${out_dir}"
  paper_tag_pairs=""
  for paper in ${PAPERS}; do
    paper_tag_pairs+="${paper}:$(glossary_tag_for "${kind}" "${paper}"),"
  done
  paper_tag_pairs="${paper_tag_pairs%,}"
  "${PYTHON_BIN}" - "${output_base}" "${lang}" "${density}" "${lm}" "${RAG_TOP_K}" "${RAG_SCORE_THRESHOLD}" "${paper_tag_pairs}" "${PAPERS}" "${out_dir}/eval_results.tsv" <<'PY'
import csv
import sys
from pathlib import Path

base, lang, density, lm, topk, tau, paper_tag_pairs_s, papers_s, out_tsv = sys.argv[1:10]
papers = [p for p in papers_s.split() if p]
tag_by_paper = {}
for pair in [x for x in paper_tag_pairs_s.split(",") if x]:
    paper, tag = pair.split(":", 1)
    tag_by_paper[paper] = tag
rows = []
for paper in papers:
    glossary_tag = tag_by_paper.get(paper)
    if not glossary_tag:
        raise SystemExit(f"missing glossary tag for paper: {paper}")
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
adopt_sent = sum(i(r, "TERM_ADOPTION_SENTENCES") for r in rows)
real_adopted = sum(i(r, "REAL_TERM_ADOPTED") for r in rows)
real_total = sum(i(r, "REAL_TERM_ADOPT_TOTAL") for r in rows)
real_sent = sum(i(r, "REAL_TERM_ADOPT_SENTENCES") for r in rows)
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
    "mode": "acl_paper_extracted_fixed_raw_aggregate",
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
    "TERM_ADOPTION_SENTENCES": adopt_sent,
    "TERM_ADOPTION_MICRO": adopted / adopt_total if adopt_total else 0.0,
    "REAL_TERM_ADOPT": real_adopted / real_total if real_total else 0.0,
    "REAL_TERM_ADOPTED": real_adopted,
    "REAL_TERM_ADOPT_TOTAL": real_total,
    "REAL_TERM_ADOPT_SENTENCES": real_sent,
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

log_one_setting() {
  local method="$1" lang="$2" lm="$3" kind="$4"
  [[ "${WANDB_LOG_OVERRIDE}" == "1" ]] || return 0
  local output_base density glossary_tag method_lbl model_path
  output_base="$(output_base_for_method "${method}")"
  density="$(density_for_method "${method}")"
  glossary_tag="$(aggregate_glossary_tag_for "${kind}")"
  method_lbl="$(method_label "${method}")"
  model_path="$(model_for "${method}" "${lang}")"
  HOME="${WANDB_HOME:-${HOME}}" \
  WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${HOME}/.config/wandb}" \
  "${WANDB_PYTHON}" "${WANDB_LOGGER}" \
    --project simuleval_eval \
    --run-name "${method_lbl}__acl_paper_extracted__${lang}__lm${lm}__${kind}__tau073" \
    --experiment-family "acl_paper_extracted_main" \
    --data-tag "acl_paper_extracted_fixed_raw_${lang}" \
    --task-tag "eval" \
    --notes-file "${NOTES_FILE}" \
    --extra-tags "variant:${method_lbl}_${lang}_${kind}_lm${lm}" "${WANDB_COMPUTE_TAG_OVERRIDE}" "tau:tau073" "glossary:${kind}" "lang:${lang}" "denom:paper_raw" \
    --density "${density}" \
    --rag-top-k "${RAG_TOP_K}" \
    --rag-score-threshold "${RAG_SCORE_THRESHOLD}" \
    --output-base "${output_base}" \
    --lang-code "${lang}" \
    --latency-multipliers "${lm}" \
    --glossary-tag "${glossary_tag}" \
    --model-name "${model_path}" \
    --rag-model-path "${RAG_MODEL_PATH}" \
    --verdict "ACL paper-extracted aggregate: ${method_lbl}, lang=${lang}, lm=${lm}, runtime glossary=${kind}, fixed per-paper raw denominator, tau=0.73."
}

run_worker() {
  local worker_idx="$1" worker_count="$2" gpu_group="$3"
  local line_no=0 method lang lm kind paper
  while IFS=$'\t' read -r method lang lm kind paper; do
    if (( line_no % worker_count == worker_idx )); then
      run_one_task "${method}" "${lang}" "${lm}" "${kind}" "${paper}" "${gpu_group}"
    fi
    line_no=$((line_no + 1))
  done < "${TASK_FILE}"
}

aggregate_and_log_all() {
  [[ "${MODE}" == "full" ]] || return 0
  local method lang lm kind
  for method in ${METHODS}; do
    for lang in ${LANGS}; do
      for lm in ${LMS}; do
        for kind in ${GLOSSARY_KINDS}; do
          aggregate_one_setting "${method}" "${lang}" "${lm}" "${kind}"
          log_one_setting "${method}" "${lang}" "${lm}" "${kind}"
        done
      done
    done
  done
}

main() {
  export TMPDIR="${EVAL_TMPDIR_OVERRIDE}"
  export TMP="${EVAL_TMPDIR_OVERRIDE}"
  export TEMP="${EVAL_TMPDIR_OVERRIDE}"

  prepare_inputs
  write_tasks
  echo "[INFO] task_file=${TASK_FILE} tasks=$(wc -l < "${TASK_FILE}")"
  if [[ "${MODE}" == "prepare" ]]; then
    echo "[DONE] prepare only: inputs=${INPUT_ROOT} task_file=${TASK_FILE}"
    return 0
  fi
  if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
    echo "[ERROR] Unsupported MODE=${MODE}; expected prepare, smoke, or full" >&2
    exit 2
  fi

  IFS=';' read -r -a GPU_GROUPS <<< "${GPU_GROUPS_CSV}"
  if (( ${#GPU_GROUPS[@]} < 1 )); then
    echo "[ERROR] Empty GPU_GROUPS_CSV" >&2
    exit 2
  fi
  if (( MAX_PARALLEL > ${#GPU_GROUPS[@]} )); then
    MAX_PARALLEL="${#GPU_GROUPS[@]}"
  fi

  prebuild_indices "${GPU_GROUPS[0]}"

  local worker_count="${MAX_PARALLEL}"
  local pids=()
  for (( wi=0; wi<worker_count; wi++ )); do
    run_worker "${wi}" "${worker_count}" "${GPU_GROUPS[$wi]}" \
      > "${LOG_ROOT}/worker${wi}_${RUN_STAMP}.out" \
      2> "${LOG_ROOT}/worker${wi}_${RUN_STAMP}.err" &
    pids+=("$!")
  done

  local failed=0 pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if (( failed )); then
    echo "[ERROR] one or more workers failed" >&2
    exit 1
  fi

  aggregate_and_log_all
  echo "[ALL DONE] output_root=${OUTPUT_ROOT} logs=${LOG_ROOT}"
}

main "$@"
