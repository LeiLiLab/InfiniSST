#!/usr/bin/env bash
set -euo pipefail

# ACL paper-extracted union zh rerun.
# A detached supervisor runs all no-RAG origin baselines first, then runs
# new_v9+HN1024 RASST workers for raw/gs1k/gs10k.

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

LAUNCHER_RELATIVE_PATH="documents/code/simuleval/launchers/2026/05/20260524__acl_paper_extracted_union_zh_origin_norag_newv9_hn1024_batch.sh"
REMOTE_ROOT_DIR="${REMOTE_ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
RUN_STAMP="${RUN_STAMP_OVERRIDE:-20260524T1545_aclpp_union_zh_origin_newv9_hn1024_m80}"
MODE="${MODE:-launch}"
PHASE="${PHASE:-both}"
LM="${LM:-}"
HOST_LABEL="${HOST_LABEL:-$(hostname -s)}"
GPU_PAIR="${GPU_PAIR:-}"

CONDA_BASE="${CONDA_BASE:-/mnt/taurus/home/jiaxuanluo/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-spaCyEnv}"
PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-${CONDA_BASE}/envs/${CONDA_ENV_NAME}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export LD_LIBRARY_PATH="${CONDA_BASE}/envs/${CONDA_ENV_NAME}/lib:${LD_LIBRARY_PATH:-}"
export CONDA_PREFIX="${CONDA_BASE}/envs/${CONDA_ENV_NAME}"
export CONDA_DEFAULT_ENV="${CONDA_ENV_NAME}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export MWERSEGMENTER_ROOT="${MWERSEGMENTER_ROOT:-/mnt/taurus/home/jiaxuanluo/mwerSegmenter}"
export PATH="${MWERSEGMENTER_ROOT}:${PATH}"

DATA_ROOT="${DATA_ROOT_OVERRIDE:-/mnt/taurus/data/siqiouyang/datasets/acl6060}"
OUTPUT_ROOT="${OUTPUT_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/acl_paper_extracted_union_zh_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs/acl_paper_extracted_union_zh_${RUN_STAMP}}"
CACHE_ROOT="${CACHE_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/cache/acl_paper_extracted_union_zh_${RUN_STAMP}}"
INPUT_DIR="${INPUT_DIR_OVERRIDE:-${OUTPUT_ROOT}/__inputs__/paper_union/zh/all}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/acl_paper_extracted_union_zh}"
NOTES_FILE="${NOTES_FILE_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260524__acl_paper_extracted_union_zh_origin_norag_newv9_hn1024.md}"

BASELINE_SCRIPT="${BASELINE_SCRIPT_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/rank16/baseline/bypass_simuleval_rank32_iter_0000452_hf_baseline_no_rag_sweep.sh}"
BATCH_RAG_LAUNCHER="${BATCH_RAG_LAUNCHER_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__batched_vllm_rag_eval.sh}"
OFFLINE_EVAL_SCRIPT="${OFFLINE_EVAL_SCRIPT_OVERRIDE:-${ROOT_DIR}/documents/code/offline_sst_eval/offline_streamlaal_eval.py}"
WANDB_LOGGER="${WANDB_LOGGER_OVERRIDE:-${ROOT_DIR}/documents/code/offline_evaluation/wandb_eval_logger.py}"
INDEX_BUILDER="${INDEX_BUILDER_OVERRIDE:-${ROOT_DIR}/retriever/gigaspeech/build_maxsim_index.py}"
INDEX_CACHE_TOOL="${INDEX_CACHE_TOOL_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/tools/maxsim_index_cache_key.py}"

ORIGIN_MODEL="${ORIGIN_MODEL_OVERRIDE:-/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_origin-bsz4}"
NEW_V9_MODEL="${NEW_V9_MODEL_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/slm_exports/speech_llm_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_zh_r32a64_tp2_taurus8/keep1.0_r32/v0-20260524-062743-hf}"
HN1024_CKPT="${HN1024_CKPT_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"

RAW_GLOSSARY="${RAW_GLOSSARY_OVERRIDE:-${ROOT_DIR}/retriever/gigaspeech/data_pre/acl6060_paper_extracted_union_raw_zh.json}"
GS1K_GLOSSARY="${GS1K_GLOSSARY_OVERRIDE:-${ROOT_DIR}/retriever/gigaspeech/data_pre/acl6060_paper_extracted_union_gs1000_zh.json}"
GS10K_GLOSSARY="${GS10K_GLOSSARY_OVERRIDE:-${ROOT_DIR}/retriever/gigaspeech/data_pre/acl6060_paper_extracted_union_gs10000_zh.json}"

LMS=(1 2 3 4)
if [[ -n "${LMS_OVERRIDE:-}" ]]; then
  # shellcheck disable=SC2206
  LMS=(${LMS_OVERRIDE})
fi
PAPERS=(
  2022.acl-long.268
  2022.acl-long.367
  2022.acl-long.590
  2022.acl-long.110
  2022.acl-long.117
)
TARGET_LANG="zh"
RAG_TOP_K="${RAG_TOP_K_OVERRIDE:-10}"
RAG_SCORE_THRESHOLD="${RAG_SCORE_THRESHOLD_OVERRIDE:-0.78}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS_OVERRIDE:-80}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN_OVERRIDE:-16384}"
VLLM_TP_SIZE="${VLLM_TP_SIZE_OVERRIDE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}"
MAX_NUM_SEQS="${MAX_NUM_SEQS_OVERRIDE:-5}"
SCHEDULER_BATCH_SIZE="${SCHEDULER_BATCH_SIZE_OVERRIDE:-5}"
MAX_CACHE_SECONDS="${MAX_CACHE_SECONDS_OVERRIDE:-80}"
KEEP_CACHE_SECONDS="${KEEP_CACHE_SECONDS_OVERRIDE:-60}"
TERM_FCR_POLICY_RASST="${TERM_FCR_POLICY_RASST_OVERRIDE:-term_map_source_ref_negative_sentence}"
TERM_FCR_POLICY_BASELINE="${TERM_FCR_POLICY_BASELINE_OVERRIDE:-source_ref_negative_sentence}"
WANDB_PROJECT="${WANDB_PROJECT_OVERRIDE:-simuleval_eval}"
WANDB_FAMILY="${WANDB_FAMILY_OVERRIDE:-acl_paper_extracted_union_zh}"
WANDB_DATA_TAG="${WANDB_DATA_TAG_OVERRIDE:-aclpp_union_zh}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${CACHE_ROOT}" "${INDEX_CACHE_DIR}" "${INPUT_DIR}"

short_tmp_for() {
  local phase="$1" lm="$2" kind="${3:-x}"
  printf '/tmp/jx_aclpp_%s_lm%s_%s' "${phase}" "${lm}" "${kind}"
}

segment_for_lm() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import sys
print(f"{0.96 * float(sys.argv[1]):.2f}")
PY
}

glossary_path_for() {
  case "$1" in
    raw) printf '%s\n' "${RAW_GLOSSARY}" ;;
    gs1k) printf '%s\n' "${GS1K_GLOSSARY}" ;;
    gs10k) printf '%s\n' "${GS10K_GLOSSARY}" ;;
    *) echo "[ERROR] unknown glossary kind: $1" >&2; return 2 ;;
  esac
}

glossary_tag_for() {
  basename "$(glossary_path_for "$1")" .json
}

check_tags() {
  local tag
  for tag in "$@"; do
    if [[ "${#tag}" -lt 1 || "${#tag}" -gt 64 ]]; then
      echo "[ERROR] invalid W&B tag length (${#tag}): ${tag}" >&2
      return 2
    fi
  done
}

count_shards() {
  find "$1" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' '
}

validate_static_paths() {
  local p
  for p in \
    "${BASELINE_SCRIPT}" "${BATCH_RAG_LAUNCHER}" "${OFFLINE_EVAL_SCRIPT}" \
    "${WANDB_LOGGER}" "${INDEX_BUILDER}" "${INDEX_CACHE_TOOL}" "${NOTES_FILE}" \
    "${DATA_ROOT}/dev.source" "${DATA_ROOT}/dev.yaml" \
    "${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.en.txt" \
    "${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.zh.txt" \
    "${RAW_GLOSSARY}" "${GS1K_GLOSSARY}" "${GS10K_GLOSSARY}" "${HN1024_CKPT}"; do
    if [[ ! -s "${p}" && ! -d "${p}" ]]; then
      echo "[ERROR] missing required path: ${p}" >&2
      exit 3
    fi
  done
  for p in "${ORIGIN_MODEL}" "${NEW_V9_MODEL}"; do
    if [[ ! -s "${p}/config.json" ]]; then
      echo "[ERROR] missing model config: ${p}/config.json" >&2
      exit 3
    fi
    if [[ "$(count_shards "${p}")" != "15" ]]; then
      echo "[ERROR] expected 15 safetensor shards under ${p}" >&2
      exit 3
    fi
  done
  check_tags \
    "family:${WANDB_FAMILY}" "task:eval" "data:${WANDB_DATA_TAG}" \
    "method:norag_origin" "method:newv9_hn1024" "glossary:raw" \
    "glossary:gs1k" "glossary:gs10k" "denom:union_raw" "max_tokens:80"
}

prepare_inputs() {
  "${PYTHON_BIN}" - "${INPUT_DIR}" "${DATA_ROOT}" "${PAPERS[@]}" <<'PY'
import sys
from collections import defaultdict
from pathlib import Path

import yaml

out_dir = Path(sys.argv[1])
data_root = Path(sys.argv[2])
papers = sys.argv[3:]
dev_source = data_root / "dev.source"
dev_audio_yaml = data_root / "dev.yaml"
dev_source_text = data_root / "dev/text/txt/ACL.6060.dev.en-xx.en.txt"
ref_path = data_root / "dev/text/txt/ACL.6060.dev.en-xx.zh.txt"

audio_entries = yaml.safe_load(dev_audio_yaml.read_text(encoding="utf-8"))
source_lines = dev_source.read_text(encoding="utf-8").splitlines()
source_texts = dev_source_text.read_text(encoding="utf-8").splitlines()
refs = ref_path.read_text(encoding="utf-8").splitlines()
if len(audio_entries) != len(source_texts) or len(audio_entries) != len(refs):
    raise SystemExit(
        f"input length mismatch: yaml={len(audio_entries)} src_text={len(source_texts)} refs={len(refs)}"
    )

def normalize_audio_path(path_s: str) -> str:
    path_s = str(path_s or "").strip()
    if not path_s:
        return path_s
    if path_s.startswith("/mnt/data/"):
        alt = "/mnt/taurus/data/" + path_s[len("/mnt/data/"):]
        if Path(alt).exists():
            return alt
    if Path(path_s).exists():
        return path_s
    local_wav = data_root / "dev" / "full_wavs" / Path(path_s).name
    if local_wav.exists():
        return str(local_wav)
    return path_s

paper_source = {}
for line in source_lines:
    normalized = normalize_audio_path(line)
    paper_source[Path(normalized).stem] = normalized

by_paper = defaultdict(list)
entries = []
for idx, item in enumerate(audio_entries):
    item = dict(item)
    if "wav" in item:
        item["wav"] = normalize_audio_path(str(item.get("wav", "")))
    entries.append(item)
    by_paper[Path(str(item.get("wav", ""))).stem].append(idx)

source_rows = []
target_rows = []
sentence_entries = []
sentence_refs = []
sentence_sources = []
sample_map_rows = ["instance_index\tpaper_id\tsentence_count"]
for instance_idx, paper in enumerate(papers):
    if paper not in paper_source:
        raise SystemExit(f"paper missing from dev.source: {paper}")
    indices = by_paper.get(paper) or []
    if not indices:
        raise SystemExit(f"paper missing from dev.yaml: {paper}")
    source_rows.append(paper_source[paper])
    target_rows.append(" ".join(refs[i].strip() for i in indices))
    sentence_entries.extend(entries[i] for i in indices)
    sentence_refs.extend(refs[i] for i in indices)
    sentence_sources.extend(source_texts[i] for i in indices)
    sample_map_rows.append(f"{instance_idx}\t{paper}\t{len(indices)}")

out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "source.list").write_text("\n".join(source_rows) + "\n", encoding="utf-8")
(out_dir / "target.list").write_text("\n".join(target_rows) + "\n", encoding="utf-8")
(out_dir / "source_text.txt").write_text("\n".join(sentence_sources) + "\n", encoding="utf-8")
(out_dir / "ref.txt").write_text("\n".join(sentence_refs) + "\n", encoding="utf-8")
(out_dir / "audio.yaml").write_text(
    yaml.safe_dump(sentence_entries, allow_unicode=True, sort_keys=False),
    encoding="utf-8",
)
(out_dir / "paper_map.tsv").write_text("\n".join(sample_map_rows) + "\n", encoding="utf-8")
print(
    f"[PREP] input_dir={out_dir} source_rows={len(source_rows)} "
    f"sentence_rows={len(sentence_entries)} papers={','.join(papers)}"
)
PY
}

validate_inputs() {
  local source_rows target_rows ref_rows text_rows yaml_rows
  source_rows="$(wc -l < "${INPUT_DIR}/source.list" | tr -d ' ')"
  target_rows="$(wc -l < "${INPUT_DIR}/target.list" | tr -d ' ')"
  ref_rows="$(wc -l < "${INPUT_DIR}/ref.txt" | tr -d ' ')"
  text_rows="$(wc -l < "${INPUT_DIR}/source_text.txt" | tr -d ' ')"
  yaml_rows="$("${PYTHON_BIN}" - "${INPUT_DIR}/audio.yaml" <<'PY'
import sys, yaml
data = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(len(data))
PY
)"
  if [[ "${source_rows}" != "5" || "${target_rows}" != "5" || "${ref_rows}" != "468" || "${text_rows}" != "468" || "${yaml_rows}" != "468" ]]; then
    echo "[ERROR] bad combined input counts: source=${source_rows} target=${target_rows} ref=${ref_rows} text=${text_rows} yaml=${yaml_rows}" >&2
    exit 3
  fi
  while IFS= read -r wav_path; do
    [[ -n "${wav_path}" ]] || continue
    if [[ ! -r "${wav_path}" ]]; then
      echo "[ERROR] unreadable source wav: ${wav_path}" >&2
      exit 3
    fi
  done < "${INPUT_DIR}/source.list"
}

resolve_index() {
  local glossary_path="$1" glossary_tag="$2"
  "${PYTHON_BIN}" "${INDEX_CACHE_TOOL}" resolve \
    --model-path "${HN1024_CKPT}" \
    --glossary-path "${glossary_path}" \
    --builder-script "${INDEX_BUILDER}" \
    --cache-dir "${INDEX_CACHE_DIR}" \
    --glossary-tag "${glossary_tag}" \
    --text-lora-rank 128 \
    --text-lora-alpha 256
}

prebuild_one_index() {
  local kind="$1" glossary_path glossary_tag resolve_out
  glossary_path="$(glossary_path_for "${kind}")"
  glossary_tag="$(glossary_tag_for "${kind}")"
  resolve_out="$(resolve_index "${glossary_path}" "${glossary_tag}")"
  eval "${resolve_out}"
  if [[ -s "${INDEX_PATH}" ]]; then
    echo "[INDEX] reuse kind=${kind} path=${INDEX_PATH}"
    return 0
  fi
  echo "[INDEX] build kind=${kind} path=${INDEX_PATH}"
  mkdir -p "$(dirname "${INDEX_PATH}")"
  CUDA_VISIBLE_DEVICES="${INDEX_BUILD_GPU_OVERRIDE:-2}" \
    "${PYTHON_BIN}" "${INDEX_BUILDER}" \
      --model-path "${HN1024_CKPT}" \
      --glossary-path "${glossary_path}" \
      --output-path "${INDEX_PATH}" \
      --device cuda:0 \
      --text-lora-rank 128 \
      --text-lora-alpha 256
  "${PYTHON_BIN}" "${INDEX_CACHE_TOOL}" finalize \
    --model-path "${HN1024_CKPT}" \
    --glossary-path "${glossary_path}" \
    --builder-script "${INDEX_BUILDER}" \
    --index-path "${INDEX_PATH}" \
    --manifest-path "${INDEX_MANIFEST_PATH}" \
    --glossary-tag "${glossary_tag}" \
    --text-lora-rank 128 \
    --text-lora-alpha 256
}

prebuild_indices() {
  prebuild_one_index raw
  prebuild_one_index gs1k
  prebuild_one_index gs10k
}

baseline_output_dir() {
  local lm="$1" model_short raw_tag segment
  model_short="$(basename "${ORIGIN_MODEL}")"
  raw_tag="$(basename "${RAW_GLOSSARY}" .json)"
  segment="$(segment_for_lm "${lm}")"
  printf '%s/origin_norag/zh/%s_g%s_cs%s_hs0.48_lm%s_k210_k110_th0p0\n' \
    "${OUTPUT_ROOT}" "${model_short}" "${raw_tag}" "${segment}" "${lm}"
}

mirror_baseline_for_logger() {
  local lm="$1" source_dir="$2" raw_tag mirror_dir
  raw_tag="$(basename "${RAW_GLOSSARY}" .json)"
  mirror_dir="${OUTPUT_ROOT}/origin_norag/zh/daclpp_origin_norag_m80_lm${lm}_k0_th0_g${raw_tag}"
  mkdir -p "${mirror_dir}"
  cp "${source_dir}/eval_results.tsv" "${mirror_dir}/eval_results.tsv"
  if [[ -s "${source_dir}/eval_results.log" ]]; then
    cp "${source_dir}/eval_results.log" "${mirror_dir}/eval_results.log"
  fi
}

log_baseline_wandb() {
  local lm="$1" raw_tag
  raw_tag="$(basename "${RAW_GLOSSARY}" .json)"
  HOME="${WANDB_HOME:-/mnt/taurus/home/jiaxuanluo}" \
  WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_HOME:-/mnt/taurus/home/jiaxuanluo}/.config/wandb}" \
  "${PYTHON_BIN}" "${WANDB_LOGGER}" \
    --project "${WANDB_PROJECT}" \
    --run-name "origin_norag__aclpp_union__zh__lm${lm}__raw__m80" \
    --experiment-family "${WANDB_FAMILY}" \
    --data-tag "${WANDB_DATA_TAG}" \
    --task-tag eval \
    --notes-file "${NOTES_FILE}" \
    --extra-tags "method:norag_origin" "glossary:raw" "lang:zh" "denom:union_raw" "max_tokens:80" "compute:${HOST_LABEL}" \
    --density "aclpp_origin_norag_m80" \
    --rag-top-k 0 \
    --rag-score-threshold 0 \
    --output-base "${OUTPUT_ROOT}/origin_norag" \
    --lang-code zh \
    --latency-multipliers "${lm}" \
    --glossary-tag "${raw_tag}" \
    --model-name "${ORIGIN_MODEL}" \
    --verdict "ACL paper-extracted union zh baseline: origin/no_tmsft no-RAG, lm=${lm}, fixed raw union denominator, max_new_tokens=80."
}

run_baseline_lm() {
  local lm="$1" output_dir tmpdir
  if [[ -z "${GPU_PAIR}" ]]; then
    echo "[ERROR] GPU_PAIR is required for baseline worker" >&2
    exit 2
  fi
  tmpdir="$(short_tmp_for baseline "${lm}")"
  mkdir -p "${tmpdir}" "${tmpdir}/torchinductor" "${tmpdir}/triton"
  export TMPDIR="${tmpdir}"
  export TMP="${tmpdir}"
  export TEMP="${tmpdir}"
  export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor_${HOST_LABEL}_lm${lm}_baseline"
  export TRITON_CACHE_DIR="${CACHE_ROOT}/triton_${HOST_LABEL}_lm${lm}_baseline"
  export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${CACHE_ROOT}/hf/transformers}"

  echo "[BASELINE] start lm=${lm} host=${HOST_LABEL} gpus=${GPU_PAIR}"
  ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
  DATA_ROOT_OVERRIDE="${DATA_ROOT}" \
  CONDA_PREFIX_OVERRIDE="${CONDA_PREFIX}" \
  GLOSSARY_PATHS_OVERRIDE="${RAW_GLOSSARY}" \
  SRC_LIST_OVERRIDE="${INPUT_DIR}/source.list" \
  TGT_LIST_OVERRIDE="${INPUT_DIR}/target.list" \
  OUTPUT_BASE_OVERRIDE="${OUTPUT_ROOT}/origin_norag" \
  MODEL_NAME_OVERRIDE="${ORIGIN_MODEL}" \
  LANG_CODE_OVERRIDE=zh \
  LATENCY_MULTIPLIERS_OVERRIDE="${lm}" \
  RAG_K2_VALUES_OVERRIDE=10 \
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${GPU_PAIR}" \
  GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_BASELINE_OVERRIDE:-0.80}" \
  MAX_CACHE_SECONDS_OVERRIDE="${MAX_CACHE_SECONDS}" \
  KEEP_CACHE_SECONDS_OVERRIDE="${KEEP_CACHE_SECONDS}" \
  MAX_NEW_TOKENS_OVERRIDE="${MAX_NEW_TOKENS}" \
  VLLM_TP_SIZE_OVERRIDE="${VLLM_TP_SIZE}" \
  VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN}" \
  VLLM_LIMIT_AUDIO_OVERRIDE="${VLLM_LIMIT_AUDIO_BASELINE_OVERRIDE:-64}" \
  VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
  RESUME_MODE=0 \
  CLEAN_OUTPUT_DIR_OVERRIDE=1 \
  BACKUP_PARTIAL_RUNS=1 \
  bash "${BASELINE_SCRIPT}"

  output_dir="$(baseline_output_dir "${lm}")"
  if [[ ! -s "${output_dir}/instances.log" ]]; then
    echo "[ERROR] baseline missing instances.log: ${output_dir}" >&2
    exit 4
  fi

  "${PYTHON_BIN}" "${OFFLINE_EVAL_SCRIPT}" \
    --mode acl6060 \
    --instances-log "${output_dir}/instances.log" \
    --lang-code zh \
    --source-file "${INPUT_DIR}/source_text.txt" \
    --ref-file "${INPUT_DIR}/ref.txt" \
    --audio-yaml "${INPUT_DIR}/audio.yaml" \
    --glossary-acl6060 "${RAW_GLOSSARY}" \
    --strip-output-tags term \
    --term-fcr-policy "${TERM_FCR_POLICY_BASELINE}" \
    --output-tsv "${output_dir}/eval_results.tsv" \
    --output-log "${output_dir}/eval_results.log" \
    --work-dir "${output_dir}/offline_work"
  if [[ ! -s "${output_dir}/eval_results.tsv" ]]; then
    echo "[ERROR] baseline missing eval_results.tsv: ${output_dir}" >&2
    exit 4
  fi
  mirror_baseline_for_logger "${lm}" "${output_dir}"
  log_baseline_wandb "${lm}"
  echo "[BASELINE] done lm=${lm} output=${output_dir}"
}

log_rasst_wandb() {
  local lm="$1" kind="$2" density="$3" glossary_tag="$4"
  HOME="${WANDB_HOME:-/mnt/taurus/home/jiaxuanluo}" \
  WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_HOME:-/mnt/taurus/home/jiaxuanluo}/.config/wandb}" \
  "${PYTHON_BIN}" "${WANDB_LOGGER}" \
    --project "${WANDB_PROJECT}" \
    --run-name "newv9_hn1024__aclpp_union__zh__lm${lm}__${kind}__tau078__m80" \
    --experiment-family "${WANDB_FAMILY}" \
    --data-tag "${WANDB_DATA_TAG}" \
    --task-tag eval \
    --notes-file "${NOTES_FILE}" \
    --extra-tags "method:newv9_hn1024" "glossary:${kind}" "lang:zh" "denom:union_raw" "tau:tau078" "max_tokens:80" "compute:${HOST_LABEL}" \
    --density "${density}" \
    --rag-top-k "${RAG_TOP_K}" \
    --rag-score-threshold "${RAG_SCORE_THRESHOLD}" \
    --output-base "${OUTPUT_ROOT}/newv9_hn1024" \
    --lang-code zh \
    --latency-multipliers "${lm}" \
    --glossary-tag "${glossary_tag}" \
    --model-name "${NEW_V9_MODEL}" \
    --rag-model-path "${HN1024_CKPT}" \
    --verdict "ACL paper-extracted union zh RASST: new_v9 + HN1024, lm=${lm}, runtime glossary=${kind}, tau=${RAG_SCORE_THRESHOLD}, fixed raw union denominator, max_new_tokens=80."
}

run_rasst_lm_kind() {
  local lm="$1" kind="$2" glossary_path glossary_tag density out_dir runtime tmpdir
  if [[ -z "${GPU_PAIR}" ]]; then
    echo "[ERROR] GPU_PAIR is required for RASST worker" >&2
    exit 2
  fi
  glossary_path="$(glossary_path_for "${kind}")"
  glossary_tag="$(basename "${glossary_path}" .json)"
  density="aclpp_newv9_hn1024_m80_${kind}"
  tmpdir="$(short_tmp_for rasst "${lm}" "${kind}")"
  mkdir -p "${tmpdir}" "${tmpdir}/torchinductor" "${tmpdir}/triton"
  export TMPDIR="${tmpdir}"
  export TMP="${tmpdir}"
  export TEMP="${tmpdir}"
  export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor_${HOST_LABEL}_lm${lm}_${kind}"
  export TRITON_CACHE_DIR="${CACHE_ROOT}/triton_${HOST_LABEL}_lm${lm}_${kind}"
  export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${CACHE_ROOT}/hf/transformers}"

  echo "[RASST] start lm=${lm} kind=${kind} host=${HOST_LABEL} gpus=${GPU_PAIR}"
  RUN_TAG_OVERRIDE="${RUN_STAMP}_lm${lm}_${kind}_${HOST_LABEL}" \
  LANG_CODE_OVERRIDE=zh \
  LMS_OVERRIDE="${lm}" \
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${GPU_PAIR}" \
  VLLM_TP_SIZE_OVERRIDE="${VLLM_TP_SIZE}" \
  MAX_NUM_SEQS_OVERRIDE="${MAX_NUM_SEQS}" \
  SCHEDULER_BATCH_SIZE_OVERRIDE="${SCHEDULER_BATCH_SIZE}" \
  SCHEDULE_MODE_OVERRIDE=round_robin \
  VLLM_ENFORCE_EAGER_OVERRIDE=1 \
  VLLM_ENABLE_PREFIX_CACHING=1 \
  VLLM_LIMIT_AUDIO_OVERRIDE="${VLLM_LIMIT_AUDIO_RASST_OVERRIDE:-128}" \
  MAX_CACHE_SECONDS_OVERRIDE="${MAX_CACHE_SECONDS}" \
  KEEP_CACHE_SECONDS_OVERRIDE="${KEEP_CACHE_SECONDS}" \
  MIN_CACHE_CHUNKS_OVERRIDE=1 \
  MAX_NEW_TOKENS_OVERRIDE="${MAX_NEW_TOKENS}" \
  MAX_NEW_TOKENS_POLICY_OVERRIDE=fixed \
  TEMPERATURE_OVERRIDE=0.6 \
  TOP_P_OVERRIDE=0.95 \
  TOP_K_DECODE_OVERRIDE=20 \
  GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION}" \
  VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN}" \
  VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
  VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}" \
  VLLM_USE_FUSED_MOE_GROUPED_TOPK="${VLLM_USE_FUSED_MOE_GROUPED_TOPK:-0}" \
  MODEL_NAME_OVERRIDE="${NEW_V9_MODEL}" \
  SRC_LIST_OVERRIDE="${INPUT_DIR}/source.list" \
  TGT_LIST_OVERRIDE="${INPUT_DIR}/target.list" \
  SOURCE_TEXT_FILE_OVERRIDE="${INPUT_DIR}/source_text.txt" \
  REF_FILE_OVERRIDE="${INPUT_DIR}/ref.txt" \
  AUDIO_YAML_OVERRIDE="${INPUT_DIR}/audio.yaml" \
  GLOSSARY_PATH_OVERRIDE="${glossary_path}" \
  EVAL_GLOSSARY_PATH_OVERRIDE="${RAW_GLOSSARY}" \
  OUTPUT_BASE_OVERRIDE="${OUTPUT_ROOT}/newv9_hn1024" \
  DENSITY_TAG_OVERRIDE="${density}" \
  GLOSSARY_TAG_OVERRIDE="${glossary_tag}" \
  RAG_SCORE_THRESHOLD_OVERRIDE="${RAG_SCORE_THRESHOLD}" \
  RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE=1.92 \
  RAG_TOP_K_OVERRIDE="${RAG_TOP_K}" \
  RAG_DEVICE_OVERRIDE="${RAG_DEVICE_OVERRIDE:-cuda:0}" \
  RAG_BATCH_RETRIEVAL_OVERRIDE=1 \
  INDEX_BUILD_DEVICE_OVERRIDE="${INDEX_BUILD_DEVICE_OVERRIDE:-cuda:0}" \
  INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}" \
  TERM_MAP_FORMAT_OVERRIDE=plain \
  TERM_FCR_POLICY_OVERRIDE="${TERM_FCR_POLICY_RASST}" \
  STRIP_OUTPUT_TAGS_OVERRIDE=term \
  WANDB_LOG_OVERRIDE=0 \
  NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
  LOG_ROOT_OVERRIDE="${LOG_ROOT}/batch_lm${lm}_${kind}_${HOST_LABEL}" \
  EVAL_TMPDIR_OVERRIDE="${tmpdir}" \
  bash "${BATCH_RAG_LAUNCHER}" \
    > "${LOG_ROOT}/rasst_lm${lm}_${kind}_${HOST_LABEL}.out" \
    2> "${LOG_ROOT}/rasst_lm${lm}_${kind}_${HOST_LABEL}.err"

  out_dir="${OUTPUT_ROOT}/newv9_hn1024/zh/d${density}_lm${lm}_k${RAG_TOP_K}_th${RAG_SCORE_THRESHOLD}_g${glossary_tag}"
  runtime="${out_dir}/runtime_omni_vllm_maxsim_rag_batched_lm${lm}.jsonl"
  for p in "${out_dir}/instances.log" "${out_dir}/eval_results.tsv" "${runtime}"; do
    if [[ ! -s "${p}" ]]; then
      echo "[ERROR] RASST output missing: ${p}" >&2
      exit 4
    fi
  done
  log_rasst_wandb "${lm}" "${kind}" "${density}" "${glossary_tag}"
  echo "[RASST] done lm=${lm} kind=${kind} output=${out_dir}"
}

run_worker() {
  if [[ -z "${LM}" ]]; then
    echo "[ERROR] LM is required in worker mode" >&2
    exit 2
  fi
  validate_static_paths
  validate_inputs
  case "${PHASE}" in
    baseline)
      run_baseline_lm "${LM}"
      ;;
    rasst)
      run_rasst_lm_kind "${LM}" raw
      run_rasst_lm_kind "${LM}" gs1k
      run_rasst_lm_kind "${LM}" gs10k
      ;;
    both)
      run_baseline_lm "${LM}"
      run_rasst_lm_kind "${LM}" raw
      run_rasst_lm_kind "${LM}" gs1k
      run_rasst_lm_kind "${LM}" gs10k
      ;;
    *)
      echo "[ERROR] unsupported PHASE=${PHASE}" >&2
      exit 2
      ;;
  esac
}

worker_host_for_lm() {
  case "$1" in
    1|2|3|4) printf '%s\n' "aries" ;;
    *) return 2 ;;
  esac
}

worker_gpu_for_lm() {
  case "$1" in
    1) printf '%s\n' "0,1" ;;
    2) printf '%s\n' "2,3" ;;
    3) printf '%s\n' "4,5" ;;
    4) printf '%s\n' "6,7" ;;
    *) return 2 ;;
  esac
}

RUN_WORKER_PID=""

run_worker_for_phase() {
  local phase="$1" lm="$2" host gpu log_prefix cmd root_for_host launcher_for_host
  host="$(worker_host_for_lm "${lm}")"
  gpu="$(worker_gpu_for_lm "${lm}")"
  log_prefix="${LOG_ROOT}/${phase}_lm${lm}_${host}"
  if [[ "${host}" == "$(hostname -s)" ]]; then
    root_for_host="${ROOT_DIR}"
  else
    root_for_host="${REMOTE_ROOT_DIR}"
  fi
  launcher_for_host="${root_for_host}/${LAUNCHER_RELATIVE_PATH}"
  cmd="cd ${root_for_host} && ROOT_DIR_OVERRIDE=${root_for_host} REMOTE_ROOT_DIR_OVERRIDE=${REMOTE_ROOT_DIR} RUN_STAMP_OVERRIDE=${RUN_STAMP} MODE=worker PHASE=${phase} LM=${lm} HOST_LABEL=${host} GPU_PAIR=${gpu} bash ${launcher_for_host}"
  echo "[SUPERVISOR] launch phase=${phase} lm=${lm} host=${host} gpu=${gpu}" >&2
  if [[ "${host}" == "$(hostname -s)" ]]; then
    bash -lc "${cmd}" > "${log_prefix}.out" 2> "${log_prefix}.err" &
  else
    ssh -o BatchMode=yes "${host}" "${cmd}" > "${log_prefix}.out" 2> "${log_prefix}.err" &
  fi
  RUN_WORKER_PID="$!"
}

wait_phase() {
  local phase="$1"
  shift
  local failed=0 pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if (( failed )); then
    echo "[ERROR] phase failed: ${phase}" >&2
    exit 1
  fi
}

resource_snapshot() {
  {
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(hostname -s)"
    echo "[df]"
    df -h /mnt/gemini/data1 /mnt/aries/data7 /tmp /dev/shm 2>/dev/null || true
    echo "[nvidia-smi taurus]"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
    echo "[nvidia-smi aries]"
    ssh -o BatchMode=yes -o ConnectTimeout=5 aries 'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits' || true
    echo "[squeue]"
    squeue -u "${USER}" -o "%.18i %.9P %.30j %.8T %.10M %.6D %R" || true
  } | tee "${LOG_ROOT}/resource_snapshot_${MODE}.txt"
}

supervisor() {
  validate_static_paths
  prepare_inputs
  validate_inputs
  resource_snapshot
  prebuild_indices

  local pids=()
  for lm in "${LMS[@]}"; do
    run_worker_for_phase baseline "${lm}"
    pids+=("${RUN_WORKER_PID}")
  done
  wait_phase baseline "${pids[@]}"

  pids=()
  for lm in "${LMS[@]}"; do
    run_worker_for_phase rasst "${lm}"
    pids+=("${RUN_WORKER_PID}")
  done
  wait_phase rasst "${pids[@]}"

  date -u +%Y-%m-%dT%H:%M:%SZ > "${OUTPUT_ROOT}/finished_utc.txt"
  echo "[SUPERVISOR] all done output_root=${OUTPUT_ROOT}"
}

launch_supervisor() {
  validate_static_paths
  prepare_inputs
  validate_inputs
  resource_snapshot
  local top_out="${LOG_ROOT}/supervisor.out"
  local top_err="${LOG_ROOT}/supervisor.err"
  local pid_file="${LOG_ROOT}/supervisor.pid"
  setsid bash -lc "cd '${ROOT_DIR}' && RUN_STAMP_OVERRIDE='${RUN_STAMP}' MODE=supervisor bash '${BASH_SOURCE[0]}'" \
    > "${top_out}" 2> "${top_err}" < /dev/null &
  echo $! > "${pid_file}"
  echo "[LAUNCHED] supervisor_pid=$(cat "${pid_file}")"
  echo "[LAUNCHED] output_root=${OUTPUT_ROOT}"
  echo "[LAUNCHED] logs=${LOG_ROOT}"
}

case "${MODE}" in
  prepare)
    validate_static_paths
    prepare_inputs
    validate_inputs
    ;;
  prebuild_indices)
    validate_static_paths
    prebuild_indices
    ;;
  worker)
    run_worker
    ;;
  supervisor)
    supervisor
    ;;
  launch)
    launch_supervisor
    ;;
  *)
    echo "[ERROR] unsupported MODE=${MODE}" >&2
    exit 2
    ;;
esac
