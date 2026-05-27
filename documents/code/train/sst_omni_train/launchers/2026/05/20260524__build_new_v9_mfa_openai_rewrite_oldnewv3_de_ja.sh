#!/usr/bin/env bash
# Build clean de/ja New V9 Speech LLM SFT data from MFA + OpenAI rewrite.
#
# Mainline lineage:
#   clean raw SFT -> MFA source candidates -> OpenAI exact span + uncommon target
#   rewrite -> old-new_v3 TCM term_map -> no-GT-zero -> assistant <term> tags.
#
# This intentionally does NOT use v13/sourceexact/timeline data and does NOT
# derive GT labels from legacy term_map entries.
set -euo pipefail

if [[ -d /mnt/taurus/home/jiaxuanluo/InfiniSST ]]; then
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
else
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
fi

CONDA_PREFIX_OVERRIDE="${CONDA_PREFIX_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv}"
if [[ -d "${CONDA_PREFIX_OVERRIDE}" ]]; then
  export PATH="${CONDA_PREFIX_OVERRIDE}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX_OVERRIDE}/lib:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/mnt/gemini/data1/jiaxuanluo/huggingface_cache}"
export TORCH_HOME="${TORCH_HOME:-/mnt/gemini/data1/jiaxuanluo/torch_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/gemini/data1/jiaxuanluo/xdg_cache}"

PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-python3}"
DATA_ROOT="${DATA_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo}"
OUT_ROOT="${OUT_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo}"
OUT_NAME_PREFIX="${OUT_NAME_PREFIX_OVERRIDE:-speech_llm_new_v9_mfa_openai_rewrite_oldnewv3}"
LANGS="${LANGS_OVERRIDE:-de ja}"
DEV_ROWS="${DEV_ROWS_OVERRIDE:-355}"
NUM_SHARDS="${NUM_SHARDS_OVERRIDE:-2}"
STAGE0_SHARDS="${STAGE0_SHARDS_OVERRIDE:-${NUM_SHARDS}}"
GPU_DEVICES_CSV="${GPU_DEVICES_CSV_OVERRIDE:-${CUDA_VISIBLE_DEVICES:-0,1}}"
FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"
DRY_RUN="${DRY_RUN_OVERRIDE:-0}"
CACHE_ONLY="${CACHE_ONLY_OVERRIDE:-0}"
LIMIT_ROWS="${LIMIT_ROWS_OVERRIDE:-0}"
MAX_API_ITEMS="${MAX_API_ITEMS_OVERRIDE:-0}"
OPENAI_API_KEY_FILE="${OPENAI_API_KEY_FILE_OVERRIDE:-${HOME}/.config/openai_api_key}"
RESUME_FROM_STAGE0="${RESUME_FROM_STAGE0_OVERRIDE:-0}"
RESUME_FROM_STAGE1_SOURCECOPY="${RESUME_FROM_STAGE1_SOURCECOPY_OVERRIDE:-0}"
SOURCE_CANDIDATE_JSONL="${SOURCE_CANDIDATE_JSONL_OVERRIDE:-}"
PREFILTER_REFERENCE_SPANS_FROM_INPUT_TERM_MAP="${PREFILTER_REFERENCE_SPANS_FROM_INPUT_TERM_MAP_OVERRIDE:-0}"
USE_LEGACY_TERMMAP_SPAN_AS_TARGET_WITHOUT_OPENAI="${USE_LEGACY_TERMMAP_SPAN_AS_TARGET_WITHOUT_OPENAI_OVERRIDE:-0}"

SOURCE_GLOSSARY="${SOURCE_GLOSSARY_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample100000.json}"
MFA_SQLITE="${MFA_SQLITE_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/gigaspeech_mfa_index/gigaspeech_mfa_index.sqlite}"
TEXTGRID_DIR="${TEXTGRID_DIR_OVERRIDE:-/mnt/taurus/data/siqiouyang/datasets/gigaspeech/textgrids}"
TCM_MODEL_PATH="${TCM_MODEL_PATH_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_ntcm_final_v3_n64_p1n4_s2000_aries-8gpu_smallest_dense_smoke4650_best.pt}"

MFA_OPENAI_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/build_mfa_openai_rewrite_gt_terms.py"
GEN_RETRIEVER_SCRIPT="${ROOT_DIR}/documents/code/data_pre/hard_negative_jsonl_for_speech_llm/generate_termmap_maxsim.py"
TRANSLATE_RETRIEVER_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/translate_retriever_results_openai.py"
REBUILD_TERMMAP_SCRIPT="${ROOT_DIR}/documents/code/data_pre/hard_negative_jsonl_for_speech_llm/rebuild_termmap.py"
ZERO_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/zero_no_gt_termmap_chunks.py"
WRAP_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/wrap_assistant_term_targets.py"

RETRIEVAL_DENSITY="${RETRIEVAL_DENSITY_OVERRIDE:-9}"
MAX_TOP_K="${MAX_TOP_K_OVERRIDE:-20}"
MAX_TERMS="${MAX_TERMS_OVERRIDE:-20}"
TAU="${TAU_OVERRIDE:-0.75}"
OPENAI_MODEL="${OPENAI_MODEL_OVERRIDE:-gpt-4.1-mini}"
OPENAI_REWRITE_BATCH="${OPENAI_REWRITE_BATCH_OVERRIDE:-4}"
OPENAI_TRANSLATE_BATCH="${OPENAI_TRANSLATE_BATCH_OVERRIDE:-32}"
OPENAI_TRANSLATE_WORKERS="${OPENAI_TRANSLATE_WORKERS_OVERRIDE:-1}"
BATCH_ACROSS_CONVERSATIONS="${BATCH_ACROSS_CONVERSATIONS_OVERRIDE:-1}"
AUDIO_ENCODE_BATCH="${AUDIO_ENCODE_BATCH_OVERRIDE:-64}"
MAX_BATCH_SECONDS="${MAX_BATCH_SECONDS_OVERRIDE:-120}"
RAG_FEATURE_EXTRACTOR_MODEL_ID="${RAG_FEATURE_EXTRACTOR_MODEL_ID_OVERRIDE:-openai/whisper-large-v3}"
EXCLUDE_SOURCE_TOKENS="${EXCLUDE_SOURCE_TOKENS_OVERRIDE:-a,an,the,this,that,these,those,his,her,hers,him,he,she,it,its,they,them,their,theirs,you,your,yours,we,our,ours,i,me,my,mine,myself,yourself,himself,herself,itself,ourselves,yourselves,themselves,what,which,who,whom,whose,someone,somebody,something,anyone,anybody,anything,everyone,everybody,everything,there,here,where,when,why,how,all,any,some,one,two}"

for p in \
  "${SOURCE_GLOSSARY}" "${MFA_SQLITE}" "${TEXTGRID_DIR}" "${TCM_MODEL_PATH}" \
  "${MFA_OPENAI_SCRIPT}" "${GEN_RETRIEVER_SCRIPT}" "${TRANSLATE_RETRIEVER_SCRIPT}" \
  "${REBUILD_TERMMAP_SCRIPT}" "${ZERO_SCRIPT}" "${WRAP_SCRIPT}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

if [[ -z "${OPENAI_API_KEY:-}" && -s "${OPENAI_API_KEY_FILE}" ]]; then
  export OPENAI_API_KEY="$(<"${OPENAI_API_KEY_FILE}")"
fi
if [[ "${DRY_RUN}" != "1" && "${CACHE_ONLY}" != "1" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ERROR] OPENAI_API_KEY is required unless DRY_RUN=1 or CACHE_ONLY=1." >&2
  echo "[ERROR] Export it in the job environment or write it to OPENAI_API_KEY_FILE=${OPENAI_API_KEY_FILE}." >&2
  exit 5
fi

IFS=',' read -r -a GPU_DEVICES <<< "${GPU_DEVICES_CSV}"
if (( ${#GPU_DEVICES[@]} < 1 )); then
  echo "[ERROR] GPU_DEVICES_CSV is empty." >&2
  exit 2
fi

cd "${ROOT_DIR}"
echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] LANGS=${LANGS}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo "[INFO] OUT_NAME_PREFIX=${OUT_NAME_PREFIX}"
echo "[INFO] SOURCE_CANDIDATE_JSONL=${SOURCE_CANDIDATE_JSONL}"
echo "[INFO] PREFILTER_REFERENCE_SPANS_FROM_INPUT_TERM_MAP=${PREFILTER_REFERENCE_SPANS_FROM_INPUT_TERM_MAP}"
echo "[INFO] USE_LEGACY_TERMMAP_SPAN_AS_TARGET_WITHOUT_OPENAI=${USE_LEGACY_TERMMAP_SPAN_AS_TARGET_WITHOUT_OPENAI}"
echo "[INFO] SOURCE_GLOSSARY=${SOURCE_GLOSSARY}"
echo "[INFO] MFA_SQLITE=${MFA_SQLITE}"
echo "[INFO] TEXTGRID_DIR=${TEXTGRID_DIR}"
echo "[INFO] TCM_MODEL_PATH=${TCM_MODEL_PATH}"
echo "[INFO] GPU_DEVICES_CSV=${GPU_DEVICES_CSV} NUM_SHARDS=${NUM_SHARDS}"
echo "[INFO] STAGE0_SHARDS=${STAGE0_SHARDS}"
echo "[INFO] TAU=${TAU} RETRIEVAL_DENSITY=${RETRIEVAL_DENSITY} MAX_TOP_K=${MAX_TOP_K} MAX_TERMS=${MAX_TERMS}"
echo "[INFO] BATCH_ACROSS_CONVERSATIONS=${BATCH_ACROSS_CONVERSATIONS}"
echo "[INFO] AUDIO_ENCODE_BATCH=${AUDIO_ENCODE_BATCH} MAX_BATCH_SECONDS=${MAX_BATCH_SECONDS}"
echo "[INFO] OPENAI_TRANSLATE_BATCH=${OPENAI_TRANSLATE_BATCH} OPENAI_TRANSLATE_WORKERS=${OPENAI_TRANSLATE_WORKERS}"
echo "[INFO] DRY_RUN=${DRY_RUN} CACHE_ONLY=${CACHE_ONLY} LIMIT_ROWS=${LIMIT_ROWS} MAX_API_ITEMS=${MAX_API_ITEMS}"
echo "[INFO] RESUME_FROM_STAGE0=${RESUME_FROM_STAGE0}"
echo "[INFO] RESUME_FROM_STAGE1_SOURCECOPY=${RESUME_FROM_STAGE1_SOURCECOPY}"

for lang in ${LANGS}; do
  input_jsonl="${DATA_ROOT}/train_s_${lang}_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl"
  audio_map_tsv="${DATA_ROOT}/train_xl_case_robust_asr-filtered_${lang}_metricx-qe3.0_align.tsv"
  out_dir="${OUT_ROOT}/${OUT_NAME_PREFIX}_${lang}_20260524"
  log_dir="${out_dir}/logs"
  shard_dir="${out_dir}/shards"
  mkdir -p "${out_dir}" "${log_dir}" "${shard_dir}"

  stage0_jsonl="${out_dir}/stage0_train_s_${lang}_mfa_openai_rewrite_gt_termmap_none.jsonl"
  stage0_stats="${out_dir}/stage0_mfa_openai_rewrite_gt_stats.json"
  stage0_samples="${out_dir}/stage0_mfa_openai_rewrite_gt_samples.json"
  rewrite_cache="${out_dir}/openai_mfa_rewrite_cache_${lang}.json"

  stage1_sourcecopy="${out_dir}/stage1_train_s_${lang}_oldnewv3_tcm_sourcecopy_retriever_results.jsonl"
  stage1_translated="${out_dir}/stage1_train_s_${lang}_oldnewv3_tcm_translated_retriever_results.jsonl"
  translation_cache="${out_dir}/openai_retriever_translation_cache_${lang}.json"

  stage2_gtbackfill="${out_dir}/stage2_train_s_${lang}_oldnewv3_mfa_openai_termmap_gtbackfill.jsonl"
  stage3_no_gt_zero="${out_dir}/stage3_train_s_${lang}_new_v5_no_gt_zero_mfa_openai_oldnewv3.jsonl"
  final_train="${out_dir}/train_s_${lang}_new_v9_mfa_openai_rewrite_oldnewv3.jsonl"
  final_dev="${out_dir}/dev_s_${lang}_new_v9_mfa_openai_rewrite_oldnewv3_first${DEV_ROWS}.jsonl"
  summary_json="${out_dir}/new_v9_mfa_openai_rewrite_oldnewv3_${lang}_summary.json"

  for p in "${input_jsonl}" "${audio_map_tsv}"; do
    if [[ ! -s "${p}" ]]; then
      echo "[ERROR] ${lang}: missing input: ${p}" >&2
      exit 3
    fi
  done
  if [[ -n "${SOURCE_CANDIDATE_JSONL}" && ! -s "${SOURCE_CANDIDATE_JSONL}" ]]; then
    echo "[ERROR] ${lang}: missing SOURCE_CANDIDATE_JSONL=${SOURCE_CANDIDATE_JSONL}" >&2
    exit 3
  fi

  outputs=(
    "${stage0_jsonl}" "${stage0_stats}" "${stage0_samples}"
    "${stage1_sourcecopy}" "${stage1_translated}"
    "${stage2_gtbackfill}" "${stage3_no_gt_zero}"
    "${final_train}" "${final_dev}" "${summary_json}"
    "${out_dir}/stage1_retriever_translation_stats.json"
    "${out_dir}/stage3_no_gt_zero_stats.json"
    "${out_dir}/stage3_no_gt_zero_samples.json"
    "${out_dir}/stage4_assistant_termtag_stats.json"
    "${out_dir}/stage4_assistant_termtag_samples.json"
  )
  if [[ "${RESUME_FROM_STAGE0}" == "1" ]]; then
    outputs=(
      "${stage1_sourcecopy}" "${stage1_translated}"
      "${stage2_gtbackfill}" "${stage3_no_gt_zero}"
      "${final_train}" "${final_dev}" "${summary_json}"
      "${out_dir}/stage1_retriever_translation_stats.json"
      "${out_dir}/stage3_no_gt_zero_stats.json"
      "${out_dir}/stage3_no_gt_zero_samples.json"
      "${out_dir}/stage4_assistant_termtag_stats.json"
      "${out_dir}/stage4_assistant_termtag_samples.json"
    )
  fi
  if [[ "${RESUME_FROM_STAGE1_SOURCECOPY}" == "1" ]]; then
    outputs=(
      "${stage1_translated}"
      "${stage2_gtbackfill}" "${stage3_no_gt_zero}"
      "${final_train}" "${final_dev}" "${summary_json}"
      "${out_dir}/stage1_retriever_translation_stats.json"
      "${out_dir}/stage3_no_gt_zero_stats.json"
      "${out_dir}/stage3_no_gt_zero_samples.json"
      "${out_dir}/stage4_assistant_termtag_stats.json"
      "${out_dir}/stage4_assistant_termtag_samples.json"
    )
  fi
  if [[ "${FORCE_OVERWRITE}" == "1" ]]; then
    rm -f "${outputs[@]}" "${shard_dir}"/stage0_shard_* "${shard_dir}"/stage1_retriever_shard_* "${log_dir}"/stage1_* "${log_dir}"/stage2_* "${log_dir}"/stage3_* "${log_dir}"/stage4_*
  else
    for p in "${outputs[@]}"; do
      if [[ -e "${p}" ]]; then
        echo "[ERROR] ${lang}: output exists: ${p}" >&2
        echo "[ERROR] Set FORCE_OVERWRITE=1 only for intentional reruns." >&2
        exit 4
      fi
    done
  fi

  if [[ "${RESUME_FROM_STAGE0}" == "1" || "${RESUME_FROM_STAGE1_SOURCECOPY}" == "1" ]]; then
    echo "[STAGE A/B] ${lang}: resume from existing Stage0"
    for p in "${stage0_jsonl}" "${stage0_stats}" "${stage0_samples}"; do
      if [[ ! -s "${p}" ]]; then
        echo "[ERROR] ${lang}: RESUME_FROM_STAGE0=1 but missing ${p}" >&2
        exit 7
      fi
    done
    "${PYTHON_BIN}" - "${stage0_stats}" "${DRY_RUN}" <<'PY'
import json, sys
p = sys.argv[1]
dry_run = sys.argv[2] == "1"
s = json.load(open(p, encoding="utf-8"))
kept = int(s.get("apply", {}).get("kept_gt_terms", 0))
if kept <= 0 and not dry_run:
    raise SystemExit(f"[ERROR] Stage0 kept no GT terms: {p}")
print(json.dumps({
    "stage0_kept_gt_terms": kept,
    "chunks_with_gt_rate": s.get("apply", {}).get("chunks_with_gt_rate"),
    "avg_gt_terms_per_chunk": s.get("apply", {}).get("avg_gt_terms_per_chunk"),
}, ensure_ascii=False))
PY
  else
  echo "[STAGE A/B] ${lang}: MFA source candidates + OpenAI exact span rewrite"
  "${PYTHON_BIN}" - "${input_jsonl}" "${shard_dir}" "${STAGE0_SHARDS}" "${LIMIT_ROWS}" <<'PY'
import sys
from pathlib import Path
src = Path(sys.argv[1])
out = Path(sys.argv[2])
n = int(sys.argv[3])
limit = int(sys.argv[4])
outs = [open(out / f"stageA_input_shard_{i:02d}_of_{n:02d}.jsonl", "w", encoding="utf-8") for i in range(n)]
counts = [0] * n
try:
    with src.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            if limit > 0 and sum(counts) >= limit:
                break
            j = idx % n
            outs[j].write(line)
            counts[j] += 1
finally:
    for fh in outs:
        fh.close()
print({"stage0_input_shards": counts})
PY

  pids=()
  for ((shard=0; shard<STAGE0_SHARDS; shard++)); do
    in_shard="${shard_dir}/stageA_input_shard_$(printf '%02d' "${shard}")_of_$(printf '%02d' "${STAGE0_SHARDS}").jsonl"
    out_shard="${shard_dir}/stageA_output_shard_$(printf '%02d' "${shard}")_of_$(printf '%02d' "${STAGE0_SHARDS}").jsonl"
    stats_shard="${shard_dir}/stageA_stats_shard_$(printf '%02d' "${shard}")_of_$(printf '%02d' "${STAGE0_SHARDS}").json"
    samples_shard="${shard_dir}/stageA_samples_shard_$(printf '%02d' "${shard}")_of_$(printf '%02d' "${STAGE0_SHARDS}").json"
    cache_shard="${out_dir}/openai_mfa_rewrite_cache_${lang}_shard$(printf '%02d' "${shard}").json"
    log_file="${log_dir}/stage0_mfa_openai_rewrite_${lang}_shard$(printf '%02d' "${shard}").log"
    source_candidate_args=()
    if [[ -n "${SOURCE_CANDIDATE_JSONL}" ]]; then
      source_candidate_args+=(--source-candidate-jsonl "${SOURCE_CANDIDATE_JSONL}")
    fi
    legacy_prefilter_args=()
    if [[ "${PREFILTER_REFERENCE_SPANS_FROM_INPUT_TERM_MAP}" == "1" ]]; then
      legacy_prefilter_args+=(--prefilter-reference-spans-from-input-term-map)
    fi
    if [[ "${USE_LEGACY_TERMMAP_SPAN_AS_TARGET_WITHOUT_OPENAI}" == "1" ]]; then
      legacy_prefilter_args+=(--use-legacy-termmap-span-as-target-without-openai)
    fi
    rewrite_args=(
      --input-jsonl "${in_shard}"
      --output-jsonl "${out_shard}"
      --stats-json "${stats_shard}"
      --sample-json "${samples_shard}"
      --glossary-json "${SOURCE_GLOSSARY}"
      "${source_candidate_args[@]}"
      "${legacy_prefilter_args[@]}"
      --openai-cache-json "${cache_shard}"
      --sqlite-index "${MFA_SQLITE}"
      --textgrid-dir "${TEXTGRID_DIR}"
      --audio-map-tsv "${audio_map_tsv}"
      --lang-code "${lang}"
      --chunk-assignment-policy overlap
      --max-candidates-per-chunk "${MAX_CANDIDATES_PER_CHUNK_OVERRIDE:-16}"
      --openai-model "${OPENAI_MODEL}"
      --openai-batch-size "${OPENAI_REWRITE_BATCH}"
      --max-api-items "${MAX_API_ITEMS}"
      --exclude-source-tokens "${EXCLUDE_SOURCE_TOKENS}"
      --require-text-boundaries
    )
    if [[ "${DRY_RUN}" == "1" ]]; then
      rewrite_args+=(--dry-run)
    fi
    if [[ "${CACHE_ONLY}" == "1" ]]; then
      rewrite_args+=(--cache-only)
    fi
    (
      "${PYTHON_BIN}" "${MFA_OPENAI_SCRIPT}" "${rewrite_args[@]}"
    ) > "${log_file}" 2>&1 &
    pids+=("$!")
    echo "[INFO] ${lang}: stageA shard=${shard}/${STAGE0_SHARDS} pid=${pids[-1]} log=${log_file}"
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  : > "${stage0_jsonl}"
  for ((shard=0; shard<STAGE0_SHARDS; shard++)); do
    out_shard="${shard_dir}/stageA_output_shard_$(printf '%02d' "${shard}")_of_$(printf '%02d' "${STAGE0_SHARDS}").jsonl"
    if [[ ! -s "${out_shard}" ]]; then
      echo "[ERROR] ${lang}: missing Stage A shard output: ${out_shard}" >&2
      exit 6
    fi
    cat "${out_shard}" >> "${stage0_jsonl}"
  done
  "${PYTHON_BIN}" - "${shard_dir}" "${STAGE0_SHARDS}" "${stage0_stats}" "${stage0_samples}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path
shard_dir = Path(sys.argv[1])
n = int(sys.argv[2])
stats_out = Path(sys.argv[3])
samples_out = Path(sys.argv[4])

def merge_counter(dst, src):
    for k, v in src.items():
        if isinstance(v, (int, float)):
            dst[k] += v

summary = {"collect": Counter(), "apply": Counter(), "glossary": None, "shards": []}
collect_samples = []
apply_samples = []
for shard in range(n):
    p = shard_dir / f"stageA_stats_shard_{shard:02d}_of_{n:02d}.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    summary["shards"].append(str(p))
    merge_counter(summary["collect"], s.get("collect", {}))
    merge_counter(summary["apply"], s.get("apply", {}))
    if summary["glossary"] is None:
        summary["glossary"] = s.get("glossary", {})
    sp = shard_dir / f"stageA_samples_shard_{shard:02d}_of_{n:02d}.json"
    if sp.exists():
        sample = json.loads(sp.read_text(encoding="utf-8"))
        collect_samples.extend(sample.get("collect_samples", [])[:20])
        apply_samples.extend(sample.get("apply_samples", [])[:20])
collect = dict(summary["collect"])
apply = dict(summary["apply"])
apply["chunks_with_gt_rate"] = apply.get("chunks_with_gt", 0) / apply.get("chunks_total", 1) if apply.get("chunks_total", 0) else 0.0
apply["avg_gt_terms_per_chunk"] = apply.get("kept_gt_terms", 0) / apply.get("chunks_total", 1) if apply.get("chunks_total", 0) else 0.0
out = {
    "glossary": summary["glossary"],
    "collect": collect,
    "apply": apply,
    "stage0_shards": summary["shards"],
}
stats_out.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
samples_out.write_text(json.dumps({
    "collect_samples": collect_samples[:80],
    "apply_samples": apply_samples[:80],
}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"merged_stage0": stats_out.as_posix(), "tasks_total": collect.get("tasks_total", 0), "kept_gt_terms": apply.get("kept_gt_terms", 0)}, ensure_ascii=False))
PY

  "${PYTHON_BIN}" - "${stage0_stats}" "${DRY_RUN}" <<'PY'
import json, sys
p = sys.argv[1]
dry_run = sys.argv[2] == "1"
s = json.load(open(p, encoding="utf-8"))
kept = int(s.get("apply", {}).get("kept_gt_terms", 0))
if kept <= 0 and not dry_run:
    raise SystemExit(f"[ERROR] Stage0 kept no GT terms: {p}")
print(json.dumps({
    "stage0_kept_gt_terms": kept,
    "chunks_with_gt_rate": s.get("apply", {}).get("chunks_with_gt_rate"),
    "avg_gt_terms_per_chunk": s.get("apply", {}).get("avg_gt_terms_per_chunk"),
}, ensure_ascii=False))
PY
  fi

  if [[ "${RESUME_FROM_STAGE1_SOURCECOPY}" == "1" ]]; then
    echo "[STAGE C1] ${lang}: resume from existing retriever sourcecopy"
    if [[ ! -s "${stage1_sourcecopy}" ]]; then
      echo "[ERROR] ${lang}: RESUME_FROM_STAGE1_SOURCECOPY=1 but missing ${stage1_sourcecopy}" >&2
      exit 8
    fi
  else
  echo "[STAGE C0] ${lang}: split Stage0 for retriever shards"
  "${PYTHON_BIN}" - "${stage0_jsonl}" "${shard_dir}" "${NUM_SHARDS}" <<'PY'
import sys
from pathlib import Path
src = Path(sys.argv[1])
out = Path(sys.argv[2])
n = int(sys.argv[3])
outs = [open(out / f"stage0_shard_{i:02d}_of_{n:02d}.jsonl", "w", encoding="utf-8") for i in range(n)]
counts = [0] * n
try:
    with src.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            j = idx % n
            outs[j].write(line)
            counts[j] += 1
finally:
    for fh in outs:
        fh.close()
print({"shards": counts})
PY

  echo "[STAGE C1] ${lang}: old-new_v3 TCM retriever term_map candidates"
  pids=()
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    gpu="${GPU_DEVICES[$((shard % ${#GPU_DEVICES[@]}))]}"
    in_shard="${shard_dir}/stage0_shard_$(printf '%02d' "${shard}")_of_$(printf '%02d' "${NUM_SHARDS}").jsonl"
    out_shard="${shard_dir}/stage1_retriever_shard_$(printf '%02d' "${shard}")_of_$(printf '%02d' "${NUM_SHARDS}").jsonl"
    log_file="${log_dir}/stage1_retriever_${lang}_shard$(printf '%02d' "${shard}").log"
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      retriever_extra_args=()
      if [[ "${BATCH_ACROSS_CONVERSATIONS}" == "1" ]]; then
        retriever_extra_args+=(--batch_across_conversations)
      fi
      "${PYTHON_BIN}" "${GEN_RETRIEVER_SCRIPT}" \
        --cleaned_jsonl "${in_shard}" \
        --glossary_json "${SOURCE_GLOSSARY}" \
        --model_path "${TCM_MODEL_PATH}" \
        --output_jsonl "${out_shard}" \
        --device cuda:0 \
        --retrieval_density "${RETRIEVAL_DENSITY}" \
        --top_k_mode duration_sec_cap \
        --max_top_k "${MAX_TOP_K}" \
        --score_threshold "${TAU}" \
        --target_lang "${lang}" \
        --allow_copy_translation_fallback \
        --audio_encode_batch "${AUDIO_ENCODE_BATCH}" \
        --max_batch_seconds "${MAX_BATCH_SECONDS}" \
        --rag_feature_extractor_model_id "${RAG_FEATURE_EXTRACTOR_MODEL_ID}" \
        "${retriever_extra_args[@]}"
    ) > "${log_file}" 2>&1 &
    pids+=("$!")
    echo "[INFO] ${lang}: shard=${shard} gpu=${gpu} pid=${pids[-1]} log=${log_file}"
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  echo "[STAGE C1] ${lang}: merge retriever shards"
  : > "${stage1_sourcecopy}"
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    out_shard="${shard_dir}/stage1_retriever_shard_$(printf '%02d' "${shard}")_of_$(printf '%02d' "${NUM_SHARDS}").jsonl"
    if [[ ! -s "${out_shard}" ]]; then
      echo "[ERROR] Missing retriever shard output: ${out_shard}" >&2
      exit 6
    fi
    cat "${out_shard}" >> "${stage1_sourcecopy}"
  done
  fi

  echo "[STAGE C2] ${lang}: translate retriever filler terms with OpenAI"
  translate_args=(
    --input-jsonl "${stage1_sourcecopy}"
    --output-jsonl "${stage1_translated}"
    --stats-json "${out_dir}/stage1_retriever_translation_stats.json"
    --openai-cache-json "${translation_cache}"
    --lang-code "${lang}"
    --openai-model "${OPENAI_MODEL}"
    --openai-batch-size "${OPENAI_TRANSLATE_BATCH}"
    --openai-workers "${OPENAI_TRANSLATE_WORKERS}"
    --max-api-items "${MAX_API_ITEMS}"
  )
  if (( LIMIT_ROWS > 0 )); then
    translate_args+=(--max-rows "${LIMIT_ROWS}")
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    translate_args+=(--dry-run)
  fi
  if [[ "${CACHE_ONLY}" == "1" ]]; then
    translate_args+=(--cache-only)
  fi
  "${PYTHON_BIN}" "${TRANSLATE_RETRIEVER_SCRIPT}" "${translate_args[@]}" 2>&1 | tee "${log_dir}/stage1_translate_retriever_${lang}.log"

  echo "[STAGE C3] ${lang}: rebuild term_map with clean GT backfill"
  "${PYTHON_BIN}" "${REBUILD_TERMMAP_SCRIPT}" \
    --input_jsonl "${stage1_translated}" \
    --output_jsonl "${stage2_gtbackfill}" \
    --termmap_mode tcm_filtered_with_gt_backfill \
    --max_terms "${MAX_TERMS}" \
    --target_lang "${lang}" \
    --seed "${REBUILD_SEED_OVERRIDE:-42}" 2>&1 | tee "${log_dir}/stage2_rebuild_termmap_${lang}.log"

  echo "[STAGE D1] ${lang}: no-GT-zero"
  "${PYTHON_BIN}" "${ZERO_SCRIPT}" \
    --input-jsonl "${stage2_gtbackfill}" \
    --output-jsonl "${stage3_no_gt_zero}" \
    --stats-json "${out_dir}/stage3_no_gt_zero_stats.json" \
    --sample-json "${out_dir}/stage3_no_gt_zero_samples.json" \
    --missing-gt-policy error

  echo "[STAGE D2] ${lang}: exact assistant <term> tags with boundary-only repair"
  "${PYTHON_BIN}" "${WRAP_SCRIPT}" \
    --input-jsonl "${stage3_no_gt_zero}" \
    --output-jsonl "${final_train}" \
    --stats-json "${out_dir}/stage4_assistant_termtag_stats.json" \
    --sample-json "${out_dir}/stage4_assistant_termtag_samples.json" \
    --lang-code "${lang}" \
    --tag-template '<term>{translation}</term>' \
    --min-target-chars 2 \
    --max-tags-per-row 16 \
    --missing-gt-policy error \
    --exclude-source-tokens "${EXCLUDE_SOURCE_TOKENS}" \
    --exact-require-text-boundaries \
    --enable-local-rewrite \
    --rewrite-boundary-only \
    --rewrite-delay-boundary-prefix \
    --rewrite-delay-boundary-min-prefix-chars 2 \
    --rewrite-require-text-boundaries \
    --sample-count 200

  echo "[DEV] ${lang}: first ${DEV_ROWS} rows for SFT validation set"
  head -n "${DEV_ROWS}" "${final_train}" > "${final_dev}"

  echo "[VALIDATE] ${lang}: final JSONL structural checks"
  "${PYTHON_BIN}" - "${lang}" "${input_jsonl}" "${stage0_stats}" "${out_dir}/stage1_retriever_translation_stats.json" "${out_dir}/stage3_no_gt_zero_stats.json" "${out_dir}/stage4_assistant_termtag_stats.json" "${final_train}" "${summary_json}" <<'PY'
import json
import re
import sys
from pathlib import Path

lang, source_jsonl, stage0_stats, trans_stats, zero_stats, tag_stats, final_train, summary = sys.argv[1:]

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

latin_alnum_re = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]")

def is_latin_alnum(ch):
    return bool(ch) and bool(latin_alnum_re.fullmatch(ch))

def tag_cuts_latin_word(text):
    start = 0
    while True:
        open_pos = text.find("<term>", start)
        if open_pos < 0:
            return False
        close_pos = text.find("</term>", open_pos + len("<term>"))
        if close_pos < 0:
            return True
        inner_start = open_pos + len("<term>")
        inner_end = close_pos
        if inner_start >= inner_end:
            return True
        before = text[open_pos - 1] if open_pos > 0 else ""
        first = text[inner_start]
        last = text[inner_end - 1]
        after_idx = close_pos + len("</term>")
        after = text[after_idx] if after_idx < len(text) else ""
        if is_latin_alnum(before) and is_latin_alnum(first):
            return True
        if is_latin_alnum(after) and is_latin_alnum(last):
            return True
        start = after_idx

rows = 0
malformed = 0
latin_cut = 0
gt_terms = 0
term_map_terms = 0
gt_in_map = 0
no_gt_chunks = 0
no_gt_zero_chunks = 0
sample_rows = []

with Path(final_train).open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        rows += 1
        obj = json.loads(line)
        messages = obj.get("messages") or []
        gt_by_chunk = obj.get("gt_terms_by_chunk") or []
        audio_user_idx = [i for i, m in enumerate(messages) if m.get("role") == "user" and "<audio>" in str(m.get("content") or "")]
        if len(audio_user_idx) != len(obj.get("audios") or []):
            raise SystemExit(f"[ERROR] audio/user count mismatch row={rows}")
        for m in messages:
            if m.get("role") != "assistant":
                continue
            text = str(m.get("content") or "")
            if text.count("<term>") != text.count("</term>"):
                malformed += 1
            if tag_cuts_latin_word(text):
                latin_cut += 1
        for chunk_i, gt_terms_chunk in enumerate(gt_by_chunk):
            gt_terms += len(gt_terms_chunk)
            user_msg = messages[audio_user_idx[chunk_i]]
            content = str(user_msg.get("content") or "")
            term_lines = [x for x in content.splitlines() if "=" in x]
            term_map_terms += len(term_lines)
            if not gt_terms_chunk:
                no_gt_chunks += 1
                if "term_map:NONE" in content:
                    no_gt_zero_chunks += 1
            term_keys = {str(x.split("=", 1)[0]).strip().casefold() for x in term_lines}
            for gt in gt_terms_chunk:
                if str(gt.get("term") or "").strip().casefold() in term_keys:
                    gt_in_map += 1
        if len(sample_rows) < 20:
            sample_rows.append({
                "utter_id": obj.get("utter_id"),
                "source_chunk_mfa_text_by_chunk": obj.get("source_chunk_mfa_text_by_chunk", [])[:4],
                "gt_terms_by_chunk": gt_by_chunk[:4],
                "first_user_term_map": next((str(messages[i].get("content") or "") for i in audio_user_idx if "term_map:" in str(messages[i].get("content") or "")), "")[:800],
                "assistant_tags": [str(m.get("content") or "") for m in messages if m.get("role") == "assistant" and "<term>" in str(m.get("content") or "")][:3],
            })

if malformed or latin_cut:
    raise SystemExit(f"[ERROR] bad tags: malformed={malformed} latin_cut={latin_cut}")

summary_obj = {
    "event": f"new_v9_mfa_openai_rewrite_oldnewv3_{lang}",
    "source_jsonl": source_jsonl,
    "final_train": final_train,
    "stage0": load(stage0_stats),
    "retriever_translation": load(trans_stats),
    "no_gt_zero": load(zero_stats),
    "assistant_tag": load(tag_stats),
    "final_validation": {
        "rows": rows,
        "gt_terms": gt_terms,
        "term_map_terms": term_map_terms,
        "gt_in_term_map_rate": gt_in_map / gt_terms if gt_terms else 0.0,
        "no_gt_zero_rate": no_gt_zero_chunks / no_gt_chunks if no_gt_chunks else 0.0,
        "malformed_tag_assistant_messages": malformed,
        "latin_word_cut_tag_messages": latin_cut,
    },
    "manual_inspection_samples": sample_rows,
}
Path(summary).write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary_obj["final_validation"], ensure_ascii=False, indent=2, sort_keys=True))
PY

  echo "[OK] ${lang}: ${final_train}"
  echo "[OK] ${lang}: ${final_dev}"
  echo "[OK] ${lang}: ${summary_json}"
done
