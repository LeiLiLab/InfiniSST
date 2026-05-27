#!/usr/bin/env bash
# Build De/Ja term-map ablation data:
#   A. old LLM-generated term maps, capped and exact-wrapped
#   B. HN1024 retriever-recalled term maps, capped and exact-wrapped
set -euo pipefail

if [[ -d /mnt/taurus/home/jiaxuanluo/InfiniSST ]]; then
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
else
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
fi

SCRIPT_DIR="${ROOT_DIR}/documents/code/data_pre/hard_negative_jsonl_for_speech_llm"
TRAIN_SRC_DIR="${ROOT_DIR}/documents/code/train/sst_omni_train/src"
GENERATE_SCRIPT="${SCRIPT_DIR}/generate_termmap_maxsim.py"
REBUILD_SCRIPT="${SCRIPT_DIR}/rebuild_termmap.py"
DERIVE_SCRIPT="${TRAIN_SRC_DIR}/derive_gt_terms_from_termmap_matches.py"
CAP_SCRIPT="${TRAIN_SRC_DIR}/cap_embedded_termmap.py"
WRAP_SCRIPT="${TRAIN_SRC_DIR}/wrap_assistant_term_targets.py"

CONDA_PREFIX="${CONDA_PREFIX_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/mnt/gemini/data1/jiaxuanluo/huggingface_cache}"
export TORCH_HOME="${TORCH_HOME:-/mnt/gemini/data1/jiaxuanluo/torch_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/gemini/data1/jiaxuanluo/xdg_cache}"

DATA_ROOT="${DATA_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo}"
OUT_ROOT="${OUT_ROOT_OVERRIDE:-${DATA_ROOT}/speech_llm_deja_termmap_ablation_cap16_exactboundary_20260525}"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-${DATA_ROOT}/logs/deja_termmap_ablation_cap16_exactboundary_20260525}"
LANGS="${LANGS_OVERRIDE:-de ja}"
BRANCHES="${BRANCHES_OVERRIDE:-llmgen retriever}"
DEV_ROWS="${DEV_ROWS_OVERRIDE:-355}"
MAX_TERMS="${MAX_TERMS_OVERRIDE:-16}"
NUM_SHARDS="${NUM_SHARDS_OVERRIDE:-4}"
RETRIEVAL_DENSITY="${RETRIEVAL_DENSITY_OVERRIDE:-9}"
MAX_TOP_K="${MAX_TOP_K_OVERRIDE:-20}"
TAU="${TAU_OVERRIDE:-0.78}"
MAX_CONVERSATIONS="${MAX_CONVERSATIONS_OVERRIDE:-0}"
FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"
RAG_FEATURE_EXTRACTOR_MODEL_ID="${RAG_FEATURE_EXTRACTOR_MODEL_ID_OVERRIDE:-openai/whisper-large-v3}"
HN1024_CKPT="${HN1024_CKPT_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
ALLOCATED_GPU_CSV="${TCM_BUILD_GPU_DEVICES_OVERRIDE:-${CUDA_VISIBLE_DEVICES:-0,2,3,4}}"

EXCLUDE_SOURCE_TOKENS="${EXCLUDE_SOURCE_TOKENS_OVERRIDE:-this,that,these,those,his,her,hers,him,he,she,it,its,they,them,their,theirs,you,your,yours,we,our,ours,i,me,my,mine,myself,yourself,himself,herself,itself,ourselves,yourselves,themselves,what,which,who,whom,whose,someone,somebody,something,anyone,anybody,anything,everyone,everybody,everything}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

for p in "${GENERATE_SCRIPT}" "${REBUILD_SCRIPT}" "${DERIVE_SCRIPT}" "${CAP_SCRIPT}" "${WRAP_SCRIPT}" "${HN1024_CKPT}"; do
  [[ -e "${p}" ]] || { echo "[ERROR] Missing required path: ${p}" >&2; exit 3; }
done

source_jsonl_for_lang() {
  case "$1" in
    de) echo "${DATA_ROOT}/train_s_de_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl" ;;
    ja) echo "${DATA_ROOT}/train_s_ja_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl" ;;
    zh) echo "${DATA_ROOT}/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl" ;;
    *) echo "[ERROR] unsupported lang: $1" >&2; return 2 ;;
  esac
}

glossary_for_lang() {
  case "$1" in
    de) echo "${DATA_ROOT}/glossary_for_de_rate1.0_k20.json" ;;
    ja) echo "${DATA_ROOT}/glossary_for_ja_rate1.0_k20.json" ;;
    zh) echo "${DATA_ROOT}/glossary_for_zh_rate1.0_k20.json" ;;
    *) echo "[ERROR] unsupported lang: $1" >&2; return 2 ;;
  esac
}

contains_word() {
  local needle="$1"
  local haystack="$2"
  for word in ${haystack}; do
    [[ "${word}" == "${needle}" ]] && return 0
  done
  return 1
}

maybe_rm_outputs() {
  if [[ "${FORCE_OVERWRITE}" == "1" ]]; then
    rm -f "$@"
    return
  fi
  for p in "$@"; do
    if [[ -e "${p}" ]]; then
      echo "[ERROR] Output exists: ${p}" >&2
      echo "[ERROR] Set FORCE_OVERWRITE=1 only for an intentional rebuild." >&2
      exit 4
    fi
  done
}

derive_gt_once() {
  local lang="$1"
  local source_jsonl="$2"
  local out_dir="$3"
  local stage0_train="${out_dir}/stage0_${lang}_exact_gt_from_llmgen_termmap.jsonl"
  local stage0_dev="${out_dir}/stage0_${lang}_exact_gt_from_llmgen_termmap_dev_first${DEV_ROWS}.jsonl"
  if [[ -s "${stage0_train}" && -s "${stage0_dev}" && "${FORCE_OVERWRITE}" != "1" ]]; then
    echo "[SKIP] existing derived GT for ${lang}: ${stage0_train}"
    return
  fi
  maybe_rm_outputs \
    "${stage0_train}" "${stage0_dev}" \
    "${out_dir}/stage0_${lang}_exact_gt_stats.json" \
    "${out_dir}/stage0_${lang}_exact_gt_samples.json" \
    "${out_dir}/stage0_${lang}_exact_gt_dev_first${DEV_ROWS}_stats.json" \
    "${out_dir}/stage0_${lang}_exact_gt_dev_first${DEV_ROWS}_samples.json"
  echo "[STAGE] ${lang} derive exact GT terms from embedded LLM-generated term_map"
  python3 "${DERIVE_SCRIPT}" \
    --input-jsonl "${source_jsonl}" \
    --output-jsonl "${stage0_train}" \
    --stats-json "${out_dir}/stage0_${lang}_exact_gt_stats.json" \
    --sample-json "${out_dir}/stage0_${lang}_exact_gt_samples.json" \
    --lang-code "${lang}" \
    --min-target-chars 2 \
    --exclude-source-tokens "${EXCLUDE_SOURCE_TOKENS}" \
    --max-terms-per-chunk "${MAX_TERMS}" \
    --sample-count 200 \
    ${MAX_CONVERSATIONS:+--max-rows "${MAX_CONVERSATIONS}"}
  python3 "${DERIVE_SCRIPT}" \
    --input-jsonl "${source_jsonl}" \
    --output-jsonl "${stage0_dev}" \
    --stats-json "${out_dir}/stage0_${lang}_exact_gt_dev_first${DEV_ROWS}_stats.json" \
    --sample-json "${out_dir}/stage0_${lang}_exact_gt_dev_first${DEV_ROWS}_samples.json" \
    --lang-code "${lang}" \
    --min-target-chars 2 \
    --exclude-source-tokens "${EXCLUDE_SOURCE_TOKENS}" \
    --max-terms-per-chunk "${MAX_TERMS}" \
    --max-rows "${DEV_ROWS}" \
    --sample-count 200
}

wrap_jsonl() {
  local lang="$1"
  local input_jsonl="$2"
  local output_jsonl="$3"
  local stats_json="$4"
  local sample_json="$5"
  python3 "${WRAP_SCRIPT}" \
    --input-jsonl "${input_jsonl}" \
    --output-jsonl "${output_jsonl}" \
    --stats-json "${stats_json}" \
    --sample-json "${sample_json}" \
    --lang-code "${lang}" \
    --tag-template '<term>{translation}</term>' \
    --min-target-chars 2 \
    --max-tags-per-row "${MAX_TERMS}" \
    --missing-gt-policy error \
    --exclude-source-tokens "${EXCLUDE_SOURCE_TOKENS}" \
    --exact-require-text-boundaries \
    --enable-local-rewrite \
    --rewrite-boundary-only \
    --rewrite-delay-boundary-prefix \
    --rewrite-delay-boundary-min-prefix-chars 2 \
    --rewrite-require-text-boundaries \
    --sample-count 200
}

build_llmgen_branch() {
  local lang="$1"
  local out_dir="$2"
  local branch_dir="${out_dir}/llmgen_cap${MAX_TERMS}_exactboundary"
  mkdir -p "${branch_dir}"
  local stage0_train="${out_dir}/stage0_${lang}_exact_gt_from_llmgen_termmap.jsonl"
  local stage0_dev="${out_dir}/stage0_${lang}_exact_gt_from_llmgen_termmap_dev_first${DEV_ROWS}.jsonl"
  local capped_train="${branch_dir}/train_s_${lang}_llmgen_cap${MAX_TERMS}_stage1.jsonl"
  local capped_dev="${branch_dir}/dev_s_${lang}_llmgen_cap${MAX_TERMS}_stage1_first${DEV_ROWS}.jsonl"
  local final_train="${branch_dir}/train_s_${lang}_llmgen_cap${MAX_TERMS}_gttermwrap_exactboundary.jsonl"
  local final_dev="${branch_dir}/dev_s_${lang}_llmgen_cap${MAX_TERMS}_gttermwrap_exactboundary_first${DEV_ROWS}.jsonl"
  maybe_rm_outputs \
    "${capped_train}" "${capped_dev}" "${final_train}" "${final_dev}" \
    "${branch_dir}/train_cap_stats.json" "${branch_dir}/train_cap_samples.json" \
    "${branch_dir}/dev_cap_stats.json" "${branch_dir}/dev_cap_samples.json" \
    "${branch_dir}/train_wrap_stats.json" "${branch_dir}/train_wrap_samples.json" \
    "${branch_dir}/dev_wrap_stats.json" "${branch_dir}/dev_wrap_samples.json" \
    "${branch_dir}/validation_summary.json"
  echo "[STAGE] ${lang} Branch A cap old LLM-generated term_map"
  python3 "${CAP_SCRIPT}" \
    --input-jsonl "${stage0_train}" \
    --output-jsonl "${capped_train}" \
    --stats-json "${branch_dir}/train_cap_stats.json" \
    --sample-json "${branch_dir}/train_cap_samples.json" \
    --max-terms "${MAX_TERMS}" \
    --sample-count 200
  python3 "${CAP_SCRIPT}" \
    --input-jsonl "${stage0_dev}" \
    --output-jsonl "${capped_dev}" \
    --stats-json "${branch_dir}/dev_cap_stats.json" \
    --sample-json "${branch_dir}/dev_cap_samples.json" \
    --max-terms "${MAX_TERMS}" \
    --sample-count 200
  echo "[STAGE] ${lang} Branch A exact/boundary-only assistant wrapping"
  wrap_jsonl "${lang}" "${capped_train}" "${final_train}" "${branch_dir}/train_wrap_stats.json" "${branch_dir}/train_wrap_samples.json"
  wrap_jsonl "${lang}" "${capped_dev}" "${final_dev}" "${branch_dir}/dev_wrap_stats.json" "${branch_dir}/dev_wrap_samples.json"
  validate_final "${lang}" "${final_train}" "${final_dev}" "${branch_dir}/validation_summary.json"
}

split_jsonl() {
  local input_jsonl="$1"
  local shard_dir="$2"
  mkdir -p "${shard_dir}"
  python3 - "${input_jsonl}" "${shard_dir}" "${NUM_SHARDS}" <<'PY'
import sys
from pathlib import Path

input_path = Path(sys.argv[1])
shard_dir = Path(sys.argv[2])
n = int(sys.argv[3])
for old in shard_dir.glob("input_shard_*.jsonl"):
    old.unlink()
handles = [(shard_dir / f"input_shard_{i}.jsonl").open("w", encoding="utf-8") for i in range(n)]
try:
    with input_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            handles[idx % n].write(line)
finally:
    for h in handles:
        h.close()
for i in range(n):
    p = shard_dir / f"input_shard_{i}.jsonl"
    count = sum(1 for _ in p.open("r", encoding="utf-8"))
    print(f"[SPLIT] shard={i} lines={count} path={p}", flush=True)
PY
}

generate_retriever_jsonl() {
  local lang="$1"
  local input_jsonl="$2"
  local glossary_json="$3"
  local shard_dir="$4"
  local merged_jsonl="$5"
  split_jsonl "${input_jsonl}" "${shard_dir}"
  IFS=',' read -r -a allocated_gpus <<< "${ALLOCATED_GPU_CSV}"
  if (( ${#allocated_gpus[@]} < NUM_SHARDS )); then
    echo "[ERROR] Need ${NUM_SHARDS} allocated GPUs, got ${ALLOCATED_GPU_CSV:-<unset>}" >&2
    exit 2
  fi
  echo "[STAGE] ${lang} HN1024 retriever term_map generation input=${input_jsonl}"
  pids=()
  for i in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu="${allocated_gpus[$i]}"
    in_shard="${shard_dir}/input_shard_${i}.jsonl"
    out_shard="${shard_dir}/retriever_results_shard_${i}.jsonl"
    log_shard="${shard_dir}/retriever_results_shard_${i}.log"
    (
      CUDA_VISIBLE_DEVICES="${gpu}" python3 "${GENERATE_SCRIPT}" \
        --cleaned_jsonl "${in_shard}" \
        --glossary_json "${glossary_json}" \
        --model_path "${HN1024_CKPT}" \
        --output_jsonl "${out_shard}" \
        --device cuda:0 \
        --retrieval_density "${RETRIEVAL_DENSITY}" \
        --top_k_mode duration_sec_cap \
        --max_top_k "${MAX_TOP_K}" \
        --score_threshold "${TAU}" \
        --rag_feature_extractor_model_id "${RAG_FEATURE_EXTRACTOR_MODEL_ID}" \
        --target_lang "${lang}" \
        --batch_across_conversations \
        ${MAX_CONVERSATIONS:+--max_conversations "${MAX_CONVERSATIONS}"}
    ) > "${log_shard}" 2>&1 &
    pids+=("$!")
    echo "[LAUNCH] lang=${lang} shard=${i} gpu=${gpu} pid=${pids[-1]}"
    sleep 2
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
  : > "${merged_jsonl}"
  for i in $(seq 0 $((NUM_SHARDS - 1))); do
    shard="${shard_dir}/retriever_results_shard_${i}.jsonl"
    [[ -s "${shard}" ]] || { echo "[ERROR] Missing/empty retriever shard: ${shard}" >&2; exit 3; }
    cat "${shard}" >> "${merged_jsonl}"
  done
}

build_retriever_branch() {
  local lang="$1"
  local out_dir="$2"
  local glossary_json="$3"
  local branch_dir="${out_dir}/retriever_hn1024_tau${TAU/./}_cap${MAX_TERMS}_exactboundary"
  mkdir -p "${branch_dir}"
  local stage0_train="${out_dir}/stage0_${lang}_exact_gt_from_llmgen_termmap.jsonl"
  local stage0_dev="${out_dir}/stage0_${lang}_exact_gt_from_llmgen_termmap_dev_first${DEV_ROWS}.jsonl"
  local train_retrieved="${branch_dir}/train_s_${lang}_retriever_results_hn1024_tau${TAU/./}.jsonl"
  local dev_retrieved="${branch_dir}/dev_s_${lang}_retriever_results_hn1024_tau${TAU/./}_first${DEV_ROWS}.jsonl"
  local rebuilt_train="${branch_dir}/train_s_${lang}_retriever_cap${MAX_TERMS}_stage1.jsonl"
  local rebuilt_dev="${branch_dir}/dev_s_${lang}_retriever_cap${MAX_TERMS}_stage1_first${DEV_ROWS}.jsonl"
  local final_train="${branch_dir}/train_s_${lang}_retriever_hn1024_tau${TAU/./}_cap${MAX_TERMS}_gttermwrap_exactboundary.jsonl"
  local final_dev="${branch_dir}/dev_s_${lang}_retriever_hn1024_tau${TAU/./}_cap${MAX_TERMS}_gttermwrap_exactboundary_first${DEV_ROWS}.jsonl"
  maybe_rm_outputs \
    "${train_retrieved}" "${dev_retrieved}" "${rebuilt_train}" "${rebuilt_dev}" "${final_train}" "${final_dev}" \
    "${branch_dir}/train_wrap_stats.json" "${branch_dir}/train_wrap_samples.json" \
    "${branch_dir}/dev_wrap_stats.json" "${branch_dir}/dev_wrap_samples.json" \
    "${branch_dir}/validation_summary.json"
  generate_retriever_jsonl "${lang}" "${stage0_train}" "${glossary_json}" "${branch_dir}/train_shards" "${train_retrieved}"
  generate_retriever_jsonl "${lang}" "${stage0_dev}" "${glossary_json}" "${branch_dir}/dev_shards" "${dev_retrieved}"
  echo "[STAGE] ${lang} Branch B rebuild term_map with GT backfill and cap=${MAX_TERMS}"
  python3 "${REBUILD_SCRIPT}" \
    --input_jsonl "${train_retrieved}" \
    --output_jsonl "${rebuilt_train}" \
    --termmap_mode tcm_filtered_with_gt_backfill \
    --max_terms "${MAX_TERMS}" \
    --target_lang "${lang}" \
    --seed 42 \
    ${MAX_CONVERSATIONS:+--max_conversations "${MAX_CONVERSATIONS}"}
  python3 "${REBUILD_SCRIPT}" \
    --input_jsonl "${dev_retrieved}" \
    --output_jsonl "${rebuilt_dev}" \
    --termmap_mode tcm_filtered_with_gt_backfill \
    --max_terms "${MAX_TERMS}" \
    --target_lang "${lang}" \
    --seed 42
  echo "[STAGE] ${lang} Branch B exact/boundary-only assistant wrapping"
  wrap_jsonl "${lang}" "${rebuilt_train}" "${final_train}" "${branch_dir}/train_wrap_stats.json" "${branch_dir}/train_wrap_samples.json"
  wrap_jsonl "${lang}" "${rebuilt_dev}" "${final_dev}" "${branch_dir}/dev_wrap_stats.json" "${branch_dir}/dev_wrap_samples.json"
  validate_final "${lang}" "${final_train}" "${final_dev}" "${branch_dir}/validation_summary.json"
}

validate_final() {
  local lang="$1"
  local train_jsonl="$2"
  local dev_jsonl="$3"
  local summary_json="$4"
  python3 - "${lang}" "${train_jsonl}" "${dev_jsonl}" "${summary_json}" "${MAX_TERMS}" <<'PY'
import json
import re
import sys
from pathlib import Path

lang, train_s, dev_s, summary_s, max_terms_s = sys.argv[1:]
paths = {"train": Path(train_s), "dev": Path(dev_s)}
max_terms = int(max_terms_s)
latin_re = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]")

def latin(ch):
    return bool(ch) and bool(latin_re.fullmatch(ch))

def count_terms(content):
    content = str(content or "")
    idx = content.find("term_map:")
    if idx < 0 or "term_map:NONE" in content:
        return 0
    body = content[idx + len("term_map:"):].strip()
    return sum(1 for line in body.splitlines() if "=" in line)

def bad_tags(text):
    malformed = text.count("<term>") != text.count("</term>")
    latin_cut = False
    pos = 0
    while True:
        start = text.find("<term>", pos)
        if start < 0:
            break
        end = text.find("</term>", start + len("<term>"))
        if end < 0:
            malformed = True
            break
        inner_start = start + len("<term>")
        inner_end = end
        if inner_start >= inner_end:
            malformed = True
            break
        before = text[start - 1] if start > 0 else ""
        first = text[inner_start]
        last = text[inner_end - 1]
        after_idx = end + len("</term>")
        after = text[after_idx] if after_idx < len(text) else ""
        latin_cut |= latin(before) and latin(first)
        latin_cut |= latin(after) and latin(last)
        pos = after_idx
    return malformed, latin_cut

out = {"lang": lang, "max_terms": max_terms}
for split, path in paths.items():
    rows = chunks = term_chunks = max_seen = tagged_rows = malformed = latin_cut = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            rows += 1
            obj = json.loads(line)
            messages = obj.get("messages")
            audios = obj.get("audios")
            gt = obj.get("gt_terms_by_chunk")
            if not isinstance(messages, list) or not isinstance(audios, list) or not isinstance(gt, list):
                raise SystemExit(f"[ERROR] malformed row {line_no} in {path}")
            user_indices = [
                i for i, m in enumerate(messages)
                if m.get("role") == "user" and str(m.get("content") or "").startswith("<audio>")
            ]
            if len(user_indices) != len(audios) or len(gt) != len(audios):
                raise SystemExit(f"[ERROR] row {line_no} user/audio/gt mismatch")
            chunks += len(audios)
            row_tagged = False
            for idx in user_indices:
                n = count_terms(messages[idx].get("content"))
                max_seen = max(max_seen, n)
                if n:
                    term_chunks += 1
                if n > max_terms:
                    raise SystemExit(f"[ERROR] row {line_no} chunk term_map exceeds cap: {n} > {max_terms}")
            for msg in messages:
                if msg.get("role") != "assistant":
                    continue
                text = str(msg.get("content") or "")
                if "<term>" in text:
                    row_tagged = True
                bad, cut = bad_tags(text)
                malformed += int(bad)
                latin_cut += int(cut)
            tagged_rows += int(row_tagged)
    if rows == 0:
        raise SystemExit(f"[ERROR] empty {split}: {path}")
    out[split] = {
        "path": str(path),
        "rows": rows,
        "chunks": chunks,
        "termmap_chunks": term_chunks,
        "termmap_chunk_rate": term_chunks / max(1, chunks),
        "max_termmap_entries": max_seen,
        "tagged_rows": tagged_rows,
        "malformed_tag_messages": malformed,
        "latin_boundary_cut_messages": latin_cut,
    }
    if malformed or latin_cut:
        raise SystemExit(f"[ERROR] bad tag validation in {path}: malformed={malformed} latin_cut={latin_cut}")
Path(summary_s).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
PY
}

echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo "[INFO] LOG_ROOT=${LOG_ROOT}"
echo "[INFO] LANGS=${LANGS}"
echo "[INFO] BRANCHES=${BRANCHES}"
echo "[INFO] MAX_TERMS=${MAX_TERMS} DEV_ROWS=${DEV_ROWS} TAU=${TAU}"
echo "[INFO] ALLOCATED_GPU_CSV=${ALLOCATED_GPU_CSV}"
df -h /mnt/gemini/data1 || true

for lang in ${LANGS}; do
  source_jsonl="$(source_jsonl_for_lang "${lang}")"
  glossary_json="$(glossary_for_lang "${lang}")"
  [[ -s "${source_jsonl}" ]] || { echo "[ERROR] Missing source JSONL: ${source_jsonl}" >&2; exit 3; }
  [[ -s "${glossary_json}" ]] || { echo "[ERROR] Missing glossary JSON: ${glossary_json}" >&2; exit 3; }
  out_dir="${OUT_ROOT}/${lang}"
  mkdir -p "${out_dir}"
  echo "[LANG] ${lang} source=${source_jsonl} glossary=${glossary_json} out=${out_dir}"
  derive_gt_once "${lang}" "${source_jsonl}" "${out_dir}"
  if contains_word llmgen "${BRANCHES}"; then
    build_llmgen_branch "${lang}" "${out_dir}"
  fi
  if contains_word retriever "${BRANCHES}"; then
    build_retriever_branch "${lang}" "${out_dir}" "${glossary_json}"
  fi
done

echo "[DONE] De/Ja term-map ablation data construction complete: ${OUT_ROOT}"
