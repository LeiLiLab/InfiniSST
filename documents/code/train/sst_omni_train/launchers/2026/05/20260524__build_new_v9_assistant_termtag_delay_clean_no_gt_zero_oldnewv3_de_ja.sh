#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
if [[ -d /mnt/taurus/home/jiaxuanluo/InfiniSST ]]; then
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
fi

CONDA_PREFIX_OVERRIDE="${CONDA_PREFIX_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv}"
if [[ -d "${CONDA_PREFIX_OVERRIDE}" ]]; then
  export PATH="${CONDA_PREFIX_OVERRIDE}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX_OVERRIDE}/lib:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/mnt/gemini/data1/jiaxuanluo/huggingface_cache}"
export TORCH_HOME="${TORCH_HOME:-/mnt/gemini/data1/jiaxuanluo/torch_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/gemini/data1/jiaxuanluo/xdg_cache}"

DATA_ROOT="${DATA_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo}"
OUT_ROOT="${OUT_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo}"
LANGS="${LANGS_OVERRIDE:-de ja}"
DEV_ROWS="${DEV_ROWS_OVERRIDE:-355}"
NUM_SHARDS="${NUM_SHARDS_OVERRIDE:-2}"
GPU_CSV="${TCM_BUILD_GPU_DEVICES_OVERRIDE:-${CUDA_VISIBLE_DEVICES:-0,1}}"
FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"

# RUN_VARIANT_STAGE=0 stops after the old-new_v3-equivalent retriever rebuild.
# RUN_VARIANT_STAGE=1 continues through New V4 -> New V5 -> New V9 and requires
# OPENAI_API_KEY unless CACHE_ONLY=1 or DRY_RUN=1.
RUN_VARIANT_STAGE="${RUN_VARIANT_STAGE_OVERRIDE:-1}"
CACHE_ONLY="${CACHE_ONLY_OVERRIDE:-0}"
DRY_RUN="${DRY_RUN_OVERRIDE:-0}"

DERIVE_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/derive_gt_terms_from_termmap_matches.py"
EXTRACT_GLOSSARY_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/extract_termmap_glossary.py"
GENERATE_SCRIPT="${ROOT_DIR}/documents/code/data_pre/hard_negative_jsonl_for_speech_llm/generate_termmap_maxsim.py"
REBUILD_SCRIPT="${ROOT_DIR}/documents/code/data_pre/hard_negative_jsonl_for_speech_llm/rebuild_termmap.py"
AUGMENT_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/augment_term_translation_llm_variants.py"
ZERO_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/zero_no_gt_termmap_chunks.py"
WRAP_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/wrap_assistant_term_targets.py"

TCM_RAG_CKPT="${TCM_RAG_CKPT_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_ntcm_final_v3_n64_p1n4_s2000_aries-8gpu_smallest_dense_smoke4650_best.pt}"
RAG_FEATURE_EXTRACTOR_MODEL_ID="${RAG_FEATURE_EXTRACTOR_MODEL_ID_OVERRIDE:-openai/whisper-large-v3}"

RETRIEVAL_DENSITY="${RETRIEVAL_DENSITY_OVERRIDE:-9}"
MAX_TOP_K="${MAX_TOP_K_OVERRIDE:-20}"
TERM_MAP_MAX_TERMS="${TERM_MAP_MAX_TERMS_OVERRIDE:-20}"
TAU="${TAU_OVERRIDE:-0.75}"
GLOSSARY_MAX_TERMS="${GLOSSARY_MAX_TERMS_OVERRIDE:-100000}"

EXCLUDE_SOURCE_TOKENS="${EXCLUDE_SOURCE_TOKENS_OVERRIDE:-this,that,these,those,his,her,hers,him,he,she,it,its,they,them,their,theirs,you,your,yours,we,our,ours,i,me,my,mine,myself,yourself,himself,herself,itself,ourselves,yourselves,themselves,what,which,who,whom,whose,someone,somebody,something,anyone,anybody,anything,everyone,everybody,everything}"

for p in \
  "${DERIVE_SCRIPT}" \
  "${EXTRACT_GLOSSARY_SCRIPT}" \
  "${GENERATE_SCRIPT}" \
  "${REBUILD_SCRIPT}" \
  "${AUGMENT_SCRIPT}" \
  "${ZERO_SCRIPT}" \
  "${WRAP_SCRIPT}" \
  "${TCM_RAG_CKPT}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

IFS=',' read -r -a ALLOCATED_GPUS <<< "${GPU_CSV}"
if (( ${#ALLOCATED_GPUS[@]} < NUM_SHARDS )); then
  echo "[ERROR] Need ${NUM_SHARDS} GPUs, got GPU_CSV=${GPU_CSV}" >&2
  exit 2
fi

if [[ "${RUN_VARIANT_STAGE}" == "1" && "${CACHE_ONLY}" != "1" && "${DRY_RUN}" != "1" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ERROR] RUN_VARIANT_STAGE=1 requires OPENAI_API_KEY unless CACHE_ONLY=1 or DRY_RUN=1." >&2
  exit 5
fi

echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] LANGS=${LANGS}"
echo "[INFO] NUM_SHARDS=${NUM_SHARDS} GPU_CSV=${GPU_CSV}"
echo "[INFO] RUN_VARIANT_STAGE=${RUN_VARIANT_STAGE} CACHE_ONLY=${CACHE_ONLY} DRY_RUN=${DRY_RUN}"

for lang in ${LANGS}; do
  source_jsonl="${DATA_ROOT}/train_s_${lang}_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl"
  out_dir="${OUT_ROOT}/speech_llm_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_${lang}_20260524"
  shard_dir="${out_dir}/shards"

  source_gt="${out_dir}/stage0_train_s_${lang}_derived_gt_from_llmgen_termmap.jsonl"
  glossary_json="${out_dir}/stage0_termmap_glossary_${lang}_top${GLOSSARY_MAX_TERMS}.json"
  retriever_merged="${out_dir}/stage1_train_s_${lang}_oldnewv3_equiv_retriever_results.jsonl"
  oldnewv3_jsonl="${out_dir}/stage2_train_s_${lang}_oldnewv3_equiv_tcmwiki_termmap_gtbackfill.jsonl"
  new_v4_train="${out_dir}/train_s_${lang}_new_v4_llm_variant_aug_oldnewv3_cache.jsonl"
  new_v4_dev="${out_dir}/dev_s_${lang}_new_v4_llm_variant_aug_oldnewv3_cache_first${DEV_ROWS}.jsonl"
  new_v5_train="${out_dir}/train_s_${lang}_new_v5_no_gt_zero_llm_variant_aug_oldnewv3.jsonl"
  new_v5_dev="${out_dir}/dev_s_${lang}_new_v5_no_gt_zero_llm_variant_aug_oldnewv3_first${DEV_ROWS}.jsonl"
  final_train="${out_dir}/train_s_${lang}_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3.jsonl"
  final_dev="${out_dir}/dev_s_${lang}_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_first${DEV_ROWS}.jsonl"
  variant_cache="${out_dir}/openai_term_variant_cache_${lang}.json"
  summary="${out_dir}/new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_${lang}_summary.json"

  if [[ ! -f "${source_jsonl}" ]]; then
    echo "[ERROR] Missing source JSONL for ${lang}: ${source_jsonl}" >&2
    exit 3
  fi
  mkdir -p "${out_dir}" "${shard_dir}"

  outputs=(
    "${source_gt}"
    "${glossary_json}"
    "${retriever_merged}"
    "${oldnewv3_jsonl}"
    "${new_v4_train}"
    "${new_v4_dev}"
    "${new_v5_train}"
    "${new_v5_dev}"
    "${final_train}"
    "${final_dev}"
    "${summary}"
    "${out_dir}/stage0_derive_gt_stats.json"
    "${out_dir}/stage0_derive_gt_samples.json"
    "${out_dir}/stage0_glossary_stats.json"
    "${out_dir}/stage2_rebuild_oldnewv3_equiv.log"
    "${out_dir}/train_new_v4_llm_variant_aug_stats.json"
    "${out_dir}/dev_new_v4_llm_variant_aug_first${DEV_ROWS}_stats.json"
    "${out_dir}/train_new_v5_no_gt_zero_stats.json"
    "${out_dir}/dev_new_v5_no_gt_zero_first${DEV_ROWS}_stats.json"
    "${out_dir}/train_new_v9_assistant_termtag_delay_clean_stats.json"
    "${out_dir}/dev_new_v9_assistant_termtag_delay_clean_first${DEV_ROWS}_stats.json"
  )
  if [[ "${FORCE_OVERWRITE}" == "1" ]]; then
    rm -f "${outputs[@]}"
    rm -f "${shard_dir}"/input_shard_*.jsonl "${shard_dir}"/retriever_results_shard_*.jsonl "${shard_dir}"/retriever_results_shard_*.log
  else
    for p in "${outputs[@]}"; do
      if [[ -e "${p}" ]]; then
        echo "[ERROR] Output exists for ${lang}: ${p}" >&2
        echo "[ERROR] Set FORCE_OVERWRITE=1 only if this rerun is intentional." >&2
        exit 4
      fi
    done
  fi

  echo "[STAGE 0A] ${lang}: derive gt_terms_by_chunk from legacy LLM-generated term_map"
  python3 "${DERIVE_SCRIPT}" \
    --input-jsonl "${source_jsonl}" \
    --output-jsonl "${source_gt}" \
    --stats-json "${out_dir}/stage0_derive_gt_stats.json" \
    --sample-json "${out_dir}/stage0_derive_gt_samples.json" \
    --lang-code "${lang}" \
    --min-target-chars 2 \
    --exclude-source-tokens "${EXCLUDE_SOURCE_TOKENS}" \
    --fuzzy-match \
    --fuzzy-min-target-chars 4 \
    --fuzzy-min-score 0.58 \
    --fuzzy-max-span-ratio 1.60 \
    --fuzzy-max-span-extra-chars 4 \
    --max-terms-per-chunk 16 \
    --sample-count 200

  echo "[STAGE 0B] ${lang}: extract top-${GLOSSARY_MAX_TERMS} train term_map glossary"
  python3 "${EXTRACT_GLOSSARY_SCRIPT}" \
    --input-jsonl "${source_jsonl}" \
    --output-json "${glossary_json}" \
    --stats-json "${out_dir}/stage0_glossary_stats.json" \
    --lang-code "${lang}" \
    --max-terms "${GLOSSARY_MAX_TERMS}"

  echo "[STAGE 1A] ${lang}: split into ${NUM_SHARDS} shards"
  rm -f "${shard_dir}"/input_shard_*.jsonl
  python3 - "${source_gt}" "${shard_dir}" "${NUM_SHARDS}" <<'PY'
import sys
from pathlib import Path

input_path = Path(sys.argv[1])
shard_dir = Path(sys.argv[2])
n = int(sys.argv[3])
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

  echo "[STAGE 1B] ${lang}: run TCM retriever term_map generation"
  pids=()
  for i in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu="${ALLOCATED_GPUS[$i]}"
    in_shard="${shard_dir}/input_shard_${i}.jsonl"
    out_shard="${shard_dir}/retriever_results_shard_${i}.jsonl"
    log_shard="${shard_dir}/retriever_results_shard_${i}.log"
    (
      env CUDA_VISIBLE_DEVICES="${gpu}" python3 "${GENERATE_SCRIPT}" \
        --cleaned_jsonl "${in_shard}" \
        --glossary_json "${glossary_json}" \
        --model_path "${TCM_RAG_CKPT}" \
        --output_jsonl "${out_shard}" \
        --device "cuda:0" \
        --retrieval_density "${RETRIEVAL_DENSITY}" \
        --top_k_mode duration_sec_cap \
        --max_top_k "${MAX_TOP_K}" \
        --score_threshold "${TAU}" \
        --rag_feature_extractor_model_id "${RAG_FEATURE_EXTRACTOR_MODEL_ID}" \
        --target_lang "${lang}"
    ) > "${log_shard}" 2>&1 &
    pids+=("$!")
    echo "[LAUNCH] ${lang} shard=${i} gpu=${gpu} pid=${pids[-1]}"
    sleep 2
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  echo "[STAGE 1C] ${lang}: merge retriever shards"
  : > "${retriever_merged}"
  for i in $(seq 0 $((NUM_SHARDS - 1))); do
    shard="${shard_dir}/retriever_results_shard_${i}.jsonl"
    if [[ ! -s "${shard}" ]]; then
      echo "[ERROR] Missing/empty retriever shard: ${shard}" >&2
      exit 3
    fi
    cat "${shard}" >> "${retriever_merged}"
  done

  echo "[STAGE 2] ${lang}: rebuild old-new_v3-equivalent train JSONL"
  python3 "${REBUILD_SCRIPT}" \
    --input_jsonl "${retriever_merged}" \
    --output_jsonl "${oldnewv3_jsonl}" \
    --termmap_mode tcm_filtered_with_gt_backfill \
    --max_terms "${TERM_MAP_MAX_TERMS}" \
    --target_lang "${lang}" \
    --seed 42 \
    > "${out_dir}/stage2_rebuild_oldnewv3_equiv.log" 2>&1

  if [[ "${RUN_VARIANT_STAGE}" != "1" ]]; then
    echo "[INFO] ${lang}: RUN_VARIANT_STAGE=${RUN_VARIANT_STAGE}; stopping after old-new_v3-equivalent data."
    continue
  fi

  augment_common=(
    --lang-code "${lang}"
    --augment-prob "${AUGMENT_PROB_OVERRIDE:-0.5}"
    --max-augmented-terms-per-row "${MAX_AUG_TERMS_PER_ROW_OVERRIDE:-8}"
    --sample-count "${SAMPLE_COUNT_OVERRIDE:-80}"
    --missing-gt-policy error
  )
  if [[ "${CACHE_ONLY}" == "1" ]]; then
    augment_common+=(--cache-only)
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    augment_common+=(--dry-run)
  fi

  echo "[STAGE 3] ${lang}: New V4 LLM-variant augmentation"
  python3 "${AUGMENT_SCRIPT}" \
    --input-jsonl "${oldnewv3_jsonl}" \
    --output-jsonl "${new_v4_train}" \
    --stats-json "${out_dir}/train_new_v4_llm_variant_aug_stats.json" \
    --variant-cache-json "${variant_cache}" \
    --sample-json "${out_dir}/train_new_v4_llm_variant_aug_samples.json" \
    --seed "${TRAIN_SEED_OVERRIDE:-1606}" \
    "${augment_common[@]}"

  python3 "${AUGMENT_SCRIPT}" \
    --input-jsonl "${oldnewv3_jsonl}" \
    --output-jsonl "${new_v4_dev}" \
    --stats-json "${out_dir}/dev_new_v4_llm_variant_aug_first${DEV_ROWS}_stats.json" \
    --variant-cache-json "${variant_cache}" \
    --sample-json "${out_dir}/dev_new_v4_llm_variant_aug_first${DEV_ROWS}_samples.json" \
    --seed "${DEV_SEED_OVERRIDE:-1607}" \
    --max-rows "${DEV_ROWS}" \
    "${augment_common[@]}"

  echo "[STAGE 4] ${lang}: New V5 no-GT-zero"
  python3 "${ZERO_SCRIPT}" \
    --input-jsonl "${new_v4_train}" \
    --output-jsonl "${new_v5_train}" \
    --stats-json "${out_dir}/train_new_v5_no_gt_zero_stats.json" \
    --sample-json "${out_dir}/train_new_v5_no_gt_zero_samples.json" \
    --missing-gt-policy error

  python3 "${ZERO_SCRIPT}" \
    --input-jsonl "${new_v4_dev}" \
    --output-jsonl "${new_v5_dev}" \
    --stats-json "${out_dir}/dev_new_v5_no_gt_zero_first${DEV_ROWS}_stats.json" \
    --sample-json "${out_dir}/dev_new_v5_no_gt_zero_first${DEV_ROWS}_samples.json" \
    --missing-gt-policy error

  echo "[STAGE 5] ${lang}: New V9 assistant <term> tagging"
  wrap_common=(
    --lang-code "${lang}"
    --tag-template '<term>{translation}</term>'
    --min-target-chars 2
    --max-tags-per-row 16
    --missing-gt-policy error
    --exclude-source-tokens "${EXCLUDE_SOURCE_TOKENS}"
    --enable-local-rewrite
    --rewrite-avoid-boundary-overlap
    --rewrite-delay-boundary-prefix
    --rewrite-delay-boundary-min-prefix-chars 2
    --rewrite-min-target-chars 4
    --rewrite-min-score 0.58
    --rewrite-min-coverage 0.40
    --rewrite-max-span-ratio 1.60
    --rewrite-max-span-extra-chars 4
    --sample-count 200
  )
  python3 "${WRAP_SCRIPT}" \
    --input-jsonl "${new_v5_train}" \
    --output-jsonl "${final_train}" \
    --stats-json "${out_dir}/train_new_v9_assistant_termtag_delay_clean_stats.json" \
    --sample-json "${out_dir}/train_new_v9_assistant_termtag_delay_clean_samples.json" \
    "${wrap_common[@]}"

  python3 "${WRAP_SCRIPT}" \
    --input-jsonl "${new_v5_dev}" \
    --output-jsonl "${final_dev}" \
    --stats-json "${out_dir}/dev_new_v9_assistant_termtag_delay_clean_first${DEV_ROWS}_stats.json" \
    --sample-json "${out_dir}/dev_new_v9_assistant_termtag_delay_clean_first${DEV_ROWS}_samples.json" \
    "${wrap_common[@]}"

  python3 - "${lang}" "${source_jsonl}" "${out_dir}" "${summary}" <<'PY'
import json
import sys
from pathlib import Path

lang = sys.argv[1]
source = sys.argv[2]
out = Path(sys.argv[3])
summary_path = Path(sys.argv[4])

def load(name):
    p = out / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

summary = {
    "event": f"new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_{lang}",
    "source_jsonl": source,
    "policy": {
        "zh_lineage_reference": "sourcefinal_tcmwiki100kgt -> New V4 LLM-variant -> New V5 no-GT-zero -> New V9 assistant term tags",
        "language_specific_delta": "de/ja source JSONL lacks gt_terms_by_chunk, so GT is derived from existing LLM-generated term_map entries supported by future assistant text",
        "glossary_source": "top source terms from legacy language-specific training term_map",
        "retriever_rebuild": "old-new_v3-equivalent d9 tau0.75 cap20 with GT backfill",
        "variant_stage": "OpenAI natural target variants when enabled",
        "no_gt_zero": True,
        "assistant_tag_template": "<term>{translation}</term>",
    },
    "stats": {
        "derive_gt": load("stage0_derive_gt_stats.json"),
        "glossary": load("stage0_glossary_stats.json"),
        "new_v4": load("train_new_v4_llm_variant_aug_stats.json"),
        "new_v5": load("train_new_v5_no_gt_zero_stats.json"),
        "new_v9": load("train_new_v9_assistant_termtag_delay_clean_stats.json"),
    },
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
PY

  echo "[OK] ${lang}: ${final_train}"
  echo "[OK] ${lang}: ${final_dev}"
  echo "[OK] ${lang}: ${summary}"
done
