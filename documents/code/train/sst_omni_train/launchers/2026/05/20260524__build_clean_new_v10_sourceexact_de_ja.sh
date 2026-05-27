#!/usr/bin/env bash
# CANCELED / DO NOT USE:
# This v13/sourceexact-derived de/ja branch is the wrong lineage for the final
# Speech LLM line.  The mainline zh model comes from old new_v3 r32 -> new_v4 ->
# new_v5 -> new_v9, not v13/sourceexact.  Kept only as a canceled provenance
# record; do not launch.
echo "[CANCELED] Wrong lineage: use old new_v3 r32 -> new_v4 -> new_v5 -> new_v9 instead." >&2
exit 2
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
LANGS="${LANGS_OVERRIDE:-de ja}"
DEV_ROWS="${DEV_ROWS_OVERRIDE:-355}"
NUM_SHARDS="${NUM_SHARDS_OVERRIDE:-2}"
GPU_DEVICES_CSV="${GPU_DEVICES_CSV_OVERRIDE:-${CUDA_VISIBLE_DEVICES:-0,1}}"
FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"
CACHE_ONLY="${CACHE_ONLY_OVERRIDE:-0}"
DRY_RUN="${DRY_RUN_OVERRIDE:-0}"
RUN_VARIANT_STAGE="${RUN_VARIANT_STAGE_OVERRIDE:-1}"

SOURCE_GT_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/build_source_glossary_exact_gt_terms.py"
RETRIEVER_TIMELINE_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/build_retriever_timeline_termmap_sft.py"
AUGMENT_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/augment_term_translation_llm_variants.py"
ZERO_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/zero_no_gt_termmap_chunks.py"
WRAP_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/wrap_assistant_term_targets.py"

HN1024_MODEL_PATH="${HN1024_MODEL_PATH_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"

TOP_K="${TOP_K_OVERRIDE:-10}"
SCORE_THRESHOLD="${SCORE_THRESHOLD_OVERRIDE:-0.78}"
LOOKBACK_SEC="${LOOKBACK_SEC_OVERRIDE:-1.92}"
MIN_CONTEXT_SEC="${MIN_CONTEXT_SEC_OVERRIDE:-2.88}"
MAX_CONTEXT_SEC="${MAX_CONTEXT_SEC_OVERRIDE:-5.76}"
# de/ja legacy JSONL does not carry row-level merge_multiplier.  Keep all rows
# by default; retriever readiness is controlled by measured audio duration plus
# min_context/lookback below.
MERGE_MULTIPLIER_MIN="${MERGE_MULTIPLIER_MIN_OVERRIDE:-0}"
MERGE_MULTIPLIER_MAX="${MERGE_MULTIPLIER_MAX_OVERRIDE:-0}"
AUDIO_BATCH_SIZE="${AUDIO_BATCH_SIZE_OVERRIDE:-4}"
TEXT_ENCODE_BATCH="${TEXT_ENCODE_BATCH_OVERRIDE:-256}"
MAX_CONVERSATIONS="${MAX_CONVERSATIONS_OVERRIDE:-0}"

EXCLUDE_SOURCE_TOKENS="${EXCLUDE_SOURCE_TOKENS_OVERRIDE:-this,that,these,those,his,her,hers,him,he,she,it,its,they,them,their,theirs,you,your,yours,we,our,ours,i,me,my,mine,myself,yourself,himself,herself,itself,ourselves,yourselves,themselves,what,which,who,whom,whose,someone,somebody,something,anyone,anybody,anything,everyone,everybody,everything}"

for p in "${SOURCE_GT_SCRIPT}" "${RETRIEVER_TIMELINE_SCRIPT}" "${AUGMENT_SCRIPT}" "${ZERO_SCRIPT}" "${WRAP_SCRIPT}" "${HN1024_MODEL_PATH}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

IFS=',' read -r -a GPU_DEVICES <<< "${GPU_DEVICES_CSV}"
if (( ${#GPU_DEVICES[@]} < NUM_SHARDS )); then
  echo "[ERROR] Need at least NUM_SHARDS GPUs; NUM_SHARDS=${NUM_SHARDS} GPU_DEVICES_CSV=${GPU_DEVICES_CSV}" >&2
  exit 2
fi
if [[ "${RUN_VARIANT_STAGE}" == "1" && "${CACHE_ONLY}" != "1" && "${DRY_RUN}" != "1" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ERROR] RUN_VARIANT_STAGE=1 requires OPENAI_API_KEY unless CACHE_ONLY=1 or DRY_RUN=1." >&2
  exit 5
fi

cd "${ROOT_DIR}"
echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] LANGS=${LANGS}"
echo "[INFO] GPU_DEVICES_CSV=${GPU_DEVICES_CSV} NUM_SHARDS=${NUM_SHARDS}"
echo "[INFO] SCORE_THRESHOLD=${SCORE_THRESHOLD} TOP_K=${TOP_K}"

for lang in ${LANGS}; do
  source_jsonl="${DATA_ROOT}/train_s_${lang}_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl"
  source_tsv="${DATA_ROOT}/train_xl_case_robust_asr-filtered_${lang}_metricx-qe3.0_align.tsv"
  glossary_var="GLOSSARY_JSON_${lang^^}_OVERRIDE"
  glossary_json="${!glossary_var:-${DATA_ROOT}/glossary_for_${lang}_rate1.0_k20.json}"
  tau_tag="$(printf '%s' "${SCORE_THRESHOLD}" | tr '.' 'p')"
  out_dir="${OUT_ROOT}/speech_llm_new_v10_sourceexact_${lang}_20260524"
  shard_dir="${out_dir}/shards_tau${tau_tag}"

  stage0_jsonl="${out_dir}/stage0_train_s_${lang}_srcchunk_sourceexact_future_ref_gt_termmap_none.jsonl"
  stage0_stats="${out_dir}/stage0_sourceexact_future_ref_gt_stats.json"
  stage0_samples="${out_dir}/stage0_sourceexact_future_ref_gt_samples.json"
  text_index="${out_dir}/hn1024_tau${tau_tag}_${lang}_text_index.pt"
  stage1_jsonl="${out_dir}/stage1_train_s_${lang}_lm1to6_retriever_timeline_tau${tau_tag}_k${TOP_K}_minctx2p88.jsonl"
  stage1_stats="${out_dir}/stage1_train_s_${lang}_lm1to6_retriever_timeline_tau${tau_tag}_k${TOP_K}_minctx2p88_stats.json"
  new_v4_train="${out_dir}/stage2_train_s_${lang}_llmvariant_sourceexact_retriever_timeline.jsonl"
  new_v4_dev="${out_dir}/stage2_dev_s_${lang}_llmvariant_sourceexact_retriever_timeline_first${DEV_ROWS}.jsonl"
  new_v5_train="${out_dir}/stage3_train_s_${lang}_no_gt_zero_sourceexact_llmvariant.jsonl"
  new_v5_dev="${out_dir}/stage3_dev_s_${lang}_no_gt_zero_sourceexact_llmvariant_first${DEV_ROWS}.jsonl"
  final_train="${out_dir}/train_s_${lang}_new_v10_sourceexact_llmvariant_no_gt_zero_termtag_boundary.jsonl"
  final_dev="${out_dir}/dev_s_${lang}_new_v10_sourceexact_llmvariant_no_gt_zero_termtag_boundary_first${DEV_ROWS}.jsonl"
  variant_cache="${out_dir}/openai_term_variant_cache_${lang}.json"
  summary_json="${out_dir}/new_v10_sourceexact_${lang}_summary.json"

  for p in "${source_jsonl}" "${source_tsv}" "${glossary_json}"; do
    if [[ ! -e "${p}" ]]; then
      echo "[ERROR] ${lang}: missing required input: ${p}" >&2
      exit 3
    fi
  done
  mkdir -p "${out_dir}" "${shard_dir}"

  outputs=(
    "${stage0_jsonl}" "${stage0_stats}" "${stage0_samples}" "${text_index}"
    "${stage1_jsonl}" "${stage1_stats}"
    "${new_v4_train}" "${new_v4_dev}"
    "${new_v5_train}" "${new_v5_dev}"
    "${final_train}" "${final_dev}" "${summary_json}"
  )
  if [[ "${FORCE_OVERWRITE}" == "1" ]]; then
    rm -f "${outputs[@]}"
    rm -f "${shard_dir}"/train_timeline_shard*_of*.jsonl "${shard_dir}"/train_timeline_shard*_of*.stats.json "${shard_dir}"/train_timeline_shard*_of*.samples.json "${shard_dir}"/train_timeline_shard*_of*.sample_chunks.json
  else
    for p in "${outputs[@]}"; do
      if [[ -e "${p}" ]]; then
        echo "[ERROR] ${lang}: output exists: ${p}" >&2
        echo "[ERROR] Set FORCE_OVERWRITE=1 only for intentional reruns." >&2
        exit 4
      fi
    done
  fi

  echo "[STAGE 0] ${lang}: source_chunk_asr + source-exact/future-ref GT"
  "${PYTHON_BIN}" "${SOURCE_GT_SCRIPT}" \
    --input-jsonl "${source_jsonl}" \
    --input-tsv "${source_tsv}" \
    --glossary-json "${glossary_json}" \
    --output-jsonl "${stage0_jsonl}" \
    --stats-json "${stage0_stats}" \
    --sample-json "${stage0_samples}" \
    --lang-code "${lang}" \
    --max-words 6 \
    --min-norm-chars 2 \
    --target-match-policy future_ref \
    --term-map-output-policy none \
    --exclude-source-tokens "${EXCLUDE_SOURCE_TOKENS}" \
    --sample-count 200

  if [[ ! -f "${text_index}" ]]; then
    echo "[STAGE 1A] ${lang}: build retriever text index"
    CUDA_VISIBLE_DEVICES="${GPU_DEVICES[0]}" "${PYTHON_BIN}" "${RETRIEVER_TIMELINE_SCRIPT}" \
      --build-index-only \
      --glossary-json "${glossary_json}" \
      --model-path "${HN1024_MODEL_PATH}" \
      --text-index-path "${text_index}" \
      --lang-code "${lang}" \
      --device cuda:0 \
      --text-encode-batch "${TEXT_ENCODE_BATCH}"
  fi

  echo "[STAGE 1B] ${lang}: retriever timeline term_map shards"
  pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu="${GPU_DEVICES[$shard]}"
    shard_out="${shard_dir}/train_timeline_shard${shard}_of${NUM_SHARDS}.jsonl"
    shard_stats="${shard_dir}/train_timeline_shard${shard}_of${NUM_SHARDS}.stats.json"
    shard_samples="${shard_dir}/train_timeline_shard${shard}_of${NUM_SHARDS}.samples.json"
    shard_sample_chunks="${shard_dir}/train_timeline_shard${shard}_of${NUM_SHARDS}.sample_chunks.json"
    (
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${RETRIEVER_TIMELINE_SCRIPT}" \
        --input-jsonl "${stage0_jsonl}" \
        --output-jsonl "${shard_out}" \
        --stats-json "${shard_stats}" \
        --sample-json "${shard_samples}" \
        --sample-chunks-json "${shard_sample_chunks}" \
        --glossary-json "${glossary_json}" \
        --model-path "${HN1024_MODEL_PATH}" \
        --text-index-path "${text_index}" \
        --lang-code "${lang}" \
        --top-k "${TOP_K}" \
        --score-threshold "${SCORE_THRESHOLD}" \
        --lookback-sec "${LOOKBACK_SEC}" \
        --min-context-sec "${MIN_CONTEXT_SEC}" \
        --max-context-sec "${MAX_CONTEXT_SEC}" \
        --merge-multiplier-min "${MERGE_MULTIPLIER_MIN}" \
        --merge-multiplier-max "${MERGE_MULTIPLIER_MAX}" \
        --audio-batch-size "${AUDIO_BATCH_SIZE}" \
        --num-shards "${NUM_SHARDS}" \
        --shard-index "${shard}" \
        --max-conversations "${MAX_CONVERSATIONS}" \
        --log-every 250
    ) &
    pids+=("$!")
    echo "[LAUNCH] ${lang} timeline shard=${shard} gpu=${gpu} pid=${pids[-1]}"
    sleep 2
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  echo "[STAGE 1C] ${lang}: merge retriever shards"
  : > "${stage1_jsonl}"
  "${PYTHON_BIN}" - "${stage1_jsonl}" "${stage1_stats}" "${shard_dir}" "${NUM_SHARDS}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

out_jsonl = Path(sys.argv[1])
out_stats = Path(sys.argv[2])
shard_dir = Path(sys.argv[3])
n = int(sys.argv[4])
numeric = Counter()
merged_rows = 0
with out_jsonl.open("w", encoding="utf-8") as fout:
    for i in range(n):
        shard = shard_dir / f"train_timeline_shard{i}_of{n}.jsonl"
        stats_path = shard_dir / f"train_timeline_shard{i}_of{n}.stats.json"
        if not shard.is_file() or shard.stat().st_size == 0:
            raise SystemExit(f"[ERROR] missing/empty shard: {shard}")
        with shard.open(encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    fout.write(line)
                    merged_rows += 1
        cur = json.loads(stats_path.read_text(encoding="utf-8"))
        for k, v in cur.items():
            if isinstance(v, int):
                numeric[k] += v
rows_written = int(numeric.get("rows_written", 0))
if rows_written and rows_written != merged_rows:
    raise SystemExit(f"[ERROR] merged rows mismatch: stats={rows_written} merged={merged_rows}")
stats = dict(numeric)
stats["rows_merged"] = merged_rows
stats["output_jsonl"] = str(out_jsonl)
stats["gt_term_recall"] = stats.get("gt_terms_hit", 0) / stats.get("gt_terms_total", 1) if stats.get("gt_terms_total", 0) else 0.0
stats["avg_term_map_entries_per_chunk"] = stats.get("term_map_entries_total", 0) / stats.get("audio_user_chunks", 1) if stats.get("audio_user_chunks", 0) else 0.0
stats["no_gt_nonempty_term_map_rate"] = stats.get("no_gt_nonempty_term_map_chunks", 0) / stats.get("no_gt_chunks", 1) if stats.get("no_gt_chunks", 0) else 0.0
out_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
PY

  if [[ "${RUN_VARIANT_STAGE}" != "1" ]]; then
    echo "[INFO] ${lang}: RUN_VARIANT_STAGE=${RUN_VARIANT_STAGE}; stopping after stage1."
    continue
  fi

  augment_common=(
    --lang-code "${lang}"
    --augment-prob "${AUGMENT_PROB_OVERRIDE:-0.5}"
    --max-augmented-terms-per-row "${MAX_AUG_TERMS_PER_ROW_OVERRIDE:-8}"
    --sample-count "${SAMPLE_COUNT_OVERRIDE:-120}"
    --missing-gt-policy error
  )
  if [[ "${CACHE_ONLY}" == "1" ]]; then
    augment_common+=(--cache-only)
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    augment_common+=(--dry-run)
  fi

  echo "[STAGE 2] ${lang}: LLM target-translation variant on clean GT"
  "${PYTHON_BIN}" "${AUGMENT_SCRIPT}" \
    --input-jsonl "${stage1_jsonl}" \
    --output-jsonl "${new_v4_train}" \
    --stats-json "${out_dir}/stage2_train_llmvariant_stats.json" \
    --variant-cache-json "${variant_cache}" \
    --sample-json "${out_dir}/stage2_train_llmvariant_samples.json" \
    --seed "${TRAIN_SEED_OVERRIDE:-1606}" \
    "${augment_common[@]}"

  "${PYTHON_BIN}" "${AUGMENT_SCRIPT}" \
    --input-jsonl "${stage1_jsonl}" \
    --output-jsonl "${new_v4_dev}" \
    --stats-json "${out_dir}/stage2_dev_llmvariant_first${DEV_ROWS}_stats.json" \
    --variant-cache-json "${variant_cache}" \
    --sample-json "${out_dir}/stage2_dev_llmvariant_first${DEV_ROWS}_samples.json" \
    --seed "${DEV_SEED_OVERRIDE:-1607}" \
    --max-rows "${DEV_ROWS}" \
    "${augment_common[@]}"

  echo "[STAGE 3] ${lang}: no-GT-zero"
  "${PYTHON_BIN}" "${ZERO_SCRIPT}" \
    --input-jsonl "${new_v4_train}" \
    --output-jsonl "${new_v5_train}" \
    --stats-json "${out_dir}/stage3_train_no_gt_zero_stats.json" \
    --sample-json "${out_dir}/stage3_train_no_gt_zero_samples.json" \
    --missing-gt-policy error
  "${PYTHON_BIN}" "${ZERO_SCRIPT}" \
    --input-jsonl "${new_v4_dev}" \
    --output-jsonl "${new_v5_dev}" \
    --stats-json "${out_dir}/stage3_dev_no_gt_zero_first${DEV_ROWS}_stats.json" \
    --sample-json "${out_dir}/stage3_dev_no_gt_zero_first${DEV_ROWS}_samples.json" \
    --missing-gt-policy error

  echo "[STAGE 4] ${lang}: assistant <term> tags, exact + adjacent-boundary repair only"
  wrap_common=(
    --lang-code "${lang}"
    --tag-template '<term>{translation}</term>'
    --min-target-chars 2
    --max-tags-per-row 16
    --missing-gt-policy error
    --exclude-source-tokens "${EXCLUDE_SOURCE_TOKENS}"
    --exact-require-text-boundaries
    --enable-local-rewrite
    --rewrite-boundary-only
    --rewrite-delay-boundary-min-prefix-chars 2
    --rewrite-require-text-boundaries
    --sample-count 200
  )
  "${PYTHON_BIN}" "${WRAP_SCRIPT}" \
    --input-jsonl "${new_v5_train}" \
    --output-jsonl "${final_train}" \
    --stats-json "${out_dir}/stage4_train_assistant_termtag_boundary_stats.json" \
    --sample-json "${out_dir}/stage4_train_assistant_termtag_boundary_samples.json" \
    "${wrap_common[@]}"
  "${PYTHON_BIN}" "${WRAP_SCRIPT}" \
    --input-jsonl "${new_v5_dev}" \
    --output-jsonl "${final_dev}" \
    --stats-json "${out_dir}/stage4_dev_assistant_termtag_boundary_first${DEV_ROWS}_stats.json" \
    --sample-json "${out_dir}/stage4_dev_assistant_termtag_boundary_first${DEV_ROWS}_samples.json" \
    "${wrap_common[@]}"

  echo "[VALIDATE] ${lang}: strict clean-data gates"
  "${PYTHON_BIN}" - "${lang}" "${out_dir}" "${stage1_stats}" "${summary_json}" "${final_train}" "${final_dev}" <<'PY'
import json
import re
import sys
from pathlib import Path

lang = sys.argv[1]
out = Path(sys.argv[2])
stage1_stats_path = Path(sys.argv[3])
summary_path = Path(sys.argv[4])
final_train = Path(sys.argv[5])
final_dev = Path(sys.argv[6])

def load_path(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

stage0 = load_path(out / "stage0_sourceexact_future_ref_gt_stats.json")
stage1 = load_path(stage1_stats_path)
stage2 = load_path(out / "stage2_train_llmvariant_stats.json")
stage3 = load_path(out / "stage3_train_no_gt_zero_stats.json")
stage4 = load_path(out / "stage4_train_assistant_termtag_boundary_stats.json")

bad = []
if stage0.get("dropped_rows", 0) != 0:
    bad.append(f"stage0 dropped_rows={stage0.get('dropped_rows')}")
if stage0.get("rows_missing_tsv", 0) != 0:
    bad.append(f"stage0 rows_missing_tsv={stage0.get('rows_missing_tsv')}")
if stage4.get("assistant_tag_global_fuzzy_replacements", 0) != 0:
    bad.append(f"global_fuzzy_replacements={stage4.get('assistant_tag_global_fuzzy_replacements')}")

open_count = close_count = latin_boundary_violations = 0
tag_re = re.compile(r"<term>(.*?)</term>")
latin = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]")
sample_bad = []
with final_train.open(encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        for msg_idx, msg in enumerate(obj.get("messages") or []):
            if msg.get("role") != "assistant":
                continue
            text = str(msg.get("content") or "")
            open_count += text.count("<term>")
            close_count += text.count("</term>")
            for m in tag_re.finditer(text):
                before = text[m.start() - 1] if m.start() > 0 else ""
                after = text[m.end()] if m.end() < len(text) else ""
                inner = m.group(1)
                if inner and ((before and latin.match(before) and latin.match(inner[0])) or (after and latin.match(after) and latin.match(inner[-1]))):
                    latin_boundary_violations += 1
                    if len(sample_bad) < 20:
                        sample_bad.append({
                            "line": line_no,
                            "utter_id": obj.get("utter_id"),
                            "msg_idx": msg_idx,
                            "span": text[max(0, m.start()-30):min(len(text), m.end()+30)],
                        })
if open_count != close_count:
    bad.append(f"unbalanced_tags open={open_count} close={close_count}")
if latin_boundary_violations:
    bad.append(f"latin_boundary_violations={latin_boundary_violations}")

summary = {
    "event": f"new_v10_sourceexact_{lang}",
    "policy": {
        "gt_source": "source_chunk_asr_by_chunk exact whole-token source match + future assistant exact target match",
        "dirty_gt_derive_script_used": False,
        "fuzzy_gt_used": False,
        "assistant_tag_policy": "exact target wrap plus adjacent assistant boundary-only repair",
        "global_fuzzy_assistant_rewrite_used": False,
    },
    "outputs": {
        "final_train": str(final_train),
        "final_dev": str(final_dev),
    },
    "stats": {
        "stage0_source_exact_terms_total": stage0.get("source_exact_terms_total"),
        "stage0_future_ref_gt_terms": stage0.get("exact_gt_terms_total"),
        "stage0_target_match_kept_term_rate": stage0.get("target_match_kept_term_rate"),
        "stage0_chunks_with_gt_rate": stage0.get("chunks_with_exact_gt_rate"),
        "stage0_avg_gt_terms_per_chunk": stage0.get("avg_exact_gt_terms_per_chunk"),
        "stage0_source_target_same_gt_terms": stage0.get("source_target_same_gt_terms"),
        "stage1_gt_term_recall": stage1.get("gt_term_recall"),
        "stage1_avg_term_map_entries_per_chunk": stage1.get("avg_term_map_entries_per_chunk"),
        "stage1_no_gt_nonempty_term_map_rate": stage1.get("no_gt_nonempty_term_map_rate"),
        "stage2_augmented_terms": stage2.get("augmented_terms"),
        "stage3_zeroed_chunks": stage3.get("zeroed_chunks"),
        "stage4_assistant_tag_replacements": stage4.get("assistant_tag_replacements"),
        "stage4_assistant_tag_exact_replacements": stage4.get("assistant_tag_exact_replacements"),
        "stage4_boundary_only_replacements": stage4.get("assistant_tag_boundary_only_replacements", 0),
        "stage4_global_fuzzy_replacements": stage4.get("assistant_tag_global_fuzzy_replacements", 0),
        "tag_open_count": open_count,
        "tag_close_count": close_count,
        "latin_boundary_violations": latin_boundary_violations,
    },
    "latin_boundary_violation_samples": sample_bad,
    "validation_errors": bad,
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
if bad:
    raise SystemExit("[ERROR] validation gates failed: " + "; ".join(bad))
PY

  echo "[OK] ${lang}: ${final_train}"
  echo "[OK] ${lang}: ${summary_json}"
done
