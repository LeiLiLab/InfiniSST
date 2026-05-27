#!/usr/bin/env bash
set -euo pipefail

# Phase 0.5: Inference-time top-k ablation on existing d5 r16 model.
#
# Goal: before committing to a 6-12h adversarial training job, answer:
#   "Does the existing d5 model get better when we reduce distractors?"
#
# Approach: reuse existing d5 HF checkpoint (no retraining), loop top-k in
# {3,5,7,10}, evaluate on 5 papers (lm=1, default per-paper glossary, stride=1.92
# i.e. no sliding overlap because stride >= chunk length), then aggregate
# TERM_ACC + TCR per (k, paper).
#
# IMPORTANT: we RE-RUN k=10 instead of reusing the existing disk output because
# the old k=10 was produced before the sliding-window integration and did not
# explicitly set RAG_RETRIEVE_STRIDE_SEC. Re-running ensures consistent retrieval
# behavior across all k values for a fair A/B comparison.
#
# All user-facing strings are in English.

# ======Configuration=====
EXIT_CONFIG_ERROR="2"
EXIT_DATA_ERROR="3"

ROOT_DIR="/home/jiaxuanluo/InfiniSST"
RUN_SCRIPT="${ROOT_DIR}/documents/code/simuleval/run_one_density_eval.sh"

# Model under test (existing d5 r16 checkpoint, same one used in k=10 baseline)
D5_MODEL="/mnt/aries/data4/jiaxuanluo/speech_llm_density_ablation/d5/r16/v0-20260414-010020-hf"

# MaxSim retriever checkpoint (same as existing k=10 run for fair A/B)
RAG_MODEL="/mnt/taurus/data/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs10752_t=0.03_3var_clean_gc_wr1000k_m0.1_maxsim_sp07_best_acl6060_gs10000.pt"

# Output base matches existing k=10 runs so reuse is automatic
OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/density_eval_maxsim_fixed"

# GPUs: 7 and 5 are fully free (TP=2); 6 has ~12GB free (for RAG, needs ~5GB).
GPUS="7,5,6"

# Fixed experiment knobs
LATENCY_MULTIPLIER="1"
DENSITY="5"

# Retrieval stride. With VLLM_SEGMENT_SEC=0.96s, setting stride=1.92 means the
# retriever runs at half the vLLM rate and encodes a 1.92s window each call,
# which is equivalent to "no sliding overlap" (window covers a full two-chunk
# vLLM turn). This matches the plan's stride=window / no sliding window intent.
RAG_RETRIEVE_STRIDE_SEC="1.92"

# Taurus P2P corruption workaround: force NCCL (+ host memory) all-reduce instead
# of vLLM's custom-all-reduce which goes through CUDA IPC/P2P. Without this, TP=2
# on GPU5/7 (NODE topo, PCIe via host bridge) produces NaN logits -> model emits
# all "!" tokens. Diagnostic confirmed 2026-04-17: same config with this flag
# enabled produces valid Chinese output. Safe to leave on everywhere on this node.
VLLM_DISABLE_CUSTOM_ALL_REDUCE="1"

# The top-k values to sweep. All are re-run with the same stride for fairness.
TOPK_VALUES=(3 5 7 10)

# Papers (per-paper evaluation via default per-paper extracted glossary)
PAPERS=(
  "2022.acl-long.110"
  "2022.acl-long.117"
  "2022.acl-long.268"
  "2022.acl-long.367"
  "2022.acl-long.590"
)

LOG_ROOT="${OUTPUT_BASE}/zh/__logs__/topk_ablation_$(date +%Y%m%d_%H%M%S)"

# Final summary TSV (aggregated across top-k values)
SUMMARY_TSV="${OUTPUT_BASE}/zh/d${DENSITY}_lm${LATENCY_MULTIPLIER}_topk_ablation_summary.tsv"
# ======Configuration=====

mkdir -p "${LOG_ROOT}"

echo "[INFO] ============================================================"
echo "[INFO] Phase 0.5: d5 r16 top-k Ablation (cheap info gate)"
echo "[INFO] MODEL=${D5_MODEL}"
echo "[INFO] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[INFO] GPUS=${GPUS}"
echo "[INFO] TOPK_VALUES=${TOPK_VALUES[*]}"
echo "[INFO] RAG_RETRIEVE_STRIDE_SEC=${RAG_RETRIEVE_STRIDE_SEC}"
echo "[INFO] VLLM_DISABLE_CUSTOM_ALL_REDUCE=${VLLM_DISABLE_CUSTOM_ALL_REDUCE}"
echo "[INFO] LOG_ROOT=${LOG_ROOT}"
echo "[INFO] ============================================================"

# Validation
if [[ ! -d "${D5_MODEL}" ]]; then
  echo "[ERROR] D5 model dir not found: ${D5_MODEL}" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi
if [[ ! -f "${RAG_MODEL}" ]]; then
  echo "[ERROR] RAG model not found: ${RAG_MODEL}" >&2
  exit "${EXIT_DATA_ERROR}"
fi
if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "[ERROR] RUN_SCRIPT not found: ${RUN_SCRIPT}" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi

# ---- Run per-k jobs (reuse run_one_density_eval.sh orchestrator) ----
for k in "${TOPK_VALUES[@]}"; do
  combined_dir="${OUTPUT_BASE}/zh/d${DENSITY}_lm${LATENCY_MULTIPLIER}_k${k}_per_paper_combined"
  combined_tsv="${combined_dir}/eval_results_by_paper.tsv"
  if [[ -f "${combined_tsv}" ]] && [[ -s "${combined_tsv}" ]]; then
    echo "[SKIP] k=${k} already has combined eval results: ${combined_tsv}"
    continue
  fi

  log_file="${LOG_ROOT}/k${k}.log"
  echo ""
  echo "[RUN] k=${k} -> ${log_file} at $(date '+%H:%M:%S')"

  (
    export DENSITY="${DENSITY}"
    export MODEL_NAME="${D5_MODEL}"
    export RAG_TOP_K="${k}"
    export RAG_MODEL_PATH_OVERRIDE="${RAG_MODEL}"
    export OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}"
    export GPU_SLOT_OVERRIDE="${GPUS}"
    export LATENCY_MULTIPLIERS_OVERRIDE="${LATENCY_MULTIPLIER}"
    export SKIP_PHASE1_TAGGED="1"
    # Propagate retrieval stride through nested bash invocations
    # (run_one_density_eval.sh -> eval_density_unified.sh).
    export RAG_RETRIEVE_STRIDE_SEC_OVERRIDE="${RAG_RETRIEVE_STRIDE_SEC}"
    # Disable vLLM custom-all-reduce to avoid corrupted P2P on this taurus node.
    # The agent reads this env var and passes disable_custom_all_reduce=True to LLM().
    export VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE}"
    bash "${RUN_SCRIPT}"
  ) > "${log_file}" 2>&1
  rc="$?"
  if [[ "${rc}" != "0" ]]; then
    echo "[WARN] k=${k} exited with code ${rc}; inspect ${log_file}"
    tail -20 "${log_file}" >&2 || true
  else
    echo "[DONE] k=${k} at $(date '+%H:%M:%S')"
  fi
done

# ---- Aggregate summary across all k values ----
echo ""
echo "[INFO] Aggregating summary TSV: ${SUMMARY_TSV}"

ALL_K=("${TOPK_VALUES[@]}")

{
  printf "density\tlm\tk\tBLEU\tStreamLAAL\tStreamLAAL_CA\tTERM_ACC\tTERM_CORRECT\tTERM_TOTAL\tTCR\tTCR_ADOPTED\tTCR_TOTAL\tTERM_FCR\tFALSE_COPY\tNEG_TOTAL\tdir\n"
  for k in "${ALL_K[@]}"; do
    bp_tsv="${OUTPUT_BASE}/zh/d${DENSITY}_lm${LATENCY_MULTIPLIER}_k${k}_per_paper_combined/eval_results_by_paper.tsv"
    if [[ -f "${bp_tsv}" ]] && [[ -s "${bp_tsv}" ]]; then
      row="$(tail -1 "${bp_tsv}")"
      # Columns in eval_results_by_paper.tsv:
      # 1 mode 2 lang_code 3 BLEU 4 StreamLAAL 5 StreamLAAL_CA 6 TERM_ACC
      # 7 TERM_CORRECT 8 TERM_TOTAL 9 TCR 10 TCR_ADOPTED 11 TCR_TOTAL
      # 12 TERM_FCR 13 FALSE_COPY 14 NEG_TOTAL 15 instances_log
      bleu="$(printf '%s' "${row}" | cut -f3)"
      slaal="$(printf '%s' "${row}" | cut -f4)"
      slaal_ca="$(printf '%s' "${row}" | cut -f5)"
      term_acc="$(printf '%s' "${row}" | cut -f6)"
      term_correct="$(printf '%s' "${row}" | cut -f7)"
      term_total="$(printf '%s' "${row}" | cut -f8)"
      tcr="$(printf '%s' "${row}" | cut -f9)"
      tcr_adopted="$(printf '%s' "${row}" | cut -f10)"
      tcr_total="$(printf '%s' "${row}" | cut -f11)"
      term_fcr="$(printf '%s' "${row}" | cut -f12)"
      false_copy="$(printf '%s' "${row}" | cut -f13)"
      neg_total="$(printf '%s' "${row}" | cut -f14)"
      dir="${OUTPUT_BASE}/zh/d${DENSITY}_lm${LATENCY_MULTIPLIER}_k${k}_per_paper_combined"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${DENSITY}" "${LATENCY_MULTIPLIER}" "${k}" \
        "${bleu}" "${slaal}" "${slaal_ca}" \
        "${term_acc}" "${term_correct}" "${term_total}" \
        "${tcr}" "${tcr_adopted}" "${tcr_total}" \
        "${term_fcr}" "${false_copy}" "${neg_total}" \
        "${dir}"
    else
      echo "[WARN] Missing eval_results_by_paper.tsv for k=${k}: ${bp_tsv}" >&2
    fi
  done
} > "${SUMMARY_TSV}"

echo ""
echo "[INFO] ============================================================"
echo "[INFO] Phase 0.5 Summary:"
echo "[INFO] ============================================================"
cat "${SUMMARY_TSV}"
echo ""
echo "[INFO] Summary TSV: ${SUMMARY_TSV}"
echo "[INFO] Per-k logs: ${LOG_ROOT}"
echo "[INFO] ============================================================"
echo "[INFO] Decision gate:"
echo "[INFO]   - k=3 >> k=10 TERM_ACC: model sensitive to distractors; adversarial viable"
echo "[INFO]   - k=3 == k=10 TERM_ACC: model globally ignores term_map; need stronger intervention"
echo "[INFO]   - k=3 << k=10 TERM_ACC: recall collapse; fix retriever precision first"
echo "[INFO] ============================================================"
