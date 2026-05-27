#!/usr/bin/env bash
set -euo pipefail

# Orchestrate all evaluation tasks for one density value.
#
# Phase 1: Tagged ACL6060 (4 lm values, sequential)
#   -> offline eval: acl6060 mode
# Phase 2: Per-paper (5 papers x 4 lm = 20 jobs, sequential)
#   -> combine per-paper instances -> offline eval: extracted_by_paper mode
#
# NOTE: vLLM instances cannot run in parallel on the same machine
# (shared memory contention). All SimulEval jobs run sequentially.
#
# All user-facing strings are in English.

# ======Configuration=====
ROOT_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST"
UNIFIED_EVAL_SCRIPT="${ROOT_DIR}/documents/code/simuleval/eval_density_unified.sh"
PREP_SCRIPT="${ROOT_DIR}/documents/code/simuleval/prepare_extracted_glossary_by_paper_inputs.py"
OFFLINE_EVAL_SCRIPT="${ROOT_DIR}/documents/code/offline_sst_eval/offline_streamlaal_eval.py"

# Dataset
DATA_ROOT="/mnt/taurus/data/siqiouyang/datasets/acl6060"
DEV_SOURCE="${DATA_ROOT}/dev.source"
LANG_CODE="zh"

# Per-paper glossaries
EXTRACTED_GLOSSARIES_DIR="${ROOT_DIR}/documents/data/data_pre/extracted_glossaries_by_paper"
EXTRACTED_GLOSSARY_MANIFEST="${ROOT_DIR}/documents/data/data_pre/extracted_glossary_by_paper_manifest.json"

# Output
OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/density_eval_maxsim"

# GPU slot (3 GPUs: 2 for vLLM TP + 1 for RAG)
GPU_SLOT="2,3,4"

# Latency multipliers
LATENCY_MULTIPLIERS=(1 2 3 4)

# Default glossary
GLOSSARY_ACL6060="${ROOT_DIR}/retriever/gigaspeech/data_pre/glossary_acl6060.json"

# Required env vars from caller
DENSITY="${DENSITY:?DENSITY is required}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
RAG_TOP_K="${RAG_TOP_K:?RAG_TOP_K is required}"

# Experiment tracking (see .cursor/rules/experiment_tracking.mdc).
# Required for WandB logging at the end of the run.
NOTES_FILE="${NOTES_FILE:?NOTES_FILE is required (see documents/code/_templates/run_notes_template.md)}"
EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:?EXPERIMENT_FAMILY is required (e.g. sst_density_ablation)}"
DATA_TAG="${DATA_TAG:?DATA_TAG is required (e.g. extracted_by_paper)}"
TRAINED_FROM_RUN="${TRAINED_FROM_RUN:?TRAINED_FROM_RUN is required (WandB run id of the training run)}"
BASELINE_RUN_IDS="${BASELINE_RUN_IDS:-}"
EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-}"
WANDB_PROJECT_EVAL="${WANDB_PROJECT_EVAL:-simuleval_eval}"
RUN_VERDICT="${RUN_VERDICT:-}"
if [[ ! -f "${NOTES_FILE}" ]]; then
    echo "[FATAL] NOTES_FILE not found: ${NOTES_FILE}" >&2
    exit 1
fi

# Optional overrides
RAG_MODEL_PATH_OVERRIDE="${RAG_MODEL_PATH_OVERRIDE:-}"
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE_OVERRIDE:-}"
GPU_SLOT_OVERRIDE="${GPU_SLOT_OVERRIDE:-}"
LATENCY_MULTIPLIERS_OVERRIDE="${LATENCY_MULTIPLIERS_OVERRIDE:-}"
SKIP_PHASE1_TAGGED="${SKIP_PHASE1_TAGGED:-0}"
GLOBAL_GLOSSARY_OVERRIDE="${GLOBAL_GLOSSARY_OVERRIDE:-}"
GLOBAL_INDEX_OVERRIDE="${GLOBAL_INDEX_OVERRIDE:-}"
# ======Configuration=====

if [[ -n "${OUTPUT_BASE_OVERRIDE}" ]]; then
  OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE}"
fi
if [[ -n "${GPU_SLOT_OVERRIDE}" ]]; then
  GPU_SLOT="${GPU_SLOT_OVERRIDE}"
fi
if [[ -n "${LATENCY_MULTIPLIERS_OVERRIDE}" ]]; then
  IFS=' ' read -r -a LATENCY_MULTIPLIERS <<< "${LATENCY_MULTIPLIERS_OVERRIDE}"
fi

LOG_DIR="${OUTPUT_BASE}/${LANG_CODE}/__logs__/d${DENSITY}"
mkdir -p "${LOG_DIR}"

echo "[INFO] ============================================================"
echo "[INFO] One-Density Evaluation Orchestrator (Sequential Mode)"
echo "[INFO] DENSITY=${DENSITY} MODEL_NAME=${MODEL_NAME} RAG_TOP_K=${RAG_TOP_K}"
echo "[INFO] GPU_SLOT=${GPU_SLOT}"
echo "[INFO] LATENCY_MULTIPLIERS=${LATENCY_MULTIPLIERS[*]}"
echo "[INFO] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[INFO] ============================================================"

# ---- Prepare per-paper inputs ----
PAPER_INPUTS_DIR="${OUTPUT_BASE}/${LANG_CODE}/__paper_inputs__"
mkdir -p "${PAPER_INPUTS_DIR}"

echo "[INFO] Preparing per-paper dev lists..."
python3 "${PREP_SCRIPT}" \
  --extracted-glossaries-dir "${EXTRACTED_GLOSSARIES_DIR}" \
  --extracted-glossary-manifest "${EXTRACTED_GLOSSARY_MANIFEST}" \
  --data-root "${DATA_ROOT}" \
  --lang-code "${LANG_CODE}" \
  --output-dir "${PAPER_INPUTS_DIR}"

MAP_JSON="${PAPER_INPUTS_DIR}/paper_inputs_map.json"
if [[ ! -f "${MAP_JSON}" ]]; then
  echo "[ERROR] paper_inputs_map.json not found: ${MAP_JSON}" >&2
  exit 2
fi

PAPERS=($(python3 -c "
import json
with open('${MAP_JSON}') as f:
    obj = json.load(f)
for p in sorted(obj.get('papers', {}).keys()):
    print(p)
"))
# Optional override for quick diagnostic runs: space/comma-separated paper ids.
# When set, we keep only the intersection (assertion fires if any id is missing).
if [[ -n "${RUN_PAPERS_OVERRIDE:-}" ]]; then
  IFS=$', \t' read -r -a _RUN_PAPERS_SELECTED <<< "${RUN_PAPERS_OVERRIDE}"
  _ALL_PAPERS=("${PAPERS[@]}")
  PAPERS=()
  for _p in "${_RUN_PAPERS_SELECTED[@]}"; do
    _found=0
    for _ap in "${_ALL_PAPERS[@]}"; do
      if [[ "${_ap}" == "${_p}" ]]; then
        PAPERS+=("${_p}")
        _found=1
        break
      fi
    done
    if [[ "${_found}" -ne 1 ]]; then
      echo "[ERROR] RUN_PAPERS_OVERRIDE paper not in MAP_JSON: ${_p}" >&2
      exit 2
    fi
  done
  echo "[INFO] RUN_PAPERS_OVERRIDE applied -> ${PAPERS[*]}"
fi
echo "[INFO] Papers: ${PAPERS[*]}"

read_paper_field() {
  local paper_id="$1"
  local field="$2"
  python3 -c "
import json
with open('${MAP_JSON}') as f:
    obj = json.load(f)
print(obj['papers']['${paper_id}']['${field}'])
"
}

run_eval_sequential() {
  local lm="$1"
  local log_file="$2"
  shift 2
  local extra_env=("$@")

  echo "[INFO] Run: lm=${lm} gpu=${GPU_SLOT} -> ${log_file}"

  (
    export MODEL_NAME_OVERRIDE="${MODEL_NAME}"
    export RAG_TOP_K_OVERRIDE="${RAG_TOP_K}"
    export RAG_MODEL_PATH_OVERRIDE="${RAG_MODEL_PATH_OVERRIDE}"
    export OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}"
    export CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${GPU_SLOT}"
    export LATENCY_MULTIPLIER_OVERRIDE="${lm}"
    export DENSITY_TAG="${DENSITY}"
    export CLEAN_OUTPUT_DIR_OVERRIDE="0"
    export SKIP_OFFLINE_EVAL="1"
    for kv in "${extra_env[@]}"; do
      export "${kv}"
    done
    bash "${UNIFIED_EVAL_SCRIPT}"
  ) > "${log_file}" 2>&1

  local rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "[WARN] Job exited with code ${rc}: ${log_file}"
    tail -5 "${log_file}" 2>/dev/null
  else
    echo "[INFO] Job completed successfully: ${log_file}"
  fi
  return ${rc}
}

export MWERSEGMENTER_ROOT="/mnt/taurus/home/jiaxuanluo/mwerSegmenter"
export PATH="${MWERSEGMENTER_ROOT}:${PATH}"

GLOSSARY_TAG="$(basename "${GLOSSARY_ACL6060}" .json)"

# ====================================================================
# Phase 1: Tagged ACL6060 (sequential)
# ====================================================================
if [[ "${SKIP_PHASE1_TAGGED}" == "1" ]]; then
  echo ""
  echo "[INFO] ===== Phase 1: SKIPPED (SKIP_PHASE1_TAGGED=1) ====="
else
  echo ""
  echo "[INFO] ===== Phase 1: Tagged ACL6060 ====="

  for lm in "${LATENCY_MULTIPLIERS[@]}"; do
    TAGGED_DIR="${OUTPUT_BASE}/${LANG_CODE}/d${DENSITY}_lm${lm}_k${RAG_TOP_K}_g${GLOSSARY_TAG}"
    INST_LOG="${TAGGED_DIR}/instances.log"

    if [[ -f "${INST_LOG}" ]] && [[ -s "${INST_LOG}" ]]; then
      echo "[INFO] Skip tagged lm=${lm}: instances.log already exists"
      continue
    fi

    log_file="${LOG_DIR}/tagged_lm${lm}.log"
    run_eval_sequential "${lm}" "${log_file}" || true
  done

  echo "[INFO] All tagged SimulEval jobs done. Running offline eval..."

  for lm in "${LATENCY_MULTIPLIERS[@]}"; do
    TAGGED_DIR="${OUTPUT_BASE}/${LANG_CODE}/d${DENSITY}_lm${lm}_k${RAG_TOP_K}_g${GLOSSARY_TAG}"
    INST_LOG="${TAGGED_DIR}/instances.log"
    if [[ -f "${INST_LOG}" ]] && [[ -s "${INST_LOG}" ]]; then
      echo "[INFO] Offline eval: tagged lm=${lm}"
      python3 "${OFFLINE_EVAL_SCRIPT}" \
      --mode acl6060 \
      --instances-log "${INST_LOG}" \
      --lang-code "${LANG_CODE}" \
      --glossary-acl6060 "${GLOSSARY_ACL6060}" \
      --output-tsv "${TAGGED_DIR}/eval_results.tsv" \
      --output-log "${TAGGED_DIR}/eval_results.log" \
      2>&1 | tee "${LOG_DIR}/tagged_offline_lm${lm}.log" \
      || echo "[WARN] Tagged offline eval failed for lm=${lm}"
  else
    echo "[WARN] No instances.log for tagged lm=${lm}: ${INST_LOG}"
  fi
done
fi

# ====================================================================
# Phase 2: Per-paper (sequential)
# ====================================================================
echo ""
echo "[INFO] ===== Phase 2: Per-Paper SimulEval ====="

for lm in "${LATENCY_MULTIPLIERS[@]}"; do
  for pid in "${PAPERS[@]}"; do
    pp_gloss="$(read_paper_field "${pid}" "glossary_path")"
    src="$(read_paper_field "${pid}" "src_list")"
    tgt="$(read_paper_field "${pid}" "tgt_list")"

    # Use global glossary override for scaled-glossary experiments
    if [[ -n "${GLOBAL_GLOSSARY_OVERRIDE}" ]]; then
      gloss="${GLOBAL_GLOSSARY_OVERRIDE}"
    else
      gloss="${pp_gloss}"
    fi

    if [[ ! -f "${gloss}" ]] || [[ ! -f "${src}" ]]; then
      echo "[WARN] Skipping ${pid} lm=${lm}: missing files"
      continue
    fi

    PP_GLOSS_TAG="$(basename "${gloss}" .json)"
    PP_DIR="${OUTPUT_BASE}/${LANG_CODE}/d${DENSITY}_lm${lm}_k${RAG_TOP_K}_g${PP_GLOSS_TAG}_pp${pid}"
    PP_INST="${PP_DIR}/instances.log"

    if [[ -f "${PP_INST}" ]] && [[ -s "${PP_INST}" ]]; then
      echo "[INFO] Skip pp ${pid} lm=${lm}: instances.log already exists"
      continue
    fi

    EXTRA_ENVS=(
      "GLOSSARY_PATH_OVERRIDE=${gloss}"
      "SRC_LIST_OVERRIDE=${src}"
      "TGT_LIST_OVERRIDE=${tgt}"
      "PAPER_ID_TAG=${pid}"
      "EVAL_MODE_OVERRIDE=acl6060"
    )
    if [[ -n "${GLOBAL_INDEX_OVERRIDE}" ]]; then
      EXTRA_ENVS+=("INDEX_PATH_OVERRIDE=${GLOBAL_INDEX_OVERRIDE}")
    fi

    log_file="${LOG_DIR}/pp_${pid}_lm${lm}.log"
    run_eval_sequential "${lm}" "${log_file}" "${EXTRA_ENVS[@]}" || true
  done
done

echo "[INFO] All per-paper SimulEval jobs done."

# ====================================================================
# Phase 3: Combine per-paper instances + extracted_by_paper offline eval
# ====================================================================
echo ""
echo "[INFO] ===== Phase 3: Combine per-paper & extracted_by_paper eval ====="

for lm in "${LATENCY_MULTIPLIERS[@]}"; do
  echo "[INFO] Combining per-paper instances for lm=${lm}..."
  COMBINED_DIR="${OUTPUT_BASE}/${LANG_CODE}/d${DENSITY}_lm${lm}_k${RAG_TOP_K}_per_paper_combined"
  mkdir -p "${COMBINED_DIR}"
  COMBINED_INST="${COMBINED_DIR}/instances.log"

  ALL_PP_FOUND=true
  for pid in "${PAPERS[@]}"; do
    if [[ -n "${GLOBAL_GLOSSARY_OVERRIDE}" ]]; then
      PP_GLOSS_TAG="$(basename "${GLOBAL_GLOSSARY_OVERRIDE}" .json)"
    else
      PP_GLOSS_TAG="$(basename "$(read_paper_field "${pid}" "glossary_path")" .json)"
    fi
    PP_DIR="${OUTPUT_BASE}/${LANG_CODE}/d${DENSITY}_lm${lm}_k${RAG_TOP_K}_g${PP_GLOSS_TAG}_pp${pid}"
    PP_INST="${PP_DIR}/instances.log"
    if [[ ! -f "${PP_INST}" ]] || [[ ! -s "${PP_INST}" ]]; then
      echo "[WARN] Missing per-paper instances.log: ${PP_INST}"
      ALL_PP_FOUND=false
    fi
  done

  if [[ "${ALL_PP_FOUND}" == "false" ]]; then
    echo "[WARN] Skipping combine for lm=${lm}: some per-paper instances missing"
    continue
  fi

  python3 -c "
import json, os, sys
from pathlib import Path

with open('${MAP_JSON}') as f:
    obj = json.load(f)

output_base = '${OUTPUT_BASE}/${LANG_CODE}'
density = '${DENSITY}'
lm = '${lm}'
rag_top_k = '${RAG_TOP_K}'
global_glossary_override = '${GLOBAL_GLOSSARY_OVERRIDE}'

src_lines = Path('${DEV_SOURCE}').read_text().strip().splitlines()
paper_order = []
for line in src_lines:
    base = os.path.basename(line.strip())
    if base.lower().endswith('.wav'):
        base = base[:-len('.wav')]
    paper_order.append(base)

paper_instances = {}
for pid in set(paper_order):
    if global_glossary_override:
        gloss_tag = os.path.basename(global_glossary_override).replace('.json', '')
    else:
        gloss_path = obj['papers'][pid]['glossary_path']
        gloss_tag = os.path.basename(gloss_path).replace('.json', '')
    pp_dir = os.path.join(output_base, f'd{density}_lm{lm}_k{rag_top_k}_g{gloss_tag}_pp{pid}')
    inst_file = os.path.join(pp_dir, 'instances.log')
    assert os.path.isfile(inst_file), f'Missing: {inst_file}'
    lines = Path(inst_file).read_text().strip().splitlines()
    paper_instances[pid] = lines
    print(f'[INFO] Paper {pid}: {len(lines)} instances')

counters = {pid: 0 for pid in set(paper_order)}
combined = []
for pid in paper_order:
    idx = counters[pid]
    assert idx < len(paper_instances[pid]), f'Exhausted instances for {pid}'
    combined.append(paper_instances[pid][idx])
    counters[pid] = idx + 1

out = '${COMBINED_INST}'
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text('\n'.join(combined) + '\n')
print(f'[INFO] Combined {len(combined)} instances -> {out}')
" 2>&1

  if [[ ! -f "${COMBINED_INST}" ]] || [[ ! -s "${COMBINED_INST}" ]]; then
    echo "[WARN] Combined instances.log empty/missing for lm=${lm}"
    continue
  fi

  # Combine per-paper runtime JSONL logs into the combined directory
  COMBINED_RUNTIME="${COMBINED_DIR}/runtime_omni_vllm_maxsim_rag_combined_lm${lm}.jsonl"
  > "${COMBINED_RUNTIME}"
  for pid in "${PAPERS[@]}"; do
    if [[ -n "${GLOBAL_GLOSSARY_OVERRIDE}" ]]; then
      PP_GLOSS_TAG="$(basename "${GLOBAL_GLOSSARY_OVERRIDE}" .json)"
    else
      PP_GLOSS_TAG="$(basename "$(read_paper_field "${pid}" "glossary_path")" .json)"
    fi
    PP_DIR="${OUTPUT_BASE}/${LANG_CODE}/d${DENSITY}_lm${lm}_k${RAG_TOP_K}_g${PP_GLOSS_TAG}_pp${pid}"
    for rt_file in "${PP_DIR}"/runtime_omni_vllm_maxsim_rag_*.jsonl; do
      if [[ -f "${rt_file}" ]]; then
        cat "${rt_file}" >> "${COMBINED_RUNTIME}"
      fi
    done
  done
  if [[ -s "${COMBINED_RUNTIME}" ]]; then
    echo "[INFO] Combined runtime JSONL for lm=${lm}: $(wc -l < "${COMBINED_RUNTIME}") lines"
  else
    echo "[INFO] No runtime JSONL found for lm=${lm} (TCR will be N/A)"
    rm -f "${COMBINED_RUNTIME}"
  fi

  echo "[INFO] Running extracted_by_paper offline eval for lm=${lm}..."
  python3 "${OFFLINE_EVAL_SCRIPT}" \
    --mode extracted_by_paper \
    --instances-log "${COMBINED_INST}" \
    --lang-code "${LANG_CODE}" \
    --glossary-acl6060 "${GLOSSARY_ACL6060}" \
    --extracted-glossary-manifest "${EXTRACTED_GLOSSARY_MANIFEST}" \
    --output-tsv "${COMBINED_DIR}/eval_results_by_paper.tsv" \
    --output-log "${COMBINED_DIR}/eval_results_by_paper.log" \
    2>&1 | tee "${LOG_DIR}/pp_combined_offline_lm${lm}.log" \
    || echo "[WARN] extracted_by_paper offline eval failed for lm=${lm}"
done

# ====================================================================
# Summary
# ====================================================================
echo ""
echo "[INFO] Collecting summary for density=${DENSITY}..."

SUMMARY_TSV="${OUTPUT_BASE}/${LANG_CODE}/d${DENSITY}_summary.tsv"
{
  echo -e "density\tlm\tmode\tBLEU\tStreamLAAL\tStreamLAAL_CA\tTERM_ACC\tTCR\tTERM_FCR\toutput_dir"

  for lm in "${LATENCY_MULTIPLIERS[@]}"; do
    TAGGED_DIR="${OUTPUT_BASE}/${LANG_CODE}/d${DENSITY}_lm${lm}_k${RAG_TOP_K}_g${GLOSSARY_TAG}"
    TSV_FILE="${TAGGED_DIR}/eval_results.tsv"
    if [[ -f "${TSV_FILE}" ]]; then
      ROW="$(tail -1 "${TSV_FILE}")"
      BLEU="$(echo "${ROW}" | cut -f3)"
      SLAAL="$(echo "${ROW}" | cut -f4)"
      SLAAL_CA="$(echo "${ROW}" | cut -f5)"
      TACC="$(echo "${ROW}" | cut -f6)"
      TCR_VAL="$(echo "${ROW}" | cut -f9)"
      TFCR="$(echo "${ROW}" | cut -f12)"
      echo -e "${DENSITY}\t${lm}\ttagged\t${BLEU}\t${SLAAL}\t${SLAAL_CA}\t${TACC}\t${TCR_VAL}\t${TFCR}\t${TAGGED_DIR}"
    fi

    COMBINED_DIR="${OUTPUT_BASE}/${LANG_CODE}/d${DENSITY}_lm${lm}_k${RAG_TOP_K}_per_paper_combined"
    BP_TSV="${COMBINED_DIR}/eval_results_by_paper.tsv"
    if [[ -f "${BP_TSV}" ]]; then
      ROW="$(tail -1 "${BP_TSV}")"
      BLEU="$(echo "${ROW}" | cut -f3)"
      SLAAL="$(echo "${ROW}" | cut -f4)"
      SLAAL_CA="$(echo "${ROW}" | cut -f5)"
      TACC="$(echo "${ROW}" | cut -f6)"
      TCR_VAL="$(echo "${ROW}" | cut -f9)"
      TFCR="$(echo "${ROW}" | cut -f12)"
      echo -e "${DENSITY}\t${lm}\tby_paper\t${BLEU}\t${SLAAL}\t${SLAAL_CA}\t${TACC}\t${TCR_VAL}\t${TFCR}\t${COMBINED_DIR}"
    fi
  done
} > "${SUMMARY_TSV}"

echo "[INFO] Summary:"
cat "${SUMMARY_TSV}"
echo ""

# ====================================================================
# WandB logging (experiment tracking)
# ====================================================================
WANDB_LOGGER="${ROOT_DIR}/documents/code/offline_evaluation/wandb_eval_logger.py"
EVAL_RUN_NAME="d${DENSITY}__lms$(IFS=-; echo "${LATENCY_MULTIPLIERS[*]}")__k${RAG_TOP_K}__${LANG_CODE}__$(date +%Y%m%d-%H%M)"

WANDB_EXTRA_TAG_ARGS=()
if [[ -n "${EXTRA_WANDB_TAGS}" ]]; then
  IFS=' ' read -r -a _EXTRA_TAGS <<< "${EXTRA_WANDB_TAGS}"
  WANDB_EXTRA_TAG_ARGS+=(--extra-tags "${_EXTRA_TAGS[@]}")
fi
WANDB_BASELINE_ARGS=()
if [[ -n "${BASELINE_RUN_IDS}" ]]; then
  IFS=' ' read -r -a _BASELINES <<< "${BASELINE_RUN_IDS}"
  WANDB_BASELINE_ARGS+=(--baseline-run-ids "${_BASELINES[@]}")
fi
WANDB_VERDICT_ARGS=()
if [[ -n "${RUN_VERDICT}" ]]; then
  WANDB_VERDICT_ARGS+=(--verdict "${RUN_VERDICT}")
fi

echo "[INFO] Logging eval run to WandB project=${WANDB_PROJECT_EVAL} name=${EVAL_RUN_NAME}"
python3 "${WANDB_LOGGER}" \
  --project "${WANDB_PROJECT_EVAL}" \
  --run-name "${EVAL_RUN_NAME}" \
  --experiment-family "${EXPERIMENT_FAMILY}" \
  --data-tag "${DATA_TAG}" \
  --notes-file "${NOTES_FILE}" \
  --trained-from-run "${TRAINED_FROM_RUN}" \
  --density "${DENSITY}" \
  --rag-top-k "${RAG_TOP_K}" \
  --output-base "${OUTPUT_BASE}" \
  --lang-code "${LANG_CODE}" \
  --latency-multipliers "${LATENCY_MULTIPLIERS[@]}" \
  --glossary-tag "${GLOSSARY_TAG}" \
  --model-name "${MODEL_NAME}" \
  --rag-model-path "${RAG_MODEL_PATH_OVERRIDE}" \
  "${WANDB_BASELINE_ARGS[@]}" \
  "${WANDB_EXTRA_TAG_ARGS[@]}" \
  "${WANDB_VERDICT_ARGS[@]}" \
  || echo "[WARN] WandB logging failed (see above). Metrics still on disk at ${SUMMARY_TSV}."

echo "[INFO] ============================================================"
echo "[INFO] Density ${DENSITY} evaluation complete."
echo "[INFO] ============================================================"
echo "[INFO] All done for density=${DENSITY}."
