#!/usr/bin/env bash
set -euo pipefail

# Launch d=5 full evaluation (tagged + per-paper, lm=1,2,3,4)
# Sequential mode - one SimulEval at a time to avoid vLLM contention.
#
# Run with:
#   nohup bash launch_d5_eval.sh > /mnt/gemini/data2/jiaxuanluo/density_eval_maxsim/zh/__logs__/d5_master.log 2>&1 &

export DENSITY="5"
export MODEL_NAME="/mnt/aries/data4/jiaxuanluo/speech_llm_density_ablation/d5/r16/v0-20260414-010020-hf"
export RAG_TOP_K="10"
export RAG_MODEL_PATH_OVERRIDE="/mnt/taurus/data/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs10752_t=0.03_3var_clean_gc_wr1000k_m0.1_maxsim_sp07_best_acl6060_gs10000.pt"
export OUTPUT_BASE_OVERRIDE="/mnt/gemini/data2/jiaxuanluo/density_eval_maxsim"
export GPU_SLOT_OVERRIDE="2,3,4"
export LATENCY_MULTIPLIERS_OVERRIDE="1 2 3 4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_one_density_eval.sh"
