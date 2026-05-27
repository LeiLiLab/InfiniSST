#!/bin/bash
#SBATCH --job-name=simuleval_rag_sweep
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --array=0-11
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_simuleval_rag_sweep_baseline.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_simuleval_rag_sweep_baseline.err

set -euo pipefail

# ==================== Environment ====================
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="$CONDA_PREFIX/bin:/mnt/taurus/home/jiaxuanluo/miniconda3/condabin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

# vLLM
export VLLM_USE_V1=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=0

# ==================== Post-eval (stream_laal_term.py) ====================
SUMMARY_LOG="${OUTPUT_BASE}/all_results_summary.log"

GLOSSARIES=(
  "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary_with_translations.json"
  "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060.json"
)

REF_FILE="/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/text/txt/ACL.6060.dev.en-xx.${CUR_LANG}.txt"
AUDIO_YAML="/mnt/taurus/data/siqiouyang/datasets/acl6060/dev.yaml"

# Use flock to avoid concurrent writes to the summary log
exec 200>>"${SUMMARY_LOG}"
flock 200

echo "--------------------------------------------------------" >> "${SUMMARY_LOG}"
echo "TIMESTAMP: $(date +'%Y-%m-%d %H:%M:%S')" >> "${SUMMARY_LOG}"
echo "TASK_ID: ${TASK_ID} | MODEL: ${MODEL_SHORT} | SEG: ${CUR_SEG} | LANG: ${CUR_LANG}" >> "${SUMMARY_LOG}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}" >> "${SUMMARY_LOG}"



for GLOS in "${GLOSSARIES[@]}"; do
  GLOS_NAME=$(basename "${GLOS}")
  echo ">>> Glossary: ${GLOS_NAME}" >> "${SUMMARY_LOG}"
  
  # Run evaluator
  python /home/jiaxuanluo/FBK-fairseq/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py \
    --simuleval-instances "${OUTPUT_PATH}/instances.log" \
    --reference "${REF_FILE}" \
    --audio-yaml "${AUDIO_YAML}" \
    --sacrebleu-tokenizer "${CUR_TOKENIZER}" \
    --latency-unit "${CUR_LATENCY_UNIT}" \
    --glossary "${GLOS}" \
    --term-lang "${CUR_LANG}" \
    --term-mismatch-examples 0 >> "${SUMMARY_LOG}" 2>&1
done

echo "--------------------------------------------------------" >> "${SUMMARY_LOG}"
echo "" >> "${SUMMARY_LOG}"

flock -u 200
exec 200>&-

echo "[INFO] All evaluations for Task ${TASK_ID} DONE"

# ==================== Batch StreamLAAL from existing instances ====================
# If you already have multiple instances.log files and only want StreamLAAL scores (zh/ja/de),
# use:
#
#   python scripts/compute_stream_laal_batch.py \
#     --instances zh:/mnt/gemini/data1/jiaxuanluo/documents/en-zh/offline/instances.log \
#     --instances ja:/mnt/gemini/data1/jiaxuanluo/documents/en-ja/offline/instances.log \
#     --instances de:/mnt/gemini/data1/jiaxuanluo/documents/en-de/offline/instances.log \
#     --glossary /mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary_with_translations.json \
#     --out /mnt/gemini/data1/jiaxuanluo/documents/stream_laal_scores.csv



#     --glossary /mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060.json \
