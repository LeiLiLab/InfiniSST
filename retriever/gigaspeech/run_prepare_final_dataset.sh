#!/bin/bash
#SBATCH --job-name=prepare_dataset
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_prepare_dataset.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_prepare_dataset.err

set -euo pipefail

# 环境设置
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="$CONDA_PREFIX/bin:/mnt/taurus/home/jiaxuanluo/miniconda3/condabin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

usage() {
    cat <<EOF
Usage:
  $(basename "$0") [train|dev] [options]

Options:
  --mode <train|dev>                 Dataset split (default: dev). Also supports positional first arg.
  --m <1|2|3|4|...>                  Multiplier merge id. If set, default paths will use *_m{m}.
  --input-tsv <path>                 Input aligned TSV.
  --pos-jsonl <path>                 Positive dataset JSONL (sampled positives).
  --full-pos-jsonl <path>            Full term JSONL used as global term library (default: same as --pos-jsonl).
  --output-dir <path>                Directory that contains chunk wavs; will be used to set chunk_audio_path.
  --neg-jsonl <path>                 Output JSONL for pure no-term samples.
  --final-jsonl <path>               Output JSONL for merged final dataset.
  --write-neg-audio                  If set, materialize wavs for NO_TERM samples into --output-dir.
  --ratio <float>                    Negative ratio in final dataset (implies --no-all-neg).
  --all-neg                          Use all negative samples (default).
  --no-all-neg                       Disable all-neg mode (use --ratio).
  -h, --help                         Show this help.

Notes:
  - This script generates NO_TERM entries (JSONL) and merges them into the final dataset.
  - By default, it does NOT generate wav files; it only writes chunk_audio_path pointing into --output-dir.
  - If you include NO_TERM samples in training, you likely want --write-neg-audio to avoid missing wav paths.
EOF
}

# Backward-compatible positional mode: ./run_prepare_final_dataset.sh dev
MODE="dev"
if [[ $# -gt 0 && "${1:-}" != "-"* ]]; then
    if [[ "$1" == "train" || "$1" == "dev" ]]; then
        MODE="$1"
        shift
    fi
fi

M=""
INPUT_TSV=""
POS_JSONL=""
FULL_POS_JSONL=""
NEG_JSONL=""
FINAL_JSONL=""
OUTPUT_DIR=""

USE_ALL_NEG=true
RATIO="0.5"
WRITE_NEG_AUDIO=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"; shift 2 ;;
        --m)
            M="$2"; shift 2 ;;
        --input-tsv)
            INPUT_TSV="$2"; shift 2 ;;
        --pos-jsonl)
            POS_JSONL="$2"; shift 2 ;;
        --full-pos-jsonl|--full-term-jsonl)
            FULL_POS_JSONL="$2"; shift 2 ;;
        --output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        --neg-jsonl)
            NEG_JSONL="$2"; shift 2 ;;
        --final-jsonl|--output-jsonl)
            FINAL_JSONL="$2"; shift 2 ;;
        --write-neg-audio)
            WRITE_NEG_AUDIO=true
            shift ;;
        --ratio)
            RATIO="$2"
            USE_ALL_NEG=false
            shift 2 ;;
        --all-neg|--use-all-neg)
            USE_ALL_NEG=true
            shift ;;
        --no-all-neg)
            USE_ALL_NEG=false
            shift ;;
        -h|--help)
            usage
            exit 0 ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage
            exit 2 ;;
    esac
done

if [[ "$MODE" != "train" && "$MODE" != "dev" ]]; then
    echo "[ERROR] --mode must be 'train' or 'dev', got: ${MODE}" >&2
    exit 2
fi

BASE_DIR="/mnt/gemini/data2/jiaxuanluo"
IN_BASE_DIR="/mnt/gemini/data1/jiaxuanluo"

if [[ "$MODE" == "train" ]]; then
    : "${INPUT_TSV:=${IN_BASE_DIR}/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv}"
    if [[ -n "${M}" ]]; then
        : "${POS_JSONL:=${IN_BASE_DIR}/term_train_dataset_m${M}.jsonl}"
        : "${OUTPUT_DIR:=${BASE_DIR}/term_train_audio_chunks_m${M}}"
        : "${NEG_JSONL:=${BASE_DIR}/term_train_dataset_all_neg_m${M}.jsonl}"
        : "${FINAL_JSONL:=${BASE_DIR}/term_train_dataset_final_m${M}.jsonl}"
    else
        : "${POS_JSONL:=${IN_BASE_DIR}/term_train_dataset_v3.jsonl}"
        : "${OUTPUT_DIR:=${BASE_DIR}/term_train_audio_chunks}"
        : "${NEG_JSONL:=${BASE_DIR}/term_train_dataset_all_neg.jsonl}"
        : "${FINAL_JSONL:=${BASE_DIR}/term_train_dataset_final.jsonl}"
    fi
else
    : "${INPUT_TSV:=${IN_BASE_DIR}/dev_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv}"
    if [[ -n "${M}" ]]; then
        : "${POS_JSONL:=${IN_BASE_DIR}/term_dev_dataset_m${M}.jsonl}"
        : "${OUTPUT_DIR:=${BASE_DIR}/term_dev_audio_chunks_m${M}}"
        : "${NEG_JSONL:=${BASE_DIR}/term_dev_dataset_all_neg_m${M}.jsonl}"
        : "${FINAL_JSONL:=${BASE_DIR}/term_dev_dataset_final_m${M}.jsonl}"
    else
        : "${POS_JSONL:=${IN_BASE_DIR}/term_dev_dataset_v3.jsonl}"
        : "${OUTPUT_DIR:=${BASE_DIR}/term_dev_audio_chunks}"
        : "${NEG_JSONL:=${BASE_DIR}/term_dev_dataset_all_neg.jsonl}"
        : "${FINAL_JSONL:=${BASE_DIR}/term_dev_dataset_final.jsonl}"
    fi
fi

: "${FULL_POS_JSONL:=${POS_JSONL}}"

mkdir -p "${OUTPUT_DIR}"

# Step 1: 提取所有负样本 (经过全局术语库严格校验)
echo "[INFO] Step 1: Extracting all PURE NO_TERM samples for ${MODE}..."
GEN_NEG_OPTS=""
if [[ -n "${M}" ]]; then
    GEN_NEG_OPTS="--multiplier-merge ${M}"
fi
if [[ "${WRITE_NEG_AUDIO}" == "true" ]]; then
    GEN_NEG_OPTS="${GEN_NEG_OPTS} --write-audio"
fi
python /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/generate_no_term_samples.py \
    --input-tsv "${INPUT_TSV}" \
    --input-jsonl "${POS_JSONL}" \
    --full-term-jsonl "${FULL_POS_JSONL}" \
    --output-dir "${OUTPUT_DIR}" \
    --output-jsonl "${NEG_JSONL}" \
    ${GEN_NEG_OPTS}

# Step 2: 合并正负样本
if [ "$USE_ALL_NEG" = true ]; then
    echo "[INFO] Step 2: Merging ALL negative samples into final dataset..."
    python merge_datasets.py \
        --pos-jsonl "${POS_JSONL}" \
        --neg-jsonl "${NEG_JSONL}" \
        --output-jsonl "${FINAL_JSONL}" \
        --all-neg
else
    echo "[INFO] Step 2: Merging datasets with ratio ${RATIO}..."
    python merge_datasets.py \
        --pos-jsonl "${POS_JSONL}" \
        --neg-jsonl "${NEG_JSONL}" \
        --output-jsonl "${FINAL_JSONL}" \
        --ratio "${RATIO}"
fi

echo "[INFO] Dataset preparation for ${MODE} completed!"
echo "[INFO] Final dataset: ${FINAL_JSONL}"

