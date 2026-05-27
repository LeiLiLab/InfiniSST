#!/bin/bash
#SBATCH --job-name=repair_gs_f031
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_repair_gs_f031.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_repair_gs_f031.err
#SBATCH --chdir=/tmp

# Repair only missing GigaSpeech MFA spans in the full GSV2 train JSONL.
# Wiki/GSV2 rows already carry spans from align_and_cut_wiki_synth.py, so they
# are preserved unchanged. Empty-term GigaSpeech no-term chunks and rare
# non-empty rows that still cannot be matched are dropped.

set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"
SCRIPT_PATH="${REPO_ROOT}/documents/code/data_pre/enrich_jsonl_with_mfa_timestamps.py"

INPUT_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa.jsonl"
OUTPUT_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired.jsonl"
SQLITE_INDEX="/mnt/gemini/data1/jiaxuanluo/gigaspeech_mfa_index/gigaspeech_mfa_index.sqlite"
GS_TEXTGRID_DIR="/mnt/taurus/data/siqiouyang/datasets/gigaspeech/textgrids"

echo "================================================================"
echo "[REPAIR][GS-FULL0_31] Input    = ${INPUT_JSONL}"
echo "[REPAIR][GS-FULL0_31] Output   = ${OUTPUT_JSONL}"
echo "[REPAIR][GS-FULL0_31] SQLite   = ${SQLITE_INDEX}"
echo "[REPAIR][GS-FULL0_31] TextGrid = ${GS_TEXTGRID_DIR}"
echo "[REPAIR][GS-FULL0_31] Start: $(date)"
echo "================================================================"

python "${SCRIPT_PATH}" \
    --input "${INPUT_JSONL}" \
    --output "${OUTPUT_JSONL}" \
    --sqlite_index "${SQLITE_INDEX}" \
    --gs_textgrid_dir "${GS_TEXTGRID_DIR}" \
    --only-source gs \
    --only-missing \
    --preserve-existing \
    --drop-empty-term \
    --drop-unmatched

echo "[REPAIR][GS-FULL0_31] Completed at $(date)"
