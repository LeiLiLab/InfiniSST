#!/bin/bash
set -euo pipefail

# ======Configuration=====
PIPELINE_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/wiki_synth/3variant"
CONDA_PREFIX_PATH="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
# ======Configuration=====

export PATH="${CONDA_PREFIX_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX_PATH}/lib:${LD_LIBRARY_PATH:-}"

echo "============================================================"
echo "[PIPELINE] 3-variant pipeline (top 1M terms × 3 variants)"
echo "[PIPELINE] Directory: ${PIPELINE_DIR}"
echo "============================================================"

# Step 1: Prepare top 1M terms JSON (runs locally, fast)
echo ""
echo "[STEP 1/5] Preparing top 1M terms with short_description..."
python "${PIPELINE_DIR}/prepare_top1m_terms.py"

# Step 2: Submit Gemini utterance generation (taurus, CPU-only)
echo ""
echo "[STEP 2/5] Submitting Gemini 3-variant utterance generation..."
GEMINI_JOB=$(sbatch --parsable "${PIPELINE_DIR}/run_gemini_3variant.sh")
echo "  Gemini job: ${GEMINI_JOB}"

# Step 3: Submit TTS (aries, 8 GPU array, depends on Gemini)
echo ""
echo "[STEP 3/5] Submitting TTS job (depends on ${GEMINI_JOB})..."
TTS_JOB=$(sbatch --parsable --dependency=afterok:${GEMINI_JOB} "${PIPELINE_DIR}/run_tts_3variant_aries.sh")
echo "  TTS job: ${TTS_JOB}"

# Step 4: Submit merge TTS outputs (taurus, CPU-only, depends on TTS)
echo ""
echo "[STEP 4/5] Submitting TTS merge job (depends on ${TTS_JOB})..."
MERGE_JOB=$(sbatch --parsable --dependency=afterok:${TTS_JOB} \
    --job-name=merge_3var \
    --partition=taurus \
    --nodes=1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=1:00:00 \
    --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_merge_3var.out \
    --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_merge_3var.err \
    --wrap="export PATH='${CONDA_PREFIX_PATH}/bin:\${PATH}'; \
            export LD_LIBRARY_PATH='${CONDA_PREFIX_PATH}/lib:\${LD_LIBRARY_PATH:-}'; \
            python ${PIPELINE_DIR}/merge_tts_3variant.py")
echo "  Merge job: ${MERGE_JOB}"

# Step 5: Submit MFA (aries, 20 CPU array, depends on merge)
echo ""
echo "[STEP 5/5] Submitting MFA job (depends on ${MERGE_JOB})..."
MFA_JOB=$(sbatch --parsable --dependency=afterok:${MERGE_JOB} "${PIPELINE_DIR}/run_mfa_3variant_aries.sh")
echo "  MFA job: ${MFA_JOB}"

echo ""
echo "============================================================"
echo "[PIPELINE] All jobs submitted:"
echo "  1. Gemini:  ${GEMINI_JOB}  (taurus, CPU)"
echo "  2. TTS:     ${TTS_JOB}  (aries, 8×GPU)"
echo "  3. Merge:   ${MERGE_JOB}  (taurus, CPU)"
echo "  4. MFA:     ${MFA_JOB}  (aries, 20×CPU)"
echo ""
echo "  Chain: Gemini → TTS → Merge → MFA"
echo "============================================================"
echo ""
echo "Monitor: squeue -u \${USER} -n gemini_3var,tts_3var,merge_3var,mfa_3var"
