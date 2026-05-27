#!/bin/bash
set -euo pipefail

# ======Configuration=====
INFINI_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"
SCRIPT_DIR="${INFINI_ROOT}/documents/code/data_pre/hard_negative_jsonl_for_speech_llm"
TRAIN_SCRIPT_DIR="${INFINI_ROOT}/documents/code/train/sst_omni_train"

CONDA_ENV="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"

CLEANED_JSONL="/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_cleaned.jsonl"
GLOSSARY_JSON="/mnt/gemini/data1/jiaxuanluo/glossary_from_gt_cleaned.json"
MODEL_PATH="/mnt/taurus/data/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs10752_t=0.03_3var_clean_gc_wr1000k_m0.1_maxsim_sp07_best_acl6060_gs10000.pt"

RETRIEVER_OUTPUT="/mnt/gemini/data1/jiaxuanluo/train_cleaned_with_retriever_results_varlen.jsonl"
ABLATION_DIR="/mnt/gemini/data1/jiaxuanluo/density_ablation"
LOG_DIR="/mnt/gemini/data1/jiaxuanluo/logs"

RETRIEVAL_DENSITY=10
DENSITY_VALUES="1 3 5 8 10"
TRAIN_GPUS="0,1,2,3"
RETRIEVER_GPU="0"
RETRIEVER_PARTITION="taurus"
TRAIN_PARTITION="taurus"

DOCKER_IMAGE="modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.1"
# ======Configuration=====

mkdir -p "${ABLATION_DIR}" "${LOG_DIR}"

echo "=============================================="
echo " Full Density Ablation Pipeline"
echo "  Retriever output: ${RETRIEVER_OUTPUT}"
echo "  Ablation dir:     ${ABLATION_DIR}"
echo "  Densities:        ${DENSITY_VALUES}"
echo "=============================================="

# --- Step 1: Submit retriever job ---
RETRIEVER_SBATCH=$(mktemp /tmp/retriever_XXXXXX.sh)
cat > "${RETRIEVER_SBATCH}" << 'RETRIEVER_EOF'
#!/bin/bash
#SBATCH --job-name=gen_termmap_varlen
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00:00
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate __CONDA_ENV__
export PYTHONPATH="__INFINI_ROOT__:${PYTHONPATH:-}"
export HF_HOME="/mnt/taurus/data/jiaxuanluo/cache/huggingface"
export TORCH_HOME="/mnt/taurus/data/jiaxuanluo/cache/torch"
export XDG_CACHE_HOME="/mnt/taurus/data/jiaxuanluo/cache"
export PYTHONUNBUFFERED=1

echo "[STEP1] Retriever inference (variable-length) starting..."
python3 __SCRIPT_DIR__/generate_termmap_maxsim.py \
    --cleaned_jsonl "__CLEANED_JSONL__" \
    --glossary_json "__GLOSSARY_JSON__" \
    --model_path "__MODEL_PATH__" \
    --output_jsonl "__RETRIEVER_OUTPUT__" \
    --device cuda:0 \
    --retrieval_density __RETRIEVAL_DENSITY__
echo "[STEP1] Retriever done."
RETRIEVER_EOF

sed -i "s|__CONDA_ENV__|${CONDA_ENV}|g" "${RETRIEVER_SBATCH}"
sed -i "s|__INFINI_ROOT__|${INFINI_ROOT}|g" "${RETRIEVER_SBATCH}"
sed -i "s|__SCRIPT_DIR__|${SCRIPT_DIR}|g" "${RETRIEVER_SBATCH}"
sed -i "s|__CLEANED_JSONL__|${CLEANED_JSONL}|g" "${RETRIEVER_SBATCH}"
sed -i "s|__GLOSSARY_JSON__|${GLOSSARY_JSON}|g" "${RETRIEVER_SBATCH}"
sed -i "s|__MODEL_PATH__|${MODEL_PATH}|g" "${RETRIEVER_SBATCH}"
sed -i "s|__RETRIEVER_OUTPUT__|${RETRIEVER_OUTPUT}|g" "${RETRIEVER_SBATCH}"
sed -i "s|__RETRIEVAL_DENSITY__|${RETRIEVAL_DENSITY}|g" "${RETRIEVER_SBATCH}"

# Check if retriever output already exists
if [ -f "${RETRIEVER_OUTPUT}" ]; then
    EXPECTED_LINES=$(wc -l < "${CLEANED_JSONL}")
    ACTUAL_LINES=$(wc -l < "${RETRIEVER_OUTPUT}")
    if [ "${ACTUAL_LINES}" -ge "${EXPECTED_LINES}" ]; then
        echo "[SKIP] Retriever output already exists with ${ACTUAL_LINES} lines (expected ${EXPECTED_LINES}). Skipping."
        RETRIEVER_JOB_ID=""
    else
        echo "[WARN] Retriever output exists but incomplete (${ACTUAL_LINES}/${EXPECTED_LINES}). Re-running."
        RETRIEVER_JOB_ID=$(sbatch --parsable -p "${RETRIEVER_PARTITION}" \
            -o "${LOG_DIR}/%j_retriever_varlen.out" \
            -e "${LOG_DIR}/%j_retriever_varlen.err" \
            "${RETRIEVER_SBATCH}")
        echo "[STEP1] Submitted retriever job: ${RETRIEVER_JOB_ID}"
    fi
else
    RETRIEVER_JOB_ID=$(sbatch --parsable -p "${RETRIEVER_PARTITION}" \
        -o "${LOG_DIR}/%j_retriever_varlen.out" \
        -e "${LOG_DIR}/%j_retriever_varlen.err" \
        "${RETRIEVER_SBATCH}")
    echo "[STEP1] Submitted retriever job: ${RETRIEVER_JOB_ID}"
fi

# --- Step 2 + 3: Submit rebuild + training pipeline ---
PIPELINE_SBATCH=$(mktemp /tmp/pipeline_XXXXXX.sh)
cat > "${PIPELINE_SBATCH}" << 'PIPELINE_EOF'
#!/bin/bash
#SBATCH --job-name=density_rebuild_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate __CONDA_ENV__
export PYTHONPATH="__INFINI_ROOT__:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=============================================="
echo "[STEP2] Rebuilding term_maps for all densities"
echo "=============================================="

for d in __DENSITY_VALUES__; do
    OUTPUT="${__ABLATION_DIR__}/train_maxsim_varlen_d${d}.jsonl"
    echo "[STEP2] density_coeff=${d} -> ${OUTPUT}"
    python3 __SCRIPT_DIR__/rebuild_termmap.py \
        --input_jsonl "__RETRIEVER_OUTPUT__" \
        --output_jsonl "${OUTPUT}" \
        --density_coeff "${d}" \
        --seed 42
    echo "[STEP2] density_coeff=${d} done: $(wc -l < "${OUTPUT}") lines"
done

echo ""
echo "=============================================="
echo "[STEP3] Training Speech LLM for each density"
echo "=============================================="

DOCKER_IMAGE="__DOCKER_IMAGE__"
TRAIN_GPUS="__TRAIN_GPUS__"

for d in __DENSITY_VALUES__; do
    DATASET="${__ABLATION_DIR__}/train_maxsim_varlen_d${d}.jsonl"
    if [ ! -f "${DATASET}" ]; then
        echo "[ERROR] Missing dataset for d=${d}: ${DATASET}"
        continue
    fi

    echo ""
    echo "=== Training d=${d} ==="
    echo "  Dataset: ${DATASET}"
    echo "  GPUs: ${TRAIN_GPUS}"

    docker run --rm \
        --gpus all \
        --shm-size=32g \
        --ipc=host \
        -e CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
        -e NCCL_P2P_DISABLE=1 \
        -e NCCL_IB_DISABLE=1 \
        -v /home/jiaxuanluo/InfiniSST:/workspace/InfiniSST \
        -v /mnt/taurus/home/jiaxuanluo/InfiniSST:/workspace/InfiniSST_taurus \
        -v /mnt/gemini/data:/mnt/gemini/data \
        -v /mnt/gemini/data1:/mnt/gemini/data1 \
        -v /mnt/gemini/data2:/mnt/gemini/data2 \
        -v /mnt/aries/data4:/mnt/aries/data4 \
        "${DOCKER_IMAGE}" \
        bash /workspace/InfiniSST/documents/code/train/sst_omni_train/run_speech_llm_4gpu_maxsim.sh "${d}"

    echo "=== Training d=${d} finished ==="
done

echo ""
echo "=============================================="
echo " All density ablation training complete!"
echo "=============================================="
PIPELINE_EOF

sed -i "s|__CONDA_ENV__|${CONDA_ENV}|g" "${PIPELINE_SBATCH}"
sed -i "s|__INFINI_ROOT__|${INFINI_ROOT}|g" "${PIPELINE_SBATCH}"
sed -i "s|__SCRIPT_DIR__|${SCRIPT_DIR}|g" "${PIPELINE_SBATCH}"
sed -i "s|__RETRIEVER_OUTPUT__|${RETRIEVER_OUTPUT}|g" "${PIPELINE_SBATCH}"
sed -i "s|__ABLATION_DIR__|${ABLATION_DIR}|g" "${PIPELINE_SBATCH}"
sed -i "s|__DENSITY_VALUES__|${DENSITY_VALUES}|g" "${PIPELINE_SBATCH}"
sed -i "s|__DOCKER_IMAGE__|${DOCKER_IMAGE}|g" "${PIPELINE_SBATCH}"
sed -i "s|__TRAIN_GPUS__|${TRAIN_GPUS}|g" "${PIPELINE_SBATCH}"

if [ -n "${RETRIEVER_JOB_ID:-}" ]; then
    PIPELINE_JOB_ID=$(sbatch --parsable -p "${TRAIN_PARTITION}" \
        --dependency=afterok:${RETRIEVER_JOB_ID} \
        -o "${LOG_DIR}/%j_density_pipeline.out" \
        -e "${LOG_DIR}/%j_density_pipeline.err" \
        "${PIPELINE_SBATCH}")
    echo "[STEP2+3] Submitted rebuild+train pipeline: ${PIPELINE_JOB_ID} (after retriever ${RETRIEVER_JOB_ID})"
else
    PIPELINE_JOB_ID=$(sbatch --parsable -p "${TRAIN_PARTITION}" \
        -o "${LOG_DIR}/%j_density_pipeline.out" \
        -e "${LOG_DIR}/%j_density_pipeline.err" \
        "${PIPELINE_SBATCH}")
    echo "[STEP2+3] Submitted rebuild+train pipeline: ${PIPELINE_JOB_ID} (no dependency, retriever already done)"
fi

echo ""
echo "=============================================="
echo " Pipeline submitted!"
echo "  Monitor: squeue -u \$(whoami)"
echo "  Logs:    ${LOG_DIR}/"
echo "=============================================="
