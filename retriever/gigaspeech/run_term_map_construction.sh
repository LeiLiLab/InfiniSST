#!/bin/bash
#SBATCH --job-name=term_map_construction
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%A_%a_term_map_construction.out
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%A_%a_term_map_construction.err
#SBATCH --partition=taurus
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#
# Convenience script to run term map dataset construction
#
# Usage:
#   ./run_term_map_construction.sh                # Full processing
#   ./run_term_map_construction.sh --dry-run      # Test with 10 messages
#   ./run_term_map_construction.sh --max 100      # Process 100 messages
#
source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../.."

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Term Map Dataset Construction${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# Check conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${YELLOW}Warning: No conda environment activated${NC}"
    echo "Activating 'infinisst' environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate infinisst
fi

echo -e "Current environment: ${GREEN}$CONDA_DEFAULT_ENV${NC}"
echo ""

# GPU mapping: Skip GPU 2 (in use by other processes)
# Map array task IDs to physical GPU IDs
# Task 0 → GPU 0, Task 1 → GPU 1, Task 2 → GPU 3, Task 3 → GPU 4
GPU_MAP=(0 1 3 4)

# Get GPU ID from SLURM array task ID
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    # Running locally (not in SLURM), default to GPU 0
    PHYSICAL_GPU_ID=0
    GPU_ID=0
    TOTAL_GPUS=1
    echo -e "${YELLOW}Not running in SLURM array, using single GPU${NC}"
else
    # Get the array task ID (0, 1, 2, 3)
    ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID
    TOTAL_GPUS=4
    
    # Map to physical GPU ID using GPU_MAP
    PHYSICAL_GPU_ID=${GPU_MAP[$ARRAY_TASK_ID]}
    
    # For data sharding, still use array task ID (0-3)
    GPU_ID=$ARRAY_TASK_ID
    
    echo -e "${GREEN}SLURM Array Task ID: $ARRAY_TASK_ID${NC}"
    echo -e "${GREEN}Mapping to Physical GPU: $PHYSICAL_GPU_ID${NC}"
    echo -e "${GREEN}Data shard: $GPU_ID/$TOTAL_GPUS${NC}"
    
    # CRITICAL: Set CUDA_VISIBLE_DEVICES to only the mapped physical GPU
    # This MUST be set BEFORE any Python/PyTorch code runs
    export CUDA_VISIBLE_DEVICES=$PHYSICAL_GPU_ID
    
    echo -e "${GREEN}CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES${NC}"
    echo ""
    
    # Show which GPU we're using
    echo -e "${YELLOW}GPU Assignment:${NC}"
    echo -e "  Task $ARRAY_TASK_ID → Physical GPU $PHYSICAL_GPU_ID"
    echo -e "  (Skipping GPU 2 - in use by other processes)"
fi

# Parse arguments
ARGS="--gpu-id $GPU_ID --total-gpus $TOTAL_GPUS"
if [ "$1" = "--dry-run" ]; then
    ARGS="$ARGS --dry-run"
    echo -e "${YELLOW}Running in DRY RUN mode (10 messages only)${NC}"
elif [ "$1" = "--max" ] && [ -n "$2" ]; then
    ARGS="$ARGS --max-messages $2"
    echo -e "${YELLOW}Processing max $2 messages${NC}"
else
    echo -e "${GREEN}Running full processing on GPU $GPU_ID${NC}"
fi
echo ""

# Run the script
cd "$PROJECT_ROOT"
echo "Working directory: $(pwd)"
echo ""

echo -e "${GREEN}Starting processing...${NC}"
python /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/handle_train_dataset_for_term_map_v2_buzz.py $ARGS

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Processing complete!${NC}"
echo -e "${GREEN}================================${NC}"

# Check output file
OUTPUT_FILE="/mnt/gemini/data1/jiaxuanluo/train_s_zh_with_candidates_gpu${GPU_ID}.jsonl"
if [ -f "$OUTPUT_FILE" ]; then
    LINE_COUNT=$(wc -l < "$OUTPUT_FILE")
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo ""
    echo -e "Output file: ${GREEN}$OUTPUT_FILE${NC}"
    echo -e "Lines: ${GREEN}$LINE_COUNT${NC}"
    echo -e "Size: ${GREEN}$FILE_SIZE${NC}"
    echo -e "Physical GPU used: ${GREEN}$PHYSICAL_GPU_ID${NC}"
    echo ""
    echo "Sample output (first message):"
    head -n 1 "$OUTPUT_FILE" | jq '.' 2>/dev/null || head -n 1 "$OUTPUT_FILE"
fi

# If this is the last GPU (task 3), merge all outputs
if [ "$ARRAY_TASK_ID" = "3" ] && [ -z "$1" ]; then
    echo ""
    echo -e "${YELLOW}Waiting for all GPUs to finish...${NC}"
    sleep 10
    
    FINAL_OUTPUT="/mnt/gemini/data1/jiaxuanluo/train_s_zh_with_candidates.jsonl"
    echo -e "${GREEN}Merging outputs from all GPUs...${NC}"
    
    # Merge all GPU outputs
    cat /mnt/gemini/data1/jiaxuanluo/train_s_zh_with_candidates_gpu*.jsonl > "$FINAL_OUTPUT"
    
    TOTAL_LINES=$(wc -l < "$FINAL_OUTPUT")
    TOTAL_SIZE=$(du -h "$FINAL_OUTPUT" | cut -f1)
    
    echo ""
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}Final merged output:${NC}"
    echo -e "${GREEN}================================${NC}"
    echo -e "File: ${GREEN}$FINAL_OUTPUT${NC}"
    echo -e "Total lines: ${GREEN}$TOTAL_LINES${NC}"
    echo -e "Total size: ${GREEN}$TOTAL_SIZE${NC}"
    
    # Optionally remove individual GPU files
    # echo -e "${YELLOW}Cleaning up individual GPU files...${NC}"
    # rm /mnt/gemini/data1/jiaxuanluo/train_s_zh_with_candidates_gpu*.jsonl
fi

