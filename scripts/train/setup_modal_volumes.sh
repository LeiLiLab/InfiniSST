#!/usr/bin/env bash
# Setup script for Modal Volumes - InfiniSST Training
# This script helps you create and populate Modal volumes for training

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}===================================${NC}"
echo -e "${GREEN}Modal Volume Setup for InfiniSST${NC}"
echo -e "${GREEN}===================================${NC}"
echo ""

# Check if modal is installed
if ! command -v modal &> /dev/null; then
    echo -e "${RED}Error: modal CLI not found${NC}"
    echo "Please install it with: pip install modal"
    exit 1
fi

# Check if user is logged in
if ! modal token show &> /dev/null; then
    echo -e "${YELLOW}Warning: Not logged in to Modal${NC}"
    echo "Please run: modal token new"
    exit 1
fi

echo -e "${GREEN}✓ Modal CLI is installed and configured${NC}"
echo ""

# Volume names
DATA_VOLUME="infinisst-data"
MODEL_VOLUME="infinisst-models"
OUTPUT_VOLUME="infinisst-outputs"

# Local paths (modify these according to your setup)
GIGASPEECH_PATH="/mnt/data/siqiouyang/datasets/gigaspeech"
QWEN_MODEL_PATH="/mnt/aries/data6/jiaxuanluo/Qwen2.5-7B-Instruct"
W2V2_MODEL_PATH="/mnt/aries/data6/xixu/demo/wav2_vec_vox_960h_pl.pt"

# Function to create volume if it doesn't exist
create_volume_if_missing() {
    local volume_name=$1
    echo -e "${YELLOW}Checking volume: ${volume_name}${NC}"
    
    if modal volume list | grep -q "│ ${volume_name} "; then
        echo -e "${GREEN}✓ Volume ${volume_name} already exists${NC}"
    else
        echo -e "${YELLOW}Creating volume: ${volume_name}${NC}"
        modal volume create "${volume_name}"
        echo -e "${GREEN}✓ Volume ${volume_name} created${NC}"
    fi
    echo ""
}

# Function to check if path exists in volume
check_volume_path() {
    local volume_name=$1
    local path=$2
    
    if modal volume ls "${volume_name}" "${path}" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to upload data to volume
upload_to_volume() {
    local volume_name=$1
    local local_path=$2
    local remote_path=$3
    local description=$4
    
    echo -e "${YELLOW}Uploading ${description}...${NC}"
    echo "  Source: ${local_path}"
    echo "  Destination: ${volume_name}:${remote_path}"
    
    # Check if local path exists
    if [ ! -e "${local_path}" ]; then
        echo -e "${RED}✗ Error: Local path does not exist: ${local_path}${NC}"
        return 1
    fi
    
    # Check if already uploaded
    if check_volume_path "${volume_name}" "${remote_path}"; then
        echo -e "${YELLOW}⚠ Remote path already exists${NC}"
        read -p "Do you want to overwrite? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}Skipping upload${NC}"
            echo ""
            return 0
        fi
    fi
    
    # Upload
    echo -e "${GREEN}Starting upload (this may take a while)...${NC}"
    
    # Determine if it's a file or directory
    if [ -d "${local_path}" ]; then
        # Directory upload
        modal volume put "${volume_name}" "${local_path}/" "${remote_path}/"
    else
        # File upload
        modal volume put "${volume_name}" "${local_path}" "${remote_path}"
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Upload completed: ${description}${NC}"
    else
        echo -e "${RED}✗ Upload failed: ${description}${NC}"
        return 1
    fi
    echo ""
}

# Main setup process
main() {
    echo -e "${GREEN}Step 1: Creating Modal Volumes${NC}"
    echo "----------------------------------------"
    create_volume_if_missing "${DATA_VOLUME}"
    create_volume_if_missing "${MODEL_VOLUME}"
    create_volume_if_missing "${OUTPUT_VOLUME}"
    
    echo -e "${GREEN}Step 2: Uploading Data${NC}"
    echo "----------------------------------------"
    echo ""
    
    # Upload GigaSpeech dataset
    read -p "Upload GigaSpeech dataset? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -d "${GIGASPEECH_PATH}" ]; then
            upload_to_volume "${DATA_VOLUME}" "${GIGASPEECH_PATH}" "gigaspeech" "GigaSpeech Dataset"
        else
            echo -e "${YELLOW}⚠ GigaSpeech path not found: ${GIGASPEECH_PATH}${NC}"
            echo "Please update GIGASPEECH_PATH in this script"
            echo ""
        fi
    fi
    
    echo -e "${GREEN}Step 3: Uploading Models${NC}"
    echo "----------------------------------------"
    echo ""
    
    # Upload Qwen model
    read -p "Upload Qwen2.5-7B-Instruct model? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -d "${QWEN_MODEL_PATH}" ]; then
            upload_to_volume "${MODEL_VOLUME}" "${QWEN_MODEL_PATH}" "Qwen2.5-7B-Instruct" "Qwen2.5-7B-Instruct Model"
        else
            echo -e "${YELLOW}⚠ Qwen model path not found: ${QWEN_MODEL_PATH}${NC}"
            echo "Please update QWEN_MODEL_PATH in this script"
            echo ""
        fi
    fi
    
    # Upload Wav2Vec2 model
    read -p "Upload Wav2Vec2 model? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -f "${W2V2_MODEL_PATH}" ]; then
            upload_to_volume "${MODEL_VOLUME}" "${W2V2_MODEL_PATH}" "wav2_vec_vox_960h_pl.pt" "Wav2Vec2 Model"
        else
            echo -e "${YELLOW}⚠ Wav2Vec2 model path not found: ${W2V2_MODEL_PATH}${NC}"
            echo "Please update W2V2_MODEL_PATH in this script"
            echo ""
        fi
    fi
    
    echo -e "${GREEN}Step 4: Verification${NC}"
    echo "----------------------------------------"
    echo ""
    
    echo "Listing volume contents..."
    echo ""
    
    echo -e "${YELLOW}Data Volume (${DATA_VOLUME}):${NC}"
    modal volume ls "${DATA_VOLUME}" | head -20
    echo ""
    
    echo -e "${YELLOW}Model Volume (${MODEL_VOLUME}):${NC}"
    modal volume ls "${MODEL_VOLUME}" | head -20
    echo ""
    
    echo -e "${GREEN}===================================${NC}"
    echo -e "${GREEN}Setup Complete!${NC}"
    echo -e "${GREEN}===================================${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Verify the uploaded data with:"
    echo "   modal volume ls ${DATA_VOLUME}"
    echo "   modal volume ls ${MODEL_VOLUME}"
    echo ""
    echo "2. Check volume structure with:"
    echo "   cd $(dirname $0)"
    echo "   modal run modal_stage1_gigaspeech_rag.py --check-volume"
    echo ""
    echo "3. Start training with:"
    echo "   modal run modal_stage1_gigaspeech_rag.py"
    echo ""
}

# Parse command line arguments
QUICK_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick     Skip upload prompts and only create volumes"
            echo "  --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

if [ "$QUICK_MODE" = true ]; then
    echo "Running in quick mode (volumes only)"
    create_volume_if_missing "${DATA_VOLUME}"
    create_volume_if_missing "${MODEL_VOLUME}"
    create_volume_if_missing "${OUTPUT_VOLUME}"
    exit 0
fi

# Run main setup
main

