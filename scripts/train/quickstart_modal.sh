#!/usr/bin/env bash
# Quick start script for Modal training
# This script provides a simple interface to common Modal operations

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODAL_SCRIPT="${SCRIPT_DIR}/modal_stage1_gigaspeech_rag.py"

# Print banner
print_banner() {
    echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║   InfiniSST Modal Training Launcher   ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
    echo ""
}

# Print help
print_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup           Setup Modal volumes and upload data"
    echo "  check           Check volume structure and data"
    echo "  train           Start training with default settings"
    echo "  train-resume    Resume training from last checkpoint"
    echo "  train-custom    Start training with custom parameters"
    echo "  monitor         Open Modal dashboard to monitor training"
    echo "  download        Download trained model from Modal"
    echo "  clean           Clean up old training runs"
    echo "  help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup        # First-time setup"
    echo "  $0 check        # Verify data is uploaded"
    echo "  $0 train        # Start training"
    echo ""
}

# Check prerequisites
check_prerequisites() {
    if ! command -v modal &> /dev/null; then
        echo -e "${RED}✗ Error: modal CLI not found${NC}"
        echo "Install it with: pip install modal"
        exit 1
    fi
    
    if ! modal token show &> /dev/null; then
        echo -e "${RED}✗ Error: Not logged in to Modal${NC}"
        echo "Run: modal token new"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Prerequisites check passed${NC}"
    echo ""
}

# Setup volumes
setup_volumes() {
    echo -e "${YELLOW}Setting up Modal volumes...${NC}"
    echo ""
    
    if [ -f "${SCRIPT_DIR}/setup_modal_volumes.sh" ]; then
        bash "${SCRIPT_DIR}/setup_modal_volumes.sh"
    else
        echo -e "${RED}✗ Setup script not found${NC}"
        echo "Creating volumes manually..."
        modal volume create infinisst-data || true
        modal volume create infinisst-models || true
        modal volume create infinisst-outputs || true
    fi
}

# Check volumes
check_volumes() {
    echo -e "${YELLOW}Checking Modal volumes...${NC}"
    echo ""
    
    cd "${SCRIPT_DIR}"
    modal run modal_stage1_gigaspeech_rag.py --check-volume
}

# Start training
start_training() {
    local resume=$1
    
    echo -e "${YELLOW}Starting training on Modal...${NC}"
    echo ""
    
    cd "${SCRIPT_DIR}"
    
    if [ "$resume" = "true" ]; then
        echo -e "${BLUE}→ Resuming from last checkpoint${NC}"
        modal run modal_stage1_gigaspeech_rag.py --resume
    else
        echo -e "${BLUE}→ Starting fresh training${NC}"
        modal run modal_stage1_gigaspeech_rag.py
    fi
}

# Custom training
custom_training() {
    echo -e "${YELLOW}Custom Training Configuration${NC}"
    echo ""
    
    # Interactive parameter collection
    read -p "Run name [stage1_custom]: " RUN_NAME
    RUN_NAME=${RUN_NAME:-stage1_custom}
    
    read -p "Number of GPUs [4]: " NUM_GPUS
    NUM_GPUS=${NUM_GPUS:-4}
    
    read -p "Learning rate [2e-4]: " LR
    LR=${LR:-2e-4}
    
    read -p "Max epochs [1]: " EPOCHS
    EPOCHS=${EPOCHS:-1}
    
    echo ""
    echo -e "${BLUE}Configuration:${NC}"
    echo "  Run name: $RUN_NAME"
    echo "  GPUs: $NUM_GPUS"
    echo "  Learning rate: $LR"
    echo "  Epochs: $EPOCHS"
    echo ""
    
    read -p "Start training with these settings? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "${SCRIPT_DIR}"
        
        # Note: This requires modifying the modal script to accept CLI args
        # For now, users should edit the script directly
        echo -e "${YELLOW}Note: To customize parameters, edit modal_stage1_gigaspeech_rag.py${NC}"
        echo "Modify the parameters in the main() function's train_infinisst.remote() call"
        echo ""
        
        modal run modal_stage1_gigaspeech_rag.py
    fi
}

# Monitor training
monitor_training() {
    echo -e "${YELLOW}Opening Modal dashboard...${NC}"
    
    # Open Modal dashboard in browser
    if command -v xdg-open &> /dev/null; then
        xdg-open "https://modal.com/apps"
    elif command -v open &> /dev/null; then
        open "https://modal.com/apps"
    else
        echo "Please visit: https://modal.com/apps"
    fi
}

# Download results
download_results() {
    echo -e "${YELLOW}Downloading trained model...${NC}"
    echo ""
    
    # List available runs
    echo "Available runs in output volume:"
    modal volume ls infinisst-outputs en-zh/runs/ 2>/dev/null || {
        echo -e "${RED}✗ No training runs found${NC}"
        exit 1
    }
    
    echo ""
    read -p "Enter run name to download: " RUN_NAME
    
    if [ -z "$RUN_NAME" ]; then
        echo -e "${RED}✗ Run name cannot be empty${NC}"
        exit 1
    fi
    
    # Create output directory
    OUTPUT_DIR="./modal_outputs/${RUN_NAME}"
    mkdir -p "${OUTPUT_DIR}"
    
    echo ""
    echo -e "${YELLOW}Downloading to: ${OUTPUT_DIR}${NC}"
    
    # Download using modal volume get
    modal volume get infinisst-outputs \
        "en-zh/runs/${RUN_NAME}/" \
        "${OUTPUT_DIR}/"
    
    echo ""
    echo -e "${GREEN}✓ Download completed${NC}"
    echo "Files saved to: ${OUTPUT_DIR}"
}

# Clean up old runs
clean_runs() {
    echo -e "${YELLOW}Cleaning up old training runs...${NC}"
    echo ""
    
    echo "This will list runs in the output volume."
    echo "You can then manually delete them using:"
    echo "  modal volume rm infinisst-outputs <path>"
    echo ""
    
    modal volume ls infinisst-outputs en-zh/runs/
    
    echo ""
    echo -e "${BLUE}To delete a run:${NC}"
    echo "  modal volume rm infinisst-outputs en-zh/runs/<run_name>/"
}

# Main menu
main_menu() {
    print_banner
    
    echo "Select an option:"
    echo ""
    echo "  1) Setup volumes and upload data"
    echo "  2) Check volume structure"
    echo "  3) Start training"
    echo "  4) Resume training"
    echo "  5) Custom training"
    echo "  6) Monitor training"
    echo "  7) Download results"
    echo "  8) Clean up old runs"
    echo "  9) Exit"
    echo ""
    
    read -p "Enter choice [1-9]: " choice
    echo ""
    
    case $choice in
        1) setup_volumes ;;
        2) check_volumes ;;
        3) start_training false ;;
        4) start_training true ;;
        5) custom_training ;;
        6) monitor_training ;;
        7) download_results ;;
        8) clean_runs ;;
        9) exit 0 ;;
        *) echo -e "${RED}Invalid choice${NC}" ;;
    esac
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    # No arguments - show interactive menu
    check_prerequisites
    main_menu
else
    case $1 in
        setup)
            check_prerequisites
            setup_volumes
            ;;
        check)
            check_prerequisites
            check_volumes
            ;;
        train)
            check_prerequisites
            start_training false
            ;;
        train-resume)
            check_prerequisites
            start_training true
            ;;
        train-custom)
            check_prerequisites
            custom_training
            ;;
        monitor)
            monitor_training
            ;;
        download)
            download_results
            ;;
        clean)
            clean_runs
            ;;
        help|--help|-h)
            print_banner
            print_help
            ;;
        *)
            echo -e "${RED}Unknown command: $1${NC}"
            echo ""
            print_help
            exit 1
            ;;
    esac
fi

