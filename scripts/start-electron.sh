#!/bin/bash

# InfiniSST Electron Application Startup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_status "Starting InfiniSST Electron Application..."
print_status "Project root: $PROJECT_ROOT"

# Change to project root
cd "$PROJECT_ROOT"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js first."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    print_error "npm is not installed. Please install npm first."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    print_error "Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Determine Python command
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

print_status "Using Python command: $PYTHON_CMD"

# Check if package.json exists
if [ ! -f "package.json" ]; then
    print_error "package.json not found. Please run this script from the project root."
    exit 1
fi

# Check if node_modules exists, if not install dependencies
if [ ! -d "node_modules" ]; then
    print_status "Installing Node.js dependencies..."
    npm install
    if [ $? -eq 0 ]; then
        print_success "Node.js dependencies installed successfully"
    else
        print_error "Failed to install Node.js dependencies"
        exit 1
    fi
else
    print_status "Node.js dependencies already installed"
fi

# Check if Python dependencies are installed
print_status "Checking Python dependencies..."
cd serve
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found in serve directory"
    exit 1
fi

# Try to import required packages
$PYTHON_CMD -c "import fastapi, uvicorn, soundfile, numpy, torch" 2>/dev/null
if [ $? -ne 0 ]; then
    print_warning "Some Python dependencies are missing. Installing..."
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        print_success "Python dependencies installed successfully"
    else
        print_error "Failed to install Python dependencies"
        exit 1
    fi
else
    print_status "Python dependencies are available"
fi

cd "$PROJECT_ROOT"

# Check for GPU availability
print_status "Checking GPU availability..."
$PYTHON_CMD -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" 2>/dev/null
if [ $? -ne 0 ]; then
    print_warning "Could not check GPU availability. PyTorch may not be properly installed."
fi

# Set environment variables
export NODE_ENV=${NODE_ENV:-development}
export ELECTRON_IS_DEV=${ELECTRON_IS_DEV:-true}

print_status "Environment: $NODE_ENV"
print_status "Electron dev mode: $ELECTRON_IS_DEV"

# Parse command line arguments
MODE="dev"
while [[ $# -gt 0 ]]; do
    case $1 in
        --prod|--production)
            MODE="prod"
            export NODE_ENV="production"
            export ELECTRON_IS_DEV="false"
            shift
            ;;
        --dev|--development)
            MODE="dev"
            export NODE_ENV="development"
            export ELECTRON_IS_DEV="true"
            shift
            ;;
        --build)
            MODE="build"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dev, --development    Run in development mode (default)"
            echo "  --prod, --production    Run in production mode"
            echo "  --build                 Build the application"
            echo "  --help, -h              Show this help message"
            echo ""
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Execute based on mode
case $MODE in
    "dev")
        print_status "Starting application in development mode..."
        npm run dev
        ;;
    "prod")
        print_status "Starting application in production mode..."
        npm run electron
        ;;
    "build")
        print_status "Building application..."
        npm run build
        if [ $? -eq 0 ]; then
            print_success "Application built successfully"
            print_status "Build artifacts are in the 'dist' directory"
        else
            print_error "Build failed"
            exit 1
        fi
        ;;
esac 