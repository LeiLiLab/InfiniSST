#!/bin/bash

# InfiniSST Electron Client Startup Script

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

print_status "Starting InfiniSST Electron Client..."
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

# Set environment variables
export NODE_ENV=${NODE_ENV:-development}
export ELECTRON_IS_DEV=${ELECTRON_IS_DEV:-true}

# Parse command line arguments
MODE="dev"
SERVER_HOST=""
SERVER_PORT=""
SERVER_PROTOCOL=""

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
        --server)
            SERVER_HOST="$2"
            shift 2
            ;;
        --port)
            SERVER_PORT="$2"
            shift 2
            ;;
        --protocol)
            SERVER_PROTOCOL="$2"
            shift 2
            ;;
        --ngrok)
            if [ ! -z "$2" ] && [[ ! "$2" =~ ^-- ]]; then
                # Custom ngrok URL provided
                NGROK_URL="$2"
                shift 2
            else
                # Default ngrok URL
                NGROK_URL="infinisst.ngrok.app"
                shift
            fi
            SERVER_HOST="$NGROK_URL"
            SERVER_PORT="443"
            SERVER_PROTOCOL="https"
            print_status "Using ngrok tunnel: https://$NGROK_URL"
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dev, --development    Run in development mode (default)"
            echo "  --prod, --production    Run in production mode"
            echo "  --build                 Build the application"
            echo "  --server HOST           Set remote server host"
            echo "  --port PORT             Set remote server port"
            echo "  --protocol PROTOCOL     Set protocol (http/https)"
            echo "  --ngrok [URL]           Connect to ngrok tunnel (default: infinisst.ngrok.app)"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --dev                                    # Development mode"
            echo "  $0 --server your-server.com --port 8001    # Connect to remote server"
            echo "  $0 --ngrok                                  # Connect to default ngrok tunnel"
            echo "  $0 --ngrok your-app.ngrok.app              # Connect to custom ngrok tunnel"
            echo "  $0 --build                                  # Build application"
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

# Set environment variables for server configuration
if [ ! -z "$SERVER_HOST" ]; then
    export INFINISST_HOST="$SERVER_HOST"
    print_status "Server host set to: $SERVER_HOST"
fi

if [ ! -z "$SERVER_PORT" ]; then
    export INFINISST_PORT="$SERVER_PORT"
    print_status "Server port set to: $SERVER_PORT"
fi

if [ ! -z "$SERVER_PROTOCOL" ]; then
    export INFINISST_PROTOCOL="$SERVER_PROTOCOL"
    print_status "Server protocol set to: $SERVER_PROTOCOL"
fi

print_status "Environment: $NODE_ENV"
print_status "Electron dev mode: $ELECTRON_IS_DEV"

# Execute based on mode
case $MODE in
    "dev")
        print_status "Starting Electron client in development mode..."
        npm run electron-dev
        ;;
    "prod")
        print_status "Starting Electron client in production mode..."
        npm run electron
        ;;
    "build")
        print_status "Building Electron application..."
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