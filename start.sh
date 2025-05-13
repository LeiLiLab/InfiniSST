#!/bin/bash

# Make script executable
chmod +x ./backend/cmd/main.go
chmod +x ./model/server.py

echo "Starting InfiniSST Translation System"
echo "===================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker and Docker Compose first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install it first."
    exit 1
fi

# Option to run without Docker (development mode)
if [ "$1" = "--dev" ]; then
    echo "Starting in development mode..."
    
    # Start model service
    echo "Starting Python model service..."
    cd model
    pip install -r requirements.txt
    python server.py &
    MODEL_PID=$!
    cd ..
    
    # Start backend service
    echo "Starting Go backend service..."
    cd backend
    go mod tidy
    go run cmd/main.go &
    BACKEND_PID=$!
    cd ..
    
    # Start frontend
    echo "Starting React frontend..."
    cd infinisst-web
    npm install
    npm start &
    FRONTEND_PID=$!
    cd ..
    
    # Setup trap to kill all processes on exit
    trap "kill $MODEL_PID $BACKEND_PID $FRONTEND_PID; exit" INT TERM EXIT
    
    echo "All services started in development mode."
    echo "Model service: http://localhost:5000"
    echo "Backend API: http://localhost:8080"
    echo "Frontend: http://localhost:3000"
    
    # Wait for interrupt
    wait
else
    # Start with Docker Compose
    echo "Starting with Docker Compose..."
    docker-compose up --build
fi 