# InfiniSST Go Backend

This is the Go backend for the InfiniSST project, which handles API requests from the React frontend and manages communication with the Python worker processes that run the actual translation models.

## Architecture

The backend has four main components:

1. HTTP API Server - Handles API requests from the frontend
2. WebSocket Server - Manages real-time audio streaming and translation
3. Session Manager - Tracks active translation sessions
4. Python Worker Processes - Run the translation models (one per session)

## Setup

### Prerequisites

- Go 1.21 or later
- Python 3.8 or later (with the InfiniSST model dependencies installed)

### Installation

1. Install Go dependencies:

```bash
cd backend
go mod download
```

2. Make sure the Python worker script is executable:

```bash
chmod +x ../infinisst-model/serve/worker.py
```

## Running the Backend

```bash
cd backend
go run .
```

The server will start on port 8080.

## API Endpoints

- `POST /api/init` - Initialize a new translation session
- `WebSocket /api/ws/{session_id}` - Stream audio and receive translations
- `POST /api/update_latency` - Update the latency multiplier
- `POST /api/reset_translation` - Reset the translation state
- `POST /api/delete_session` - Clean up a session
- `POST /api/ping` - Keep a session alive

## Building for Production

To build the backend for production:

```bash
cd backend
go build -o infinisst-backend
```

Then run the backend:

```bash
./infinisst-backend
``` 