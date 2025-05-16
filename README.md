# InfiniSST Project

InfiniSST is a real-time speech-to-text translation system that processes audio input and provides translations with minimal latency.

## Project Structure

The project is organized into three main components:

1. **infinisst-web/** - React frontend application
   - User interface for uploading or recording audio
   - Real-time display of translations
   - WebSocket communication with the backend

2. **infinisst-model/serve/** - Python FastAPI backend server
   - API endpoints for managing translation sessions
   - WebSocket server for streaming audio and receiving translations
   - Serves as the backend for the React frontend

3. **infinisst-model/** - Python translation model
   - Neural network model for speech-to-text translation
   - Audio processing and transcription
   - Integrated with the FastAPI server

## Getting Started

### Prerequisites

- Node.js and npm for the frontend
- Python 3.8+ with PyTorch and FastAPI for the backend and model

### Running the Frontend

```bash
cd infinisst-web
npm install
npm start
```

This will start the React frontend on http://localhost:3000.

### Running the Backend

```bash
cd infinisst-model/serve
pip install -r requirements.txt
python api.py
```

This will start the FastAPI server on http://localhost:8000.

### Development Notes

The React frontend communicates with the FastAPI backend through HTTP endpoints and WebSockets. The model is integrated with the FastAPI server.

## API Endpoints

- **POST /init** - Initialize a new translation session
- **WebSocket /wss/{session_id}** - Stream audio and receive translations
- **POST /update_latency** - Update the latency multiplier
- **POST /reset_translation** - Reset the translation state
- **POST /delete_session** - Clean up a session
- **POST /ping** - Keep a session alive
- **GET /queue_status/{session_id}** - Check the status of a session
- **GET /health** - Health check endpoint

## License

[MIT License](LICENSE)