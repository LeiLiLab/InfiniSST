from fastapi import FastAPI, WebSocket, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import soundfile as sf
import numpy as np
import json
import asyncio
import argparse
import copy
import time
from typing import Dict, Optional
from eval.agents.streamllama import StreamLlama
from eval.agents.tt_alignatt_sllama_stream_att_fw import AlignAttStreamAttFW
import io
import uvicorn
import gc
import torch

# 支持的翻译模型列表
TRANSLATION_AGENTS = {
    "InfiniSST": StreamLlama,
    # 暂时禁用StreamAtt
    # "StreamAtt": AlignAttStreamAttFW,
}

# 支持的语言方向
LANGUAGE_PAIRS = {
    "English -> Chinese": ("English", "Chinese", "en", "zh"),
    "English -> German": ("English", "German", "en", "de"),
    "English -> Spanish": ("English", "Spanish", "en", "es"),
}

model_path = "/compute/babel-5-23/siqiouya/runs/{}-{}/8B-traj-s2-v3.6/last.ckpt/pytorch_model.bin"

app = FastAPI()

# Store active translation sessions with last activity timestamp
active_sessions: Dict[str, dict] = {}
session_last_activity: Dict[str, float] = {}

# Idle timeout in seconds (5 minutes)
IDLE_TIMEOUT = 5 * 60

# Short timeout for detecting browser disconnections
DISCONNECT_CHECK_INTERVAL = 5  # Check every 5 seconds
DISCONNECT_TIMEOUT = 10  # Consider a session orphaned if no activity for 10 seconds

class TranslationSession:
    def __init__(self, agent_type: str, language_pair: str, args):
        self.agent_type = agent_type
        self.language_pair = language_pair
        source_lang, target_lang, src_code, tgt_code = LANGUAGE_PAIRS[language_pair]
        
        args.source_lang = source_lang
        args.target_lang = target_lang
        args.state_dict_path = model_path.format(src_code, tgt_code)
        
        self.agent = TRANSLATION_AGENTS[agent_type](args)
        self.agent.update_multiplier(args.latency_multiplier)
        self.states = self.agent.build_states()
        self.states.reset()
        self.args = args  # Store args in the session
        
    async def process_segment(self, segment: np.ndarray, is_last: bool = False) -> str:
        self.states.source.extend(segment)
        print(len(self.states.source) / 16000)
        if is_last:
            self.states.source_finished = True
            
        action = self.agent.policy(self.states)
        if not action.is_read():
            output = action.content
            self.states.target.append(output)
            return ' '.join(self.states.target) if self.args.target_lang != 'Chinese' else ''.join(self.states.target)
        return ""
    
    def cleanup(self):
        """Clean up GPU resources used by this session"""
        # Clear agent's GPU memory
        if hasattr(self.agent, 'model'):
            if hasattr(self.agent.model, 'to'):
                self.agent.model.to('cpu')  # Move model to CPU first
            
            # Delete model attributes that might hold GPU tensors
            for attr_name in dir(self.agent.model):
                if not attr_name.startswith('__'):
                    attr = getattr(self.agent.model, attr_name)
                    if isinstance(attr, torch.Tensor) and attr.is_cuda:
                        delattr(self.agent.model, attr_name)
        
        # Clear states
        if hasattr(self.states, 'clear'):
            self.states.clear()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"Cleaned up GPU resources for session with {self.agent_type} model")

def update_session_activity(session_id: str):
    """Update the last activity timestamp for a session"""
    if session_id in active_sessions:
        session_last_activity[session_id] = time.time()

async def cleanup_idle_sessions():
    """Background task to clean up idle sessions"""
    while True:
        current_time = time.time()
        sessions_to_remove = []
        
        for session_id, last_activity in session_last_activity.items():
            if current_time - last_activity > IDLE_TIMEOUT:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            if session_id in active_sessions:
                print(f"Cleaning up idle session {session_id} after {IDLE_TIMEOUT} seconds of inactivity")
                session = active_sessions[session_id]
                session.cleanup()
                del active_sessions[session_id]
                del session_last_activity[session_id]
        
        # Check every 60 seconds
        await asyncio.sleep(60)

async def check_orphaned_sessions():
    """Background task to check for orphaned sessions every 5 seconds"""
    while True:
        current_time = time.time()
        sessions_to_check = []
        
        # First, identify potentially orphaned sessions (inactive for 10 seconds)
        for session_id, last_activity in session_last_activity.items():
            inactive_time = current_time - last_activity
            if inactive_time > DISCONNECT_TIMEOUT:
                sessions_to_check.append((session_id, inactive_time))
        
        if sessions_to_check:
            print(f"Checking {len(sessions_to_check)} potentially orphaned sessions")
            
        for session_id, inactive_time in sessions_to_check:
            if session_id in active_sessions:
                # Check if this session is truly orphaned by attempting to ping it
                # In a real implementation, you might check a heartbeat status or connection state
                # For now, we'll just clean it up based on the timeout
                print(f"Cleaning up orphaned session {session_id} after {inactive_time:.1f} seconds of inactivity (threshold: {DISCONNECT_TIMEOUT}s)")
                session = active_sessions[session_id]
                session.cleanup()
                del active_sessions[session_id]
                del session_last_activity[session_id]
                print(f"Session {session_id} successfully cleaned up, {len(active_sessions)} active sessions remaining")
        
        # Check every 5 seconds
        await asyncio.sleep(DISCONNECT_CHECK_INTERVAL)

async def log_active_sessions():
    """Background task to log active sessions every 30 seconds"""
    while True:
        if active_sessions:
            current_time = time.time()
            print(f"\n===== Active Sessions Report ({len(active_sessions)} sessions) =====")
            for session_id, session in active_sessions.items():
                last_activity = session_last_activity.get(session_id, current_time)
                idle_time = current_time - last_activity
                print(f"  - {session_id}: {session.agent_type} | {session.language_pair} | Latency: {session.args.latency_multiplier}x | Idle: {idle_time:.1f}s")
            print("=============================================\n")
        
        # Log every 30 seconds
        await asyncio.sleep(30)

@app.on_event("startup")
async def startup_event():
    """Start background tasks when the application starts"""
    asyncio.create_task(cleanup_idle_sessions())
    asyncio.create_task(check_orphaned_sessions())
    asyncio.create_task(log_active_sessions())

@app.post("/init")
async def initialize_translation(agent_type: str, language_pair: str, latency_multiplier: int = 2):
    global args
    session_args = copy.deepcopy(args)
    session_args.latency_multiplier = latency_multiplier
    session_args.max_new_tokens = 10 * latency_multiplier
    session = TranslationSession(agent_type, language_pair, session_args)
    session_id = f"{agent_type}_{language_pair}_{len(active_sessions)}"
    active_sessions[session_id] = session
    session_last_activity[session_id] = time.time()
    return {"session_id": session_id}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    if session_id not in active_sessions:
        await websocket.close(code=4000, reason="Invalid session ID")
        return
        
    session = active_sessions[session_id]
    chunk_count = 0
    update_session_activity(session_id)
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Update activity timestamp
            update_session_activity(session_id)
            
            # Convert bytes to numpy array
            audio_data = np.frombuffer(data, dtype=np.float32)
            chunk_count += 1
            print(f"Received chunk {chunk_count}, size: {len(audio_data)}")
            
            # Process the segment
            translation = await session.process_segment(audio_data)
            
            if translation:
                print(f"Got translation for chunk {chunk_count}: {translation}")
                await websocket.send_text(translation)
                
    except Exception as e:
        print(f"Error in WebSocket connection: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # This block will execute when the WebSocket connection is closed
        print(f"WebSocket connection closed for session {session_id}")
        # Don't immediately delete the session, let the idle cleanup handle it
        # This allows reconnection if the page is refreshed

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    audio_data, sr = sf.read(io.BytesIO(contents))
    return {"sample_rate": sr, "duration": len(audio_data) / sr}

@app.post("/update_latency")
async def update_latency(session_id: str, latency_multiplier: int):
    if session_id not in active_sessions:
        return {"success": False, "error": "Invalid session ID"}
    
    try:
        update_session_activity(session_id)
        session = active_sessions[session_id]
        session.args.latency_multiplier = int(latency_multiplier)
        session.agent.update_multiplier(int(latency_multiplier))
        return {"success": True}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.post("/heartbeat")
async def heartbeat(session_id: str):
    """Lightweight endpoint to keep a session alive"""
    if session_id not in active_sessions:
        return {"success": False, "error": "Invalid session ID"}
    
    # Get the current time
    current_time = time.time()
    
    # Get the previous activity time if it exists
    previous_activity = session_last_activity.get(session_id, current_time)
    
    # Calculate time since last activity
    time_since_last_activity = current_time - previous_activity
    
    # Update the session's last activity timestamp
    update_session_activity(session_id)
    
    # Get session info
    session = active_sessions[session_id]
    
    return {
        "success": True, 
        "timestamp": current_time,
        "session_info": {
            "id": session_id,
            "type": session.agent_type,
            "language_pair": session.language_pair,
            "latency_multiplier": session.args.latency_multiplier,
            "last_activity_seconds_ago": time_since_last_activity,
            "created_at": session_last_activity.get(session_id, current_time)
        }
    }

@app.post("/reset_translation")
async def reset_translation(session_id: str):
    if session_id not in active_sessions:
        return {"success": False, "error": "Invalid session ID"}
    
    try:
        update_session_activity(session_id)
        session = active_sessions[session_id]
        # Reset the states without reloading the model
        session.states.reset()
        return {"success": True, "message": "Translation reset successfully"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.post("/delete_session")
async def delete_session(session_id: Optional[str] = None, request: Request = None):
    # Handle both query parameters and form data (for navigator.sendBeacon)
    if session_id is None and request:
        try:
            # Try to get session_id from form data (sent by navigator.sendBeacon)
            form = await request.form()
            if 'session_id' in form:
                session_id = form['session_id']
        except:
            # If form parsing fails, try to get from JSON body
            try:
                body = await request.json()
                if 'session_id' in body:
                    session_id = body['session_id']
            except:
                pass
    
    if not session_id:
        return {"success": False, "error": "No session ID provided"}
        
    if session_id not in active_sessions:
        return {"success": False, "error": "Invalid session ID"}
    
    try:
        # Get the session
        session = active_sessions[session_id]
        
        # Clean up GPU resources
        session.cleanup()
        
        # Delete the session
        del active_sessions[session_id]
        if session_id in session_last_activity:
            del session_last_activity[session_id]
            
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return {"success": True, "message": "Session deleted and resources cleaned up successfully"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

# Mount static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    StreamLlama.add_args(parser)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=8000) 