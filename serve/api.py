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
# Track the last ping time for each session to detect closed/refreshed webpages
session_last_ping: Dict[str, float] = {}

# Short timeout for detecting browser disconnections
DISCONNECT_CHECK_INTERVAL = 5  # Check every 5 seconds
# Timeout for detecting closed/refreshed webpages (15 seconds without a ping)
WEBPAGE_DISCONNECT_TIMEOUT = 15  # Consider a webpage closed if no ping for 15 seconds
# DISCONNECT_TIMEOUT is no longer used since orphaned sessions are now tracked client-side

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
    
    def reset(self):
        """Reset the translation state without reloading the model"""
        if hasattr(self.states, 'reset'):
            self.states.reset()
        print(f"Reset translation state for session with {self.agent_type} model")
    
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
    """Update the last activity timestamp for a session.
    This timestamp is used for tracking session activity but not for orphan detection."""
    if session_id in active_sessions:
        session_last_activity[session_id] = time.time()

def update_session_ping(session_id: str):
    """Update the last ping timestamp for a session.
    This is used to detect if the webpage is still open."""
    if session_id in active_sessions:
        session_last_ping[session_id] = time.time()
        # Also update activity timestamp
        session_last_activity[session_id] = time.time()

async def check_orphaned_sessions():
    """Background task to check for orphaned sessions every 5 seconds.
    A session is considered orphaned if:
    1. It wasn't properly deleted when the browser was closed/refreshed (tracked client-side)
    2. The webpage hasn't sent a ping in WEBPAGE_DISCONNECT_TIMEOUT seconds (15s by default)
    """
    while True:
        current_time = time.time()
        sessions_to_delete = []
        
        # Check for sessions without recent pings (closed/refreshed webpages)
        for session_id in list(active_sessions.keys()):
            # Skip sessions that don't have a ping record yet (new sessions)
            if session_id not in session_last_ping:
                continue
                
            last_ping = session_last_ping[session_id]
            time_since_last_ping = current_time - last_ping
            
            # If no ping received for WEBPAGE_DISCONNECT_TIMEOUT seconds, consider the webpage closed
            if time_since_last_ping > WEBPAGE_DISCONNECT_TIMEOUT:
                print(f"Session {session_id} detected as orphaned: no ping for {time_since_last_ping:.1f}s (threshold: {WEBPAGE_DISCONNECT_TIMEOUT}s)")
                sessions_to_delete.append(session_id)
        
        # Delete orphaned sessions
        for session_id in sessions_to_delete:
            try:
                if session_id in active_sessions:
                    session = active_sessions[session_id]
                    print(f"Cleaning up orphaned session {session_id} (webpage closed/refreshed)")
                    
                    # Clean up GPU resources
                    session.cleanup()
                    
                    # Remove from active sessions
                    del active_sessions[session_id]
                    
                    # Remove from activity tracking
                    if session_id in session_last_activity:
                        del session_last_activity[session_id]
                        
                    # Remove from ping tracking
                    if session_id in session_last_ping:
                        del session_last_ping[session_id]
                    
                    # Force garbage collection
                    gc.collect()
            except Exception as e:
                print(f"Error cleaning up orphaned session {session_id}: {e}")
        
        # Log active sessions count periodically
        if active_sessions:
            print(f"Active sessions: {len(active_sessions)}")
        
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
                inactivity_time = current_time - last_activity
                print(f"  - {session_id}: {session.agent_type} | {session.language_pair} | Latency: {session.args.latency_multiplier}x | Inactive for: {inactivity_time:.1f}s")
            print("=============================================\n")
        
        # Log every 30 seconds
        await asyncio.sleep(30)

@app.on_event("startup")
async def startup_event():
    """Start background tasks when the application starts"""
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
    # Initialize ping timestamp
    session_last_ping[session_id] = time.time()
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
    # Update ping timestamp when WebSocket connection is established
    update_session_ping(session_id)
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Update activity timestamp
            update_session_activity(session_id)
            # Update ping timestamp when data is received
            update_session_ping(session_id)
            
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
    """Update the latency multiplier for a session."""
    try:
        if session_id not in active_sessions:
            return {"success": False, "error": "Invalid session ID"}
        
        # Get the session
        session = active_sessions[session_id]
        
        # Update the latency multiplier
        session.args.latency_multiplier = latency_multiplier
        session.agent.update_multiplier(int(latency_multiplier))
        
        # Update the last activity timestamp
        update_session_activity(session_id)
        # Update ping timestamp
        update_session_ping(session_id)
        
        print(f"Updated latency multiplier for session {session_id} to {latency_multiplier}x")
        
        return {"success": True}
    except Exception as e:
        print(f"Error updating latency: {e}")
        return {"success": False, "error": str(e)}

@app.post("/reset_translation")
async def reset_translation(session_id: str):
    """Reset the translation state for a session."""
    try:
        if session_id not in active_sessions:
            return {"success": False, "error": "Invalid session ID"}
        
        # Get the session
        session = active_sessions[session_id]
        
        # Reset the translation state
        session.reset()
        
        # Update the last activity timestamp
        update_session_activity(session_id)
        # Update ping timestamp
        update_session_ping(session_id)
        
        print(f"Reset translation state for session {session_id}")
        
        return {"success": True, "message": "Translation state reset successfully"}
    except Exception as e:
        print(f"Error resetting translation: {e}")
        return {"success": False, "error": str(e)}

@app.post("/delete_session")
async def delete_session(request: Request, session_id: Optional[str] = None):
    """Delete a session and clean up its resources."""
    try:
        # Check if session_id is provided in query parameters
        if session_id is None:
            # If not, try to get it from form data (for sendBeacon)
            form_data = await request.form()
            session_id = form_data.get("session_id")
            
            # If still not found, return an error
            if session_id is None:
                return {"success": False, "error": "No session_id provided"}
        
        # Check if the session exists
        if session_id not in active_sessions:
            return {"success": False, "error": "Invalid session ID"}
        
        # Get the session
        session = active_sessions[session_id]
        
        # Clean up GPU resources
        session.cleanup()
        
        # Remove the session from active sessions
        del active_sessions[session_id]
        
        # Remove the session from last activity tracking
        if session_id in session_last_activity:
            del session_last_activity[session_id]
        
        # Remove the session from ping tracking
        if session_id in session_last_ping:
            del session_last_ping[session_id]
        
        # Force garbage collection to free up memory
        gc.collect()
        
        print(f"Session {session_id} deleted, {len(active_sessions)} active sessions remaining")
        
        return {"success": True}
    except Exception as e:
        print(f"Error deleting session: {e}")
        return {"success": False, "error": str(e)}

@app.post("/ping")
async def ping_session(session_id: str):
    """Update the last ping timestamp for a session to indicate the webpage is still open."""
    try:
        if session_id not in active_sessions:
            return {"success": False, "error": "Invalid session ID"}
            
        # Update the last ping timestamp
        update_session_ping(session_id)
        
        return {"success": True}
    except Exception as e:
        print(f"Error updating ping: {e}")
        return {"success": False, "error": str(e)}

# Mount static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    StreamLlama.add_args(parser)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=8000) 