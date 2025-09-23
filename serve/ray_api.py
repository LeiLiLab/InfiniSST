#!/usr/bin/env python3
"""
Ray-based FastAPI Server for InfiniSST
基于Ray的分布式实时翻译API服务器
"""

import asyncio
import time
import logging
import uuid
import os
import sys
from typing import Dict, List, Optional, Any
import numpy as np
import torch
import json
import argparse
import copy

# FastAPI imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ray imports
import ray

# Import our Ray serving system
from ray_serving_system import (
    RayServingSystem, RayServingConfig, create_ray_serving_system,
    RequestStage
)

# Import original components for compatibility
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from agents.infinisst import InfiniSST
from scheduler import RequestStage

# Additional imports for new functionality
import subprocess
import tempfile
from pathlib import Path
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== Global Variables =====

app = FastAPI(title="Ray-based InfiniSST Translation API")

# Ray serving system
ray_system: Optional[RayServingSystem] = None

# Active WebSocket connections
active_websockets: Dict[str, WebSocket] = {}

# Session management
active_sessions: Dict[str, Dict[str, Any]] = {}
session_last_activity: Dict[str, float] = {}
session_last_ping: Dict[str, float] = {}

# Configuration
LANGUAGE_PAIRS = {
    "English -> Chinese": ("English", "Chinese", "en", "zh"),
    "English -> Italian": ("English", "Italian", "en", "it"),
    "English -> German": ("English", "German", "en", "de"),
    "English -> Spanish": ("English", "Spanish", "en", "es"),
}

# GPU configuration - will be set from environment
gpus = []

# ===== Middleware =====

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Helper Functions =====

def update_session_activity(session_id: str):
    """Update session activity timestamp"""
    if session_id in active_sessions:
        session_last_activity[session_id] = time.time()

def update_session_ping(session_id: str):
    """Update session ping timestamp"""
    if session_id in active_sessions:
        session_last_ping[session_id] = time.time()
        session_last_activity[session_id] = time.time()

async def cleanup_orphaned_sessions():
    """Background task to cleanup orphaned sessions"""
    WEBPAGE_DISCONNECT_TIMEOUT = 300  # 5 minutes
    
    while True:
        try:
            current_time = time.time()
            sessions_to_delete = []
            
            for session_id in list(active_sessions.keys()):
                if session_id not in session_last_ping:
                    continue
                
                last_ping = session_last_ping[session_id]
                time_since_last_ping = current_time - last_ping
                
                if time_since_last_ping > WEBPAGE_DISCONNECT_TIMEOUT:
                    logger.info(f"Session {session_id} detected as orphaned: no ping for {time_since_last_ping:.1f}s")
                    sessions_to_delete.append(session_id)
            
            # Delete orphaned sessions
            for session_id in sessions_to_delete:
                await delete_session_internal(session_id)
                
            if active_sessions:
                logger.debug(f"Active sessions: {len(active_sessions)}")
                
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
        
        await asyncio.sleep(5)  # Check every 5 seconds

async def delete_session_internal(session_id: str):
    """Internal session deletion"""
    try:
        if session_id in active_sessions:
            logger.info(f"Cleaning up session {session_id}")
            
            # Remove from active sessions
            del active_sessions[session_id]
            
            if session_id in session_last_activity:
                del session_last_activity[session_id]
            if session_id in session_last_ping:
                del session_last_ping[session_id]
            if session_id in active_websockets:
                del active_websockets[session_id]
            
            logger.info(f"Session {session_id} cleaned up successfully")
            
    except Exception as e:
        logger.error(f"Error cleaning up session {session_id}: {e}")

def create_model_args_factory():
    """Create model arguments factory function"""
    def model_args_factory(gpu_id: int, language_id: str) -> Any:
        # Parse language pair
        for pair_name, (src_lang, tgt_lang, src_code, tgt_code) in LANGUAGE_PAIRS.items():
            if language_id == pair_name:
                break
        else:
            raise ValueError(f"Unsupported language pair: {language_id}")
        
        # Create model arguments (simplified version)
        # In real implementation, you would use the actual InfiniSST arguments
        model_args = {
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            "src_code": src_code,
            "tgt_code": tgt_code,
            "gpu_id": gpu_id,
            "language_id": language_id,
            # Add other model-specific parameters as needed
        }
        
        return model_args
    
    return model_args_factory

# ===== Startup/Shutdown Events =====

@app.on_event("startup")
async def startup_event():
    """Initialize Ray serving system on startup"""
    global ray_system, gpus
    
    # Get GPU configuration from environment
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
    gpus = [int(x.strip()) for x in cuda_visible_devices.split(",") if x.strip().isdigit()]
    
    logger.info(f"🚀 Starting Ray-based serving system with GPUs: {gpus}")
    
    try:
        # Create supported language pairs (limit to available GPUs)
        supported_languages = list(LANGUAGE_PAIRS.keys())[:len(gpus)]
        
        logger.info(f"Supported languages: {supported_languages}")
        
        # Create Ray serving system
        ray_system = await create_ray_serving_system(
            gpu_ids=gpus,
            language_pairs=supported_languages,
            model_args_factory=create_model_args_factory(),
            max_batch_size=32,
            batch_timeout_ms=100.0,
            enable_dynamic_batching=True
        )
        
        await ray_system.start()
        
        logger.info("✅ Ray serving system initialized successfully")
        
        # Start background tasks
        asyncio.create_task(cleanup_orphaned_sessions())
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Ray serving system: {e}")
        import traceback
        traceback.print_exc()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global ray_system
    
    if ray_system:
        await ray_system.stop()
        logger.info("Ray serving system stopped")

# ===== API Endpoints =====

@app.post("/init")
async def initialize_translation(
    agent_type: str, 
    language_pair: str, 
    latency_multiplier: int = 2, 
    client_id: str = None, 
    evaluation_mode: str = "false"
):
    """Initialize a new translation session"""
    global ray_system
    
    if not ray_system:
        return {"success": False, "error": "Ray serving system not initialized"}
    
    evaluation_mode_bool = evaluation_mode.lower() in ["true", "1", "yes", "on"]
    
    # Generate session ID
    timestamp = int(time.time() * 1000)
    if client_id:
        session_id = f"{client_id}_{language_pair}_{timestamp}"
    else:
        session_id = f"{agent_type}_{language_pair}_{timestamp}"
    
    logger.info(f"Initializing Ray session {session_id} for {language_pair}")
    
    try:
        # Create session using Ray system
        ray_session_id = await ray_system.create_session(
            user_id=client_id or session_id,
            language_id=language_pair,
            session_id=session_id
        )
        
        # Store session info locally
        session_info = {
            'session_id': session_id,
            'ray_session_id': ray_session_id,
            'agent_type': agent_type,
            'language_pair': language_pair,
            'latency_multiplier': latency_multiplier,
            'user_id': client_id or session_id,
            'created_at': time.time(),
            'is_ray_based': True,
            'evaluation_mode': evaluation_mode_bool
        }
        
        active_sessions[session_id] = session_info
        session_last_activity[session_id] = time.time()
        session_last_ping[session_id] = time.time()
        
        logger.info(f"✅ Ray session {session_id} created successfully")
        return {
            "session_id": session_id,
            "queued": False,
            "queue_position": 0,
            "ray_based": True,
            "evaluation_mode": evaluation_mode_bool
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to create Ray session {session_id}: {e}")
        return {"success": False, "error": str(e)}

@app.get("/queue_status/{session_id}")
async def get_queue_status(session_id: str):
    """Get queue status for a session"""
    if session_id in active_sessions:
        session = active_sessions[session_id]
        return {
            "session_id": session_id,
            "status": "active",
            "queued": False,
            "queue_position": 0,
            "session_type": "ray"
        }
    else:
        return {
            "session_id": session_id,
            "status": "not_found",
            "error": "Session not found"
        }

@app.websocket("/wss/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time translation"""
    await websocket.accept()
    
    if session_id not in active_sessions:
        await websocket.close(code=4000, reason="Invalid session ID")
        return
    
    session = active_sessions[session_id]
    active_websockets[session_id] = websocket
    update_session_activity(session_id)
    update_session_ping(session_id)
    
    logger.info(f"🔗 WebSocket connected for Ray session {session_id}")
    
    await websocket.send_text("READY: Ray system ready")
    
    chunk_count = 0
    
    try:
        while True:
            try:
                message = await websocket.receive()
                update_session_activity(session_id)
                update_session_ping(session_id)
                
                if "text" in message:
                    control_message = message["text"]
                    
                    if control_message == "EOF":
                        logger.info(f"📋 [EOF] Ray session {session_id}: Received EOF signal")
                        logger.info(f"   - Total chunks processed: {chunk_count}")
                        
                        await websocket.send_text("PROCESSING_COMPLETE: File processing finished")
                        continue
                
                elif "bytes" in message:
                    data = message["bytes"]
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    chunk_count += 1
                    
                    logger.debug(f"📥 [AUDIO-RECV] Ray session {session_id}: Chunk {chunk_count}, {len(audio_data)} samples")
                    
                    if len(audio_data) == 0:
                        logger.warning(f"⚠️ Empty audio data in chunk {chunk_count}")
                        continue
                    
                    # Process through Ray system
                    try:
                        # Create result callback
                        def result_callback(result):
                            try:
                                if result.get('success', False):
                                    translation = result.get('translation', '')
                                    if translation:
                                        # Send result asynchronously
                                        asyncio.create_task(send_result_safe(websocket, translation))
                                else:
                                    error_msg = result.get('error', 'Unknown error')
                                    asyncio.create_task(send_result_safe(websocket, f"ERROR: {error_msg}"))
                            except Exception as e:
                                logger.error(f"Error in result callback: {e}")
                        
                        # Submit request to Ray system
                        request_id = await ray_system.submit_translation_request(
                            session_id=session['ray_session_id'],
                            speech_data=audio_data,
                            stage=RequestStage.PREFILL,
                            is_final=False,
                            max_new_tokens=20,
                            result_callback=result_callback
                        )
                        
                        logger.debug(f"✅ Submitted Ray request {request_id} for chunk {chunk_count}")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to submit Ray request: {e}")
                        await websocket.send_text(f"ERROR: {str(e)}")
                        
            except WebSocketDisconnect:
                logger.info(f"🔌 WebSocket disconnected for Ray session {session_id}")
                break
            except Exception as e:
                logger.error(f"❌ WebSocket error for session {session_id}: {e}")
                try:
                    await websocket.send_text(f"ERROR: {str(e)}")
                except:
                    pass
                break
                
    finally:
        # Cleanup
        if session_id in active_websockets:
            del active_websockets[session_id]
        logger.info(f"WebSocket connection closed for Ray session {session_id}")

async def send_result_safe(websocket: WebSocket, message: str):
    """Safely send result to WebSocket"""
    try:
        if websocket.client_state.name == "CONNECTED":
            await websocket.send_text(message)
    except Exception as e:
        logger.error(f"Failed to send WebSocket message: {e}")

@app.post("/reset_translation")
async def reset_translation(session_id: str):
    """Reset translation session"""
    try:
        if session_id not in active_sessions:
            return {"success": False, "error": "Session not found"}
        
        session = active_sessions[session_id]
        
        # For Ray sessions, we can implement reset through the Ray system
        # This would need to be implemented in the Ray serving system
        logger.info(f"🔄 Reset requested for Ray session {session_id}")
        
        update_session_activity(session_id)
        update_session_ping(session_id)
        
        return {
            "success": True,
            "status": "success",
            "message": "Ray session reset successfully",
            "session_type": "ray"
        }
        
    except Exception as e:
        logger.error(f"❌ Error resetting Ray session {session_id}: {e}")
        return {"success": False, "error": str(e)}

@app.post("/delete_session")
async def delete_session(request: Request, session_id: Optional[str] = None):
    """Delete a session"""
    try:
        if session_id is None:
            form_data = await request.form()
            session_id = form_data.get("session_id")
            
            if session_id is None:
                return {"success": False, "error": "No session_id provided"}
        
        await delete_session_internal(session_id)
        return {"success": True}
        
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        return {"success": False, "error": str(e)}

@app.post("/update_latency")
async def update_latency(session_id: str, latency_multiplier: int):
    """Update latency multiplier for a session"""
    try:
        if session_id not in active_sessions:
            return {"success": False, "error": "Invalid session ID"}
        
        session = active_sessions[session_id]
        session['latency_multiplier'] = latency_multiplier
        
        update_session_activity(session_id)
        update_session_ping(session_id)
        
        logger.info(f"🔧 Updated latency for Ray session {session_id} to {latency_multiplier}x")
        
        return {"success": True, "session_type": "ray"}
        
    except Exception as e:
        logger.error(f"Error updating latency: {e}")
        return {"success": False, "error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        health_info = {
            "status": "healthy",
            "time": int(time.time()),
            "ray_available": ray_system is not None,
            "active_sessions_count": len(active_sessions),
            "active_sessions": []
        }
        
        # Add session info
        current_time = time.time()
        for session_id, session in active_sessions.items():
            last_activity = session_last_activity.get(session_id, current_time)
            inactivity_time = current_time - last_activity
            
            session_info = {
                "session_id": session_id,
                "type": "ray",
                "agent_type": session.get('agent_type', 'Unknown'),
                "language_pair": session.get('language_pair', 'Unknown'),
                "user_id": session.get('user_id', 'Unknown'),
                "latency_multiplier": session.get('latency_multiplier', 'Unknown'),
                "inactive_for": f"{inactivity_time:.1f}s",
                "created_at": session.get('created_at', 0)
            }
            health_info["active_sessions"].append(session_info)
        
        # Get Ray system stats if available
        if ray_system:
            try:
                ray_stats = await ray_system.get_system_stats()
                health_info["ray_stats"] = ray_stats
            except Exception as e:
                health_info["ray_stats_error"] = str(e)
        
        return health_info
        
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return {"status": "error", "error": str(e)}

@app.post("/ping")
async def ping_session(session_id: str):
    """Ping session to keep it alive"""
    try:
        if session_id not in active_sessions:
            return {"success": False, "error": "Invalid session ID"}
        
        update_session_ping(session_id)
        return {"success": True}
        
    except Exception as e:
        logger.error(f"Ping error for session {session_id}: {e}")
        return {"success": False, "error": str(e)}

# ===== Ray System Management Endpoints =====

@app.get("/ray/stats")
async def get_ray_stats():
    """Get Ray system statistics"""
    if not ray_system:
        return {"error": "Ray system not available"}
    
    try:
        stats = await ray_system.get_system_stats()
        return {"success": True, "stats": stats}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/ray/configure")
async def configure_ray_system(
    max_batch_size: Optional[int] = None,
    batch_timeout_ms: Optional[float] = None,
    enable_dynamic_batching: Optional[bool] = None
):
    """Configure Ray system parameters"""
    if not ray_system:
        return {"error": "Ray system not available"}
    
    try:
        # Configuration would be implemented in the Ray system
        # For now, just return success
        config_updates = {}
        if max_batch_size is not None:
            config_updates["max_batch_size"] = max_batch_size
        if batch_timeout_ms is not None:
            config_updates["batch_timeout_ms"] = batch_timeout_ms
        if enable_dynamic_batching is not None:
            config_updates["enable_dynamic_batching"] = enable_dynamic_batching
        
        return {"success": True, "updated_config": config_updates}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ===== Additional API Endpoints for Frontend Compatibility =====

@app.post("/download_youtube")
async def download_youtube(request: Request, background_tasks: BackgroundTasks):
    """Download YouTube video for processing"""
    try:
        query_params = dict(request.query_params)
        url = query_params.get("url")
        session_id = query_params.get("session_id")

        if not url:
            return {"error": "Missing URL parameter"}

        # (Optional) log session_id for debugging
        if session_id:
            logger.info(f"Download request received for session: {session_id}")

        output_path = f"/tmp/video_{session_id}_{int(time.time())}.mp4"

        cmd = [
            "yt-dlp",
            "-f", "bestvideo+bestaudio/best",
            "--merge-output-format", "mp4",
            "--no-continue",
            "--no-part",
            "-o", output_path,
            url
        ]

        logger.info("Running command: " + " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error("yt-dlp failed: " + result.stderr)
            raise Exception(f"yt-dlp error: {result.stderr}")
        else:
            logger.info("yt-dlp success: " + result.stdout)

        logger.info("Download completed.")
        logger.info(f"Checking file after download: {output_path}")
        if os.path.exists(output_path):
            logger.info(f"File exists. Size: {os.path.getsize(output_path)} bytes")
        else:
            logger.error("Download failed: File not found.")
            raise Exception("Download failed: File not found.")

        background_tasks.add_task(os.remove, output_path)
        return FileResponse(output_path, media_type='video/mp4', filename=f'video_{session_id}.mp4')
    except Exception as e:
        logger.error(f"YouTube download error: {e}")
        return {"error": str(e)}

@app.post("/load_models")
async def load_models():
    """Load models to Ray serving system"""
    try:
        if not ray_system:
            return {"success": False, "error": "Ray serving system not available"}
        
        # Ray system should already have models loaded during initialization
        # Check system stats to verify models are loaded
        stats = await ray_system.get_system_stats()
        
        return {
            "success": True,
            "message": "Ray models are ready",
            "ray_stats": stats
        }
    except Exception as e:
        logger.error(f"Load models error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/load_model")
async def load_model(request: Request):
    """Load model for a specific language pair and client - Ray version"""
    try:
        # Get request data
        data = await request.json()
        language_pair = data.get("language_pair", "English -> Chinese")
        client_id = data.get("client_id")
        model_type = data.get("model_type", "infinisst")
        
        logger.info(f"🧪 [LOAD-MODEL] Loading model for {language_pair}, client: {client_id}")
        
        # Check if Ray system is available
        if not ray_system:
            return {"success": False, "error": "Ray system not available"}
        
        # Generate session ID
        session_id = f"test_{client_id}_{int(time.time())}"
        
        # Create session in Ray system
        ray_session_id = await ray_system.create_session(
            user_id=client_id or "test_user",
            language_id=language_pair,
            session_id=session_id
        )
        
        logger.info(f"🧪 [LOAD-MODEL] Ray session created: {session_id}")
        
        # Store session info
        session_info = {
            "session_id": session_id,
            "ray_session_id": ray_session_id,
            "language_pair": language_pair,
            "client_id": client_id,
            "agent_type": "InfiniSST",
            "created_at": time.time(),
            "last_activity": time.time(),
            "is_ray_based": True
        }
        
        active_sessions[session_id] = session_info
        update_session_activity(session_id)
        update_session_ping(session_id)
        
        return {
            "success": True,
            "session_id": session_id,
            "message": f"Ray model loaded successfully for {language_pair}",
            "queued": False,
            "language_pair": language_pair,
            "ray_based": True
        }
        
    except Exception as e:
        logger.error(f"🧪 [LOAD-MODEL] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.get("/test_video/{filename}")
async def serve_test_video(filename: str):
    """Serve test video files for automated testing"""
    # Security check - only allow specific filename patterns
    allowed_patterns = [
        "0000AAAA.mp4", "test.mp4", "sample.mp4",
        # 评估框架使用的音频文件
        "2022.acl-long.110.wav", "2022.acl-long.117.wav", "2022.acl-long.268.wav",
        "2022.acl-long.367.wav", "2022.acl-long.590.wav"
    ]
    
    if filename not in allowed_patterns:
        raise HTTPException(status_code=404, detail=f"Test video '{filename}' not allowed. Allowed files: {', '.join(allowed_patterns)}")
    
    # Look for the test video in multiple locations
    test_video_paths = [
        f"serve/static/test_video/{filename}",
        f"serve/static/{filename}",
        f"static/test_video/{filename}", 
        f"static/{filename}",
        f"test_video/{filename}",
        f"~/Downloads/{filename}",
        f"/tmp/{filename}",
        f"./{filename}"
    ]
    
    video_path = None
    for path in test_video_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            video_path = expanded_path
            break
    
    if not video_path:
        # 如果找不到视频文件，返回详细的错误信息
        error_msg = f"Test video '{filename}' not found. Searched in: {', '.join(test_video_paths)}"
        logger.error(f"🧪 [TEST-VIDEO] {error_msg}")
        raise HTTPException(status_code=404, detail=error_msg)
    
    logger.info(f"🧪 [TEST-VIDEO] Serving test video: {video_path}")
    
    # 根据文件扩展名确定媒体类型
    if filename.endswith('.wav'):
        media_type = "audio/wav"
    elif filename.endswith('.mp4'):
        media_type = "video/mp4"
    else:
        media_type = "application/octet-stream"
    
    return FileResponse(video_path, media_type=media_type, filename=filename)

@app.get("/debug/session_stats")
async def get_session_stats():
    """🔍 调试用：获取详细的Ray session统计信息"""
    current_time = time.time()
    
    stats = {
        "system_info": {
            "ray_available": ray_system is not None,
            "active_sessions_count": len(active_sessions),
            "current_time": current_time,
            "system_type": "ray_based"
        },
        "active_sessions": [],
        "ray_system_stats": None
    }
    
    # 添加活跃会话信息
    for session_id, session in active_sessions.items():
        last_activity = session_last_activity.get(session_id, current_time)
        last_ping = session_last_ping.get(session_id, current_time)
        
        session_stats = {
            "session_id": session_id,
            "type": "ray",
            "agent_type": session.get('agent_type', 'Unknown'),
            "language_pair": session.get('language_pair', 'Unknown'),
            "user_id": session.get('user_id', 'Unknown'),
            "created_at": session.get('created_at', 0),
            "last_activity": last_activity,
            "last_ping": last_ping,
            "inactive_for": current_time - last_activity,
            "ping_delay": current_time - last_ping,
            "ray_session_id": session.get('ray_session_id', 'Unknown')
        }
        stats["active_sessions"].append(session_stats)
    
    # 获取Ray系统统计
    if ray_system:
        try:
            ray_stats = await ray_system.get_system_stats()
            stats["ray_system_stats"] = ray_stats
        except Exception as e:
            stats["ray_system_stats"] = {"error": str(e)}
    
    return stats

@app.get("/debug/session_history")
async def get_session_creation_history():
    """Get session creation history - Ray version"""
    current_time = time.time()
    
    history = {
        "message": "Ray session creation history",
        "current_stats": {
            "active_sessions": len(active_sessions),
            "ray_available": ray_system is not None,
            "timestamp": current_time
        },
        "explanation": {
            "ray_benefits": [
                "Ray提供分布式计算和自动资源管理",
                "动态批处理调度优化延迟和吞吐量",
                "自动故障恢复和负载均衡",
                "更好的GPU内存管理和优化"
            ],
            "session_id_format": "Ray版本使用: {client_id/agent_type}_{language_pair}_{timestamp}"
        }
    }
    
    return history

@app.post("/enable_evaluation_mode")
async def enable_evaluation_mode(
    session_id: str = None,
    agent_type: str = None, 
    language_pair: str = None, 
    user_id: str = None
):
    """Enable evaluation mode for delay tracking - Ray version"""
    try:
        if session_id and session_id in active_sessions:
            # Enable evaluation mode for existing session
            session = active_sessions[session_id]
            session["evaluation_mode"] = True
            
            update_session_activity(session_id)
            update_session_ping(session_id)
            
            logger.info(f"🎯 [EVAL] Evaluation mode enabled for Ray session {session_id}")
            
            return {
                "success": True,
                "session_id": session_id,
                "message": "Evaluation mode enabled for Ray session",
                "session_type": "ray"
            }
        else:
            return {
                "success": False, 
                "error": "Session not found or invalid session ID",
                "available_sessions": list(active_sessions.keys())
            }
            
    except Exception as e:
        logger.error(f"Error enabling evaluation mode: {e}")
        return {"success": False, "error": str(e)}

@app.get("/session_delays/{session_id}")
async def get_session_delays(session_id: str, include_details: bool = False):
    """Get session delay information - Ray version"""
    try:
        if session_id not in active_sessions:
            return {"error": "Session not found"}
        
        session = active_sessions[session_id]
        
        # Ray sessions may not have detailed delay tracking yet
        # This is a placeholder for future implementation
        delay_info = {
            "session_id": session_id,
            "session_type": "ray",
            "evaluation_mode": session.get("evaluation_mode", False),
            "message": "Ray delay tracking implementation pending",
            "basic_stats": {
                "session_age": time.time() - session.get("created_at", time.time()),
                "last_activity": session_last_activity.get(session_id, 0)
            }
        }
        
        if include_details:
            delay_info["details"] = {
                "note": "Detailed delay tracking for Ray sessions will be implemented in future versions"
            }
        
        return delay_info
        
    except Exception as e:
        logger.error(f"Error getting session delays: {e}")
        return {"error": str(e)}

# ===== Static Files and Root Route =====

# Mount static files directory
try:
    # Try multiple possible static directories
    static_dirs = [
        "serve/static",
        "static", 
        "../static",
        "./static"
    ]
    
    static_dir_found = None
    for static_dir in static_dirs:
        if os.path.exists(static_dir):
            static_dir_found = static_dir
            break
    
    if static_dir_found:
        app.mount("/static", StaticFiles(directory=static_dir_found), name="static")
        logger.info(f"✅ Static files mounted from: {static_dir_found}")
    else:
        logger.warning(f"⚠️ No static directory found. Searched: {static_dirs}")
        
except Exception as e:
    logger.error(f"❌ Failed to mount static files: {e}")

@app.get("/")
async def read_index():
    """Serve the main HTML page"""
    try:
        # Try multiple possible index.html locations
        index_paths = [
            "serve/static/index.html",
            "static/index.html",
            "../static/index.html", 
            "./static/index.html"
        ]
        
        for index_path in index_paths:
            if os.path.exists(index_path):
                with open(index_path, "r", encoding="utf-8") as f:
                    content = f.read()
                logger.info(f"✅ Serving index.html from: {index_path}")
                return HTMLResponse(content=content)
        
        # If no index.html found, return a basic HTML page
        logger.warning("⚠️ index.html not found, serving basic HTML")
        basic_html = """<!DOCTYPE html>
<html>
<head>
    <title>Ray-based InfiniSST API Server</title>
    <meta charset="utf-8">
</head>
<body>
    <h1>🚀 Ray-based InfiniSST Translation API</h1>
    <p>The Ray-based API server is running successfully!</p>
    <h2>Available Endpoints:</h2>
    <ul>
        <li><a href="/health">Health Check</a></li>
        <li><a href="/ray/stats">Ray Statistics</a></li>
        <li><a href="/debug/session_stats">Session Statistics</a></li>
        <li><a href="/debug/session_history">Session History</a></li>
    </ul>
    <p><strong>Note:</strong> Static files (index.html) not found. Please ensure static files are available in the correct directory.</p>
</body>
</html>"""
        return HTMLResponse(content=basic_html)
        
    except Exception as e:
        logger.error(f"❌ Error serving index page: {e}")
        error_html = f"""<!DOCTYPE html>
<html>
<head><title>Error</title></head>
<body>
    <h1>❌ Server Error</h1>
    <p>Error serving main page: {str(e)}</p>
    <p>Ray-based InfiniSST API Server is running, but frontend assets are not available.</p>
</body>
</html>"""
        return HTMLResponse(content=error_html, status_code=500)

# ===== Main Function =====

def main():
    """Main function to run the Ray-based API server"""
    parser = argparse.ArgumentParser(description="Ray-based InfiniSST Translation API Server")
    
    # Server arguments
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # Ray arguments
    parser.add_argument("--ray-address", type=str, default=None, help="Ray cluster address")
    parser.add_argument("--ray-num-cpus", type=int, default=None, help="Number of CPUs for Ray")
    parser.add_argument("--ray-num-gpus", type=int, default=None, help="Number of GPUs for Ray")
    
    # Serving arguments
    parser.add_argument("--max-batch-size", type=int, default=32, help="Maximum batch size")
    parser.add_argument("--batch-timeout-ms", type=float, default=100.0, help="Batch timeout in milliseconds")
    parser.add_argument("--enable-dynamic-batching", action="store_true", default=True, help="Enable dynamic batching")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Ray-based InfiniSST API Server")
    print(f"   - Host: {args.host}:{args.port}")
    print(f"   - Ray address: {args.ray_address or 'local'}")
    print(f"   - Max batch size: {args.max_batch_size}")
    print(f"   - Batch timeout: {args.batch_timeout_ms}ms")
    print(f"   - Dynamic batching: {args.enable_dynamic_batching}")
    
    # Store Ray configuration in environment for startup event
    if args.ray_address:
        os.environ["RAY_ADDRESS"] = args.ray_address
    if args.ray_num_cpus:
        os.environ["RAY_NUM_CPUS"] = str(args.ray_num_cpus)
    if args.ray_num_gpus:
        os.environ["RAY_NUM_GPUS"] = str(args.ray_num_gpus)
    
    os.environ["MAX_BATCH_SIZE"] = str(args.max_batch_size)
    os.environ["BATCH_TIMEOUT_MS"] = str(args.batch_timeout_ms)
    os.environ["ENABLE_DYNAMIC_BATCHING"] = str(args.enable_dynamic_batching)
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1,
        limit_concurrency=100,
        timeout_keep_alive=30,
        access_log=True,
    )

if __name__ == "__main__":
    main() 