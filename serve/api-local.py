#!/usr/bin/env python3
"""
本地测试用的API服务器
去掉所有GPU和模型推理部分，只保留FastAPI接口用于测试前端连接
"""

from fastapi import FastAPI, WebSocket, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import json
import asyncio
import argparse
import time
import random
from typing import Dict, Optional, Any
import uvicorn
import starlette.websockets

print("=== InfiniSST Local API Server ===")
print("This is a local test server without GPU/model inference")
print("All translation results are simulated for frontend testing")
print("=" * 50)

app = FastAPI(title="InfiniSST Local API", version="1.0.0-local")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 支持的翻译模型列表（模拟）
TRANSLATION_AGENTS = {
    "InfiniSST": "Mock InfiniSST Agent",
}

# 支持的语言方向
LANGUAGE_PAIRS = {
    "English -> Chinese": ("English", "Chinese", "en", "zh"),
    "English -> Italian": ("English", "Italian", "en", "it"),
    "English -> German": ("English", "German", "en", "de"),
    "English -> Spanish": ("English", "Spanish", "en", "es"),
}

# 模拟的会话存储
active_sessions: Dict[str, dict] = {}
session_last_activity: Dict[str, float] = {}
session_last_ping: Dict[str, float] = {}

# 模拟的队列和GPU映射
session_queue: list = []
session_gpu_map: Dict[str, int] = {}
queue_lock = asyncio.Lock()

# 模拟GPU列表
gpus = [0, 1]  # 模拟2个GPU
print(f"Simulated GPUs: {gpus}")

# 超时设置
DISCONNECT_CHECK_INTERVAL = 5
WEBPAGE_DISCONNECT_TIMEOUT = 60

# 模拟翻译会话类
class MockTranslationSession:
    def __init__(self, agent_type: str, language_pair: str, latency_multiplier: int = 2):
        self.agent_type = agent_type
        self.language_pair = language_pair
        self.latency_multiplier = latency_multiplier
        self.is_ready = False
        self.translation_buffer = ""
        
        # 模拟初始化延迟
        asyncio.create_task(self._mock_initialization())
    
    async def _mock_initialization(self):
        """模拟模型初始化过程"""
        await asyncio.sleep(1)  # 模拟1秒初始化时间
        self.is_ready = True
        print(f"Mock session initialized: {self.agent_type} for {self.language_pair}")
    
    async def wait_for_ready(self, timeout=60):
        """等待模拟初始化完成"""
        start_time = time.time()
        while not self.is_ready and time.time() - start_time < timeout:
            await asyncio.sleep(0.1)
        return self.is_ready
    
    async def process_segment(self, segment, is_last: bool = False):
        """模拟音频处理"""
        if not self.is_ready:
            return ""
        
        # 模拟翻译延迟
        await asyncio.sleep(0.1)
        
        # 模拟翻译结果
        mock_translations = {
            "English -> Chinese": [
                "这是一个模拟的翻译结果",
                "正在测试实时翻译功能",
                "InfiniSST 翻译系统正常工作",
                "感谢您的耐心测试"
            ],
            "English -> Italian": [
                "Questo è un risultato di traduzione simulato",
                "Sto testando la funzione di traduzione in tempo reale",
                "Il sistema di traduzione InfiniSST funziona normalmente",
                "Grazie per la vostra paziente prova"
            ],
            "English -> German": [
                "Dies ist ein simuliertes Übersetzungsergebnis",
                "Ich teste die Echtzeit-Übersetzungsfunktion",
                "Das InfiniSST-Übersetzungssystem funktioniert normal",
                "Vielen Dank für Ihren geduldigen Test"
            ],
            "English -> Spanish": [
                "Este es un resultado de traducción simulado",
                "Estoy probando la función de traducción en tiempo real",
                "El sistema de traducción InfiniSST funciona normalmente",
                "Gracias por su paciente prueba"
            ]
        }
        
        translations = mock_translations.get(self.language_pair, ["Mock translation result"])
        
        # 随机选择一个翻译结果
        if len(self.translation_buffer) < 100:  # 模拟累积翻译
            new_text = random.choice(translations)
            if self.translation_buffer:
                self.translation_buffer += " " + new_text
            else:
                self.translation_buffer = new_text
        
        return self.translation_buffer
    
    def reset(self):
        """重置翻译状态"""
        self.translation_buffer = ""
        return True
    
    def cleanup(self):
        """清理资源"""
        print(f"Cleaning up mock session: {self.agent_type}")

# 工具函数
def find_free_gpu():
    """模拟查找空闲GPU"""
    gpus_in_use = set(session_gpu_map.values())
    for gpu_id in gpus:
        if gpu_id not in gpus_in_use:
            return gpu_id
    return None

def get_queue_position(session_id):
    """获取队列位置"""
    for i, queued_session in enumerate(session_queue):
        if queued_session['session_id'] == session_id:
            return i + 1
    return None

def update_session_activity(session_id: str):
    """更新会话活动时间"""
    if session_id in active_sessions:
        session_last_activity[session_id] = time.time()

def update_session_ping(session_id: str):
    """更新会话ping时间"""
    if session_id in active_sessions:
        session_last_ping[session_id] = time.time()
        session_last_activity[session_id] = time.time()

# 后台任务
async def process_queue():
    """处理队列的后台任务"""
    while True:
        try:
            async with queue_lock:
                if session_queue and len(session_queue) > 0:
                    free_gpu = find_free_gpu()
                    
                    if free_gpu is not None:
                        next_session = session_queue.pop(0)
                        session_id = next_session['session_id']
                        agent_type = next_session['agent_type']
                        language_pair = next_session['language_pair']
                        latency_multiplier = next_session['latency_multiplier']
                        
                        print(f"Processing queued session {session_id} on mock GPU {free_gpu}")
                        
                        # 创建模拟会话
                        session = MockTranslationSession(agent_type, language_pair, latency_multiplier)
                        
                        active_sessions[session_id] = session
                        session_last_activity[session_id] = time.time()
                        session_last_ping[session_id] = time.time()
                        session_gpu_map[session_id] = free_gpu
                        
                        print(f"Mock session {session_id} created on GPU {free_gpu}")
        except Exception as e:
            print(f"Error processing queue: {e}")
        
        await asyncio.sleep(1)

async def check_orphaned_sessions():
    """检查孤立会话的后台任务"""
    while True:
        current_time = time.time()
        sessions_to_delete = []
        
        for session_id in list(active_sessions.keys()):
            if session_id not in session_last_ping:
                continue
                
            last_ping = session_last_ping[session_id]
            time_since_last_ping = current_time - last_ping
            
            if time_since_last_ping > WEBPAGE_DISCONNECT_TIMEOUT:
                print(f"Session {session_id} detected as orphaned: no ping for {time_since_last_ping:.1f}s")
                sessions_to_delete.append(session_id)
        
        # 删除孤立的会话
        for session_id in sessions_to_delete:
            try:
                if session_id in active_sessions:
                    session = active_sessions[session_id]
                    session.cleanup()
                    
                    del active_sessions[session_id]
                    
                    if session_id in session_last_activity:
                        del session_last_activity[session_id]
                    if session_id in session_last_ping:
                        del session_last_ping[session_id]
                    if session_id in session_gpu_map:
                        gpu_id = session_gpu_map[session_id]
                        del session_gpu_map[session_id]
                        print(f"Released mock GPU {gpu_id} from session {session_id}")
            except Exception as e:
                print(f"Error cleaning up orphaned session {session_id}: {e}")
        
        await asyncio.sleep(DISCONNECT_CHECK_INTERVAL)

async def log_active_sessions():
    """记录活跃会话的后台任务"""
    while True:
        if active_sessions:
            print(f"Active sessions: {len(active_sessions)}")
            print(f"Mock GPU usage: {len(session_gpu_map)}/{len(gpus)} GPUs in use")
        await asyncio.sleep(30)

# 启动事件
@app.on_event("startup")
async def startup_event():
    """启动后台任务"""
    asyncio.create_task(check_orphaned_sessions())
    asyncio.create_task(log_active_sessions())
    asyncio.create_task(process_queue())

# API端点
@app.post("/init")
async def initialize_translation(agent_type: str, language_pair: str, latency_multiplier: int = 2, client_id: str = None):
    """初始化翻译会话"""
    timestamp = int(time.time() * 1000)
    client_suffix = f"_{client_id}" if client_id else f"_{timestamp}"
    session_id = f"{agent_type}_{language_pair}_{len(active_sessions) + len(session_queue)}{client_suffix}"
    
    print(f"Initializing mock session {session_id} with {agent_type} for {language_pair}, latency: {latency_multiplier}x")
    
    free_gpu = find_free_gpu()
    
    if free_gpu is not None:
        # 立即创建会话
        session = MockTranslationSession(agent_type, language_pair, latency_multiplier)
        
        active_sessions[session_id] = session
        session_last_activity[session_id] = time.time()
        session_last_ping[session_id] = time.time()
        session_gpu_map[session_id] = free_gpu
        
        print(f"Mock session {session_id} created on GPU {free_gpu}")
        
        return {"session_id": session_id, "queued": False, "queue_position": 0, "initializing": True}
    else:
        # 添加到队列
        async with queue_lock:
            queue_item = {
                "session_id": session_id,
                "agent_type": agent_type,
                "language_pair": language_pair,
                "latency_multiplier": latency_multiplier,
                "timestamp": time.time()
            }
            session_queue.append(queue_item)
            queue_position = len(session_queue)
            
            print(f"Session {session_id} added to queue at position {queue_position}")
            
            return {"session_id": session_id, "queued": True, "queue_position": queue_position}

@app.get("/queue_status/{session_id}")
async def get_queue_status(session_id: str):
    """获取队列状态"""
    if session_id in active_sessions:
        session = active_sessions[session_id]
        if session.is_ready:
            return {"session_id": session_id, "status": "active", "queued": False, "queue_position": 0}
        else:
            return {"session_id": session_id, "status": "initializing", "queued": False, "queue_position": 0}
    
    queue_position = get_queue_position(session_id)
    if queue_position is not None:
        return {"session_id": session_id, "status": "queued", "queued": True, "queue_position": queue_position}
    
    return {"session_id": session_id, "status": "not_found", "error": "Session not found"}

@app.websocket("/wss/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket端点用于实时翻译"""
    await websocket.accept()
    
    if session_id not in active_sessions:
        await websocket.close(code=4000, reason="Invalid session ID")
        return
        
    session = active_sessions[session_id]
    update_session_activity(session_id)
    update_session_ping(session_id)
    
    # 等待会话准备就绪
    if not session.is_ready:
        await websocket.send_text("INITIALIZING: Mock worker process is starting...")
        if not await session.wait_for_ready(timeout=10):
            await websocket.send_text("ERROR: Mock worker process initialization timeout")
            await websocket.close(code=4001, reason="Worker process initialization timeout")
            return
        await websocket.send_text("READY: Mock worker process is ready")

    try:
        while True:
            try:
                message = await websocket.receive()
                update_session_activity(session_id)
                update_session_ping(session_id)

                if "text" in message:
                    control_message = message["text"]
                    print(f"Received control message: {control_message}")

                    if control_message == "EOF":
                        await websocket.send_text("PROCESSING_COMPLETE: Mock file processing finished")
                        continue
                    elif control_message.startswith("LATENCY:"):
                        try:
                            latency_value = int(control_message.split(":")[1])
                            session.latency_multiplier = latency_value
                            await websocket.send_text(f"LATENCY_UPDATED: Set to {latency_value}")
                        except (ValueError, IndexError):
                            await websocket.send_text("ERROR: Invalid latency format")
                        continue

                elif "bytes" in message:
                    # 模拟处理音频数据
                    data = message["bytes"]
                    print(f"Received mock audio data, size: {len(data)}")
                    
                    # 模拟翻译处理
                    translation = await session.process_segment(data, is_last=False)
                    if translation:
                        await websocket.send_text(translation)
                    
            except starlette.websockets.WebSocketDisconnect:
                print(f"WebSocket disconnected for session {session_id}")
                break
            except Exception as e:
                print(f"Error processing WebSocket message: {e}")
                try:
                    await websocket.send_text(f"ERROR: {str(e)}")
                except:
                    pass
                break

    except Exception as e:
        print(f"Error in WebSocket connection: {e}")
    finally:
        print(f"WebSocket connection closed for session {session_id}")

@app.post("/download_youtube")
async def download_youtube(request: Request, background_tasks: BackgroundTasks):
    """模拟YouTube下载功能"""
    try:
        query_params = dict(request.query_params)
        url = query_params.get("url")
        session_id = query_params.get("session_id")

        if not url:
            return {"error": "Missing URL parameter"}

        print(f"Mock YouTube download for session: {session_id}, URL: {url}")
        
        # 返回一个模拟视频文件路径或者错误
        return {"error": "YouTube download is not available in local test mode. Please use file upload instead."}
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/update_latency")
async def update_latency(session_id: str, latency_multiplier: int):
    """更新延迟倍数"""
    try:
        if session_id not in active_sessions:
            return {"success": False, "error": "Invalid session ID"}
        
        session = active_sessions[session_id]
        
        if not session.is_ready:
            return {"success": False, "error": "Session not ready"}
        
        session.latency_multiplier = latency_multiplier
        update_session_activity(session_id)
        update_session_ping(session_id)
        
        print(f"Updated latency multiplier for session {session_id} to {latency_multiplier}x")
        
        return {"success": True}
    except Exception as e:
        print(f"Error updating latency: {e}")
        return {"success": False, "error": str(e)}

@app.post("/reset_translation")
async def reset_translation(session_id: str):
    """重置翻译状态"""
    try:
        if session_id not in active_sessions:
            return {"success": False, "error": "Invalid session ID"}
        
        session = active_sessions[session_id]
        
        if not session.is_ready:
            return {"success": False, "error": "Session not ready"}
        
        session.reset()
        update_session_activity(session_id)
        update_session_ping(session_id)
        
        print(f"Reset translation state for session {session_id}")
        
        return {"success": True, "message": "Translation state reset successfully"}
    except Exception as e:
        print(f"Error resetting translation: {e}")
        return {"success": False, "error": str(e)}

@app.post("/delete_session")
async def delete_session(request: Request, session_id: Optional[str] = None):
    """删除会话"""
    try:
        if session_id is None:
            form_data = await request.form()
            session_id = form_data.get("session_id")
            
            if session_id is None:
                return {"success": False, "error": "No session_id provided"}
        
        if session_id not in active_sessions:
            return {"success": False, "error": "Invalid session ID"}
        
        session = active_sessions[session_id]
        session.cleanup()
        
        del active_sessions[session_id]
        
        if session_id in session_last_activity:
            del session_last_activity[session_id]
        if session_id in session_last_ping:
            del session_last_ping[session_id]
        if session_id in session_gpu_map:
            gpu_id = session_gpu_map[session_id]
            del session_gpu_map[session_id]
            print(f"Released mock GPU {gpu_id} from session {session_id}")
        
        print(f"Session {session_id} deleted, {len(active_sessions)} active sessions remaining")
        
        return {"success": True}
    except Exception as e:
        print(f"Error deleting session: {e}")
        return {"success": False, "error": str(e)}

@app.post("/ping")
async def ping_session(session_id: str):
    """更新会话ping时间"""
    try:
        if session_id not in active_sessions:
            return {"success": False, "error": "Invalid session ID"}
            
        update_session_ping(session_id)
        
        return {"success": True}
    except Exception as e:
        print(f"Error updating ping: {e}")
        return {"success": False, "error": str(e)}

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")
# 注意：不要挂载根路径，这会覆盖API端点
# app.mount("/", StaticFiles(directory="static", html=True), name="static_root")

# 明确的根路径处理
@app.get("/")
async def read_index():
    """返回index.html"""
    return FileResponse('static/index.html')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InfiniSST Local API Server (No GPU Required)")
    
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                       help="Host to bind the server to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001, 
                       help="Port to bind the server to (default: 8001)")
    parser.add_argument("--reload", action="store_true", 
                       help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print(f"Starting InfiniSST Local API Server...")
    print(f"🚀 Server will be available at http://{args.host}:{args.port}")
    print(f"📝 This is a mock server for frontend testing")
    print(f"🔧 No GPU or model inference required")

    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        reload=args.reload if hasattr(args, 'reload') else False
    ) 