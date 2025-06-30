import multiprocessing as mp
# Set the start method for multiprocessing to 'spawn' for better compatibility across platforms
# This is especially important on macOS where 'fork' can cause issues with multithreading
# Do this at the very beginning before any other imports that might use multiprocessing
try:
    mp.set_start_method('spawn')
except RuntimeError:
    # If the context has already been set, just use the current context
    print("Multiprocessing start method already set, using current context")

from fastapi import FastAPI, WebSocket, UploadFile, File, BackgroundTasks, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import tempfile
import os
import yt_dlp
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import soundfile as sf
import numpy as np
import json
import asyncio
import argparse
import copy
import time
from multiprocessing import Process, Queue, Manager
from queue import Empty
from typing import Dict, Optional, Any, Tuple
from agents.infinisst import InfiniSST
from agents.streamatt import StreamAtt
import io
import uvicorn
import gc
import torch
import starlette.websockets

# 导入我们的 scheduler 和 inference engine
try:
    from serve.scheduler import LLMScheduler, RequestStage, InferenceRequest, UserSession
    from serve.inference_engine import MultiGPUInferenceEngine, EngineConfig
    SCHEDULER_AVAILABLE = True
    print("✅ Scheduler 和 Inference Engine 可用")
except ImportError as e:
    print(f"⚠️ Scheduler 不可用: {e}")
    SCHEDULER_AVAILABLE = False

# 支持的翻译模型列表
TRANSLATION_AGENTS = {
    "InfiniSST": InfiniSST,
    # 暂时禁用StreamAtt
    # "StreamAtt": StreamAtt,
}

# 支持的语言方向
LANGUAGE_PAIRS = {
    "English -> Chinese": ("English", "Chinese", "en", "zh"),
    "English -> Italian": ("English", "Italian", "en", "it"),
    "English -> German": ("English", "German", "en", "de"),
    "English -> Spanish": ("English", "Spanish", "en", "es"),
}

model_path_de = "/mnt/aries/data6/xixu/demo/en-de/pytorch_model.bin"
model_path_es = "/mnt/aries/data6/xixu/demo/en-es/pytorch_model.bin"
model_path = "/mnt/aries/data6/jiaxuanluo/demo/{}-{}/pytorch_model.bin"
lora_path = "/mnt/aries/data6/jiaxuanluo/demo/{}-{}/lora.bin"

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加全局异常处理中间件
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to ensure all API responses are JSON"""
    print(f"Global exception handler caught: {exc}")
    import traceback
    traceback.print_exc()
    
    # 对于API请求，返回JSON错误响应
    if request.url.path.startswith("/ping") or request.url.path.startswith("/queue_status") or request.url.path.startswith("/init"):
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Internal server error: {str(exc)}"}
        )
    
    # 对于其他请求，抛出HTTP异常
    raise HTTPException(status_code=500, detail=str(exc))

# Store active translation sessions with last activity timestamp
active_sessions: Dict[str, dict] = {}
session_last_activity: Dict[str, float] = {}
# Track the last ping time for each session to detect closed/refreshed webpages
session_last_ping: Dict[str, float] = {}

# Queue for pending session initialization requests
session_queue: list = []
# Dictionary to track which GPU each session is using
session_gpu_map: Dict[str, int] = {}
# Lock for queue operations
queue_lock = asyncio.Lock()

# Dictionary to store worker processes and communication queues
session_workers: Dict[str, Dict[str, Any]] = {}

# Get the number of available GPUs
#num_gpus = torch.cuda.device_count()
gpus = [int(x.strip()) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip().isdigit()]

print(f"Number of available GPUs: gpus={gpus}, len(gpus)={len(gpus)}")

# 全局 scheduler 和 inference engine
global_scheduler: Optional[LLMScheduler] = None
global_inference_engine: Optional[MultiGPUInferenceEngine] = None

# Short timeout for detecting browser disconnections
DISCONNECT_CHECK_INTERVAL = 5  # Check every 5 seconds
# Timeout for detecting closed/refreshed webpages (15 seconds without a ping)
WEBPAGE_DISCONNECT_TIMEOUT = 300  # Consider a webpage closed if no ping for 5 minutes (increased from 60s to handle 503 errors)
# DISCONNECT_TIMEOUT is no longer used since orphaned sessions are now tracked client-side

# Worker process function that runs the translation model
def session_worker_process(
    agent_type: str, 
    language_pair: str, 
    args_dict: dict, 
    gpu_id: int,
    input_queue: Queue, 
    output_queue: Queue, 
    control_queue: Queue,
    ready_event: mp.Event
):
    """
    Worker process function that runs a translation model in a separate process.
    
    Args:
        agent_type: Type of translation agent to use
        language_pair: Language pair for translation
        args_dict: Arguments for the translation agent
        gpu_id: GPU ID to use for this worker
        input_queue: Queue for receiving audio segments
        output_queue: Queue for sending translation results
        control_queue: Queue for receiving control commands
        ready_event: Event to signal when the worker is ready
    """
    try:
        # Set process name for better debugging
        import setproctitle
        setproctitle.setproctitle(f"sllama_worker_{agent_type}_{language_pair}_gpu{gpu_id}")
        
        # Convert args_dict back to an argparse.Namespace
        args = argparse.Namespace(**args_dict)
        
        # Parse language pair
        source_lang, target_lang, src_code, tgt_code = LANGUAGE_PAIRS[language_pair]
        
        args.source_lang = source_lang
        args.target_lang = target_lang
        # Conditional model and lora loading
        if language_pair == "English -> German":
            args.state_dict_path = model_path_de
            args.lora_path = None  # or '' if preferred
        elif language_pair == "English -> Spanish":
            args.state_dict_path = model_path_es
            args.lora_path = None  # or '' if preferred
        else:
            args.state_dict_path = model_path.format(src_code, tgt_code) if '{}' in model_path else model_path
            args.lora_path = lora_path.format(src_code, tgt_code) if '{}' in lora_path else lora_path

        # Set the GPU device
        print(f"Worker process initializing on GPU {gpu_id}")
        with torch.cuda.device(get_logical_index_from_physical_id(gpu_id)):
            # Initialize the agent
            agent = TRANSLATION_AGENTS[agent_type](args)
            agent.update_multiplier(args.latency_multiplier)
            states = agent.build_states()
            states.reset()
            
            # Signal that the worker is ready
            ready_event.set()
            
            # Process commands from the control queue and audio segments from the input queue
            while True:
                # Check for control commands (non-blocking)
                try:
                    cmd = control_queue.get_nowait()
                    if cmd == "reset":
                        # Reset translation state
                        if hasattr(states, 'reset'):
                            states.reset()
                        print(f"Reset translation state for worker with {agent_type} model on GPU {gpu_id}")
                    elif cmd == "terminate":
                        # Clean up and exit
                        print(f"Terminating worker process for {agent_type} model on GPU {gpu_id}")
                        break
                    elif cmd.startswith("update_latency:"):
                        # Update latency multiplier
                        try:
                            # Extract the latency multiplier from the command
                            latency_multiplier = int(cmd.split(":")[1])
                            # Update the agent's latency multiplier
                            agent.update_multiplier(latency_multiplier)
                            # Update args for future reference
                            args.latency_multiplier = latency_multiplier
                            args.max_new_tokens = 10 * latency_multiplier
                            print(f"Updated latency multiplier to {latency_multiplier}x in worker process on GPU {gpu_id}")
                        except (ValueError, IndexError) as e:
                            print(f"Error parsing latency multiplier on GPU {gpu_id}: {e}")
                except Empty:
                    pass
                except Exception as e:
                    print(f"Error processing control command on GPU {gpu_id}: {e}")
                
                # Process audio segments (blocking with timeout)
                try:
                    segment_data = input_queue.get(timeout=0.1)
                    segment, is_last = segment_data
                    
                    # Process the segment
                    states.source.extend(segment)
                    print(f"Worker on GPU {gpu_id} processing segment, total audio length: {len(states.source) / 16000}s")
                    
                    if is_last:
                        states.source_finished = True
                    
                    action = agent.policy(states)
                    if not action.is_read():
                        output = action.content
                        states.target.append(output)
                        translation = ' '.join(states.target) if args.target_lang != 'Chinese' else ''.join(states.target)
                        output_queue.put(translation)
                except Empty:
                    # No audio segment available, continue checking for control commands
                    pass
                except Exception as e:
                    print(f"Error processing audio segment on GPU {gpu_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Send error message to main process
                    output_queue.put(("ERROR", f"Error processing audio: {str(e)}"))
            
            # Clean up GPU resources
            if hasattr(agent, 'model'):
                if hasattr(agent.model, 'to'):
                    agent.model.to('cpu')  # Move model to CPU first
                
                # Delete model attributes that might hold GPU tensors
                for attr_name in dir(agent.model):
                    if not attr_name.startswith('__'):
                        attr = getattr(agent.model, attr_name)
                        if isinstance(attr, torch.Tensor) and attr.is_cuda:
                            delattr(agent.model, attr_name)
            
            # Clear states
            if hasattr(states, 'clear'):
                states.clear()
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print(f"Worker process for {agent_type} model on GPU {gpu_id} terminated")
    
    except Exception as e:
        import traceback
        print(f"Error in worker process on GPU {gpu_id}: {e}")
        traceback.print_exc()
        # Signal error to main process
        output_queue.put(("ERROR", str(e)))
    finally:
        # Ensure queues are properly closed
        try:
            input_queue.close()
            output_queue.close()
            control_queue.close()
        except Exception as e:
            print(f"Error closing queues on GPU {gpu_id}: {e}")

class TranslationSession:
    def __init__(self, agent_type: str, language_pair: str, args, gpu_id=None):
        self.agent_type = agent_type
        self.language_pair = language_pair
        self.args = copy.deepcopy(args)  # Store args in the session
        self.gpu_id = gpu_id
        self.is_ready = False
        
        # Create queues for communication with the worker process
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.control_queue = Queue()
        self.ready_event = mp.Event()
        
        # Convert args to a dictionary for passing to the worker process
        args_dict = vars(self.args)
        
        # Start the worker process
        self.process = Process(
            target=session_worker_process,
            args=(
                agent_type,
                language_pair,
                args_dict,
                gpu_id,
                self.input_queue,
                self.output_queue,
                self.control_queue,
                self.ready_event
            )
        )
        self.process.daemon = True  # Ensure process terminates when main process exits
        self.process.start()
        
        # Store the process and queues in the session_workers dictionary
        session_workers[id(self)] = {
            "process": self.process,
            "input_queue": self.input_queue,
            "output_queue": self.output_queue,
            "control_queue": self.control_queue,
            "ready_event": self.ready_event,
            "gpu_id": gpu_id
        }
        
        print(f"Worker process started on GPU {gpu_id}, waiting for initialization...")
        
    async def wait_for_ready(self, timeout=60):
        """异步等待工作进程准备就绪"""
        start_time = time.time()
        while not self.ready_event.is_set() and time.time() - start_time < timeout:
            # 使用asyncio.sleep让出控制权，允许其他任务执行
            await asyncio.sleep(0.1)
            
        if self.ready_event.is_set():
            self.is_ready = True
            print(f"Worker process ready on GPU {self.gpu_id} after {time.time() - start_time:.1f}s")
            return True
        else:
            print(f"Timeout waiting for worker process on GPU {self.gpu_id} after {timeout}s")
            return False

    # sends audio segment to worker process
    async def process_segment(self, segment: np.ndarray, is_last: bool = False) -> str:
        # 确保工作进程已准备就绪
        if not self.is_ready:
            print(f"Warning: Trying to process segment before worker is ready on GPU {self.gpu_id}")
            if not await self.wait_for_ready(timeout=10):
                return "ERROR: Worker process not ready"

        # Safety check: ensure process is alive before putting into queue
        if not self.process.is_alive():
            print(f"Worker process is not alive for session on GPU {self.gpu_id}")
            return "ERROR: Worker process terminated"

        # Send the segment to the worker process
        try:
            self.input_queue.put((segment, is_last))
        except ValueError as e:
            print(f"Failed to put segment into input_queue: {e}")
            return "ERROR: Input queue is closed"

        # Check for output (non-blocking)
        try:
            # Use asyncio to avoid blocking the event loop
            translation = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.output_queue.get(block=False) if not self.output_queue.empty() else ""
            )
            return translation
        except Empty:
            return ""
    
    def reset(self):
        """Reset the translation state without reloading the model"""
        # 确保工作进程已准备就绪
        if not self.is_ready:
            print(f"Warning: Trying to reset before worker is ready on GPU {self.gpu_id}")
            return False
            
        self.control_queue.put("reset")
        print(f"Sent reset command to worker process for session with {self.agent_type} model")
        return True
    
    def cleanup(self):
        """Clean up GPU resources used by this session"""
        try:
            # Send termination command to the worker process
            self.control_queue.put("terminate")
            
            # Wait for the process to terminate (with timeout)
            self.process.join(timeout=5)
            
            # If the process is still alive, terminate it forcefully
            if self.process.is_alive():
                print(f"Worker process did not terminate gracefully, forcing termination")
                self.process.terminate()
                self.process.join(timeout=2)
                
                # If still alive, kill it
                if self.process.is_alive():
                    print(f"Worker process still alive after terminate, killing it")
                    self.process.kill()
                    self.process.join(timeout=1)
            
            # Close the queues
            self.input_queue.close()
            self.output_queue.close()
            self.control_queue.close()
            
            # Remove from session_workers dictionary
            if id(self) in session_workers:
                del session_workers[id(self)]
            
            print(f"Cleaned up worker process for session with {self.agent_type} model")
        except Exception as e:
            print(f"Error cleaning up worker process: {e}")
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Function to find a free GPU
def find_free_gpu():
    # Get list of GPUs currently in use
    gpus_in_use = set(session_gpu_map.values())
    
    # Find a free GPU
    for gpu_id in gpus:
        if gpu_id not in gpus_in_use:
            return gpu_id
    
    # No free GPU found
    return None

# Function to get queue position for a session
def get_queue_position(session_id):
    for i, queued_session in enumerate(session_queue):
        if queued_session['session_id'] == session_id:
            return i + 1
    return None

# Background task to process the queue
async def process_queue():
    """Background task to process the queue of pending session initialization requests"""
    while True:
        try:
            async with queue_lock:
                # Check if there are any sessions in the queue
                if session_queue and len(session_queue) > 0:
                    # Find a free GPU
                    free_gpu = find_free_gpu()
                    
                    if free_gpu is not None:
                        # Get the first session in the queue
                        next_session = session_queue.pop(0)
                        session_id = next_session['session_id']
                        agent_type = next_session['agent_type']
                        language_pair = next_session['language_pair']
                        latency_multiplier = next_session['latency_multiplier']
                        
                        print(f"Processing queued session {session_id} on GPU {free_gpu}")
                        
                        try:
                            # Initialize the session on the free GPU
                            session_args = copy.deepcopy(args)
                            session_args.latency_multiplier = latency_multiplier
                            session_args.max_new_tokens = 10 * latency_multiplier
                            
                            # Create the session with the specified GPU
                            print(f"Creating queued session {session_id} on GPU {free_gpu}")
                            session = TranslationSession(agent_type, language_pair, session_args, gpu_id=free_gpu)
                            
                            # Add the session to active sessions immediately, but mark it as initializing
                            active_sessions[session_id] = session
                            session_last_activity[session_id] = time.time()
                            session_last_ping[session_id] = time.time()
                            
                            # Map the session to the GPU
                            session_gpu_map[session_id] = free_gpu
                            
                            # 异步等待工作进程准备就绪，但不阻塞队列处理
                            # 创建一个后台任务来等待工作进程准备就绪
                            asyncio.create_task(session.wait_for_ready())
                            
                            print(f"Queued session {session_id} initialization started on GPU {free_gpu}")
                        except Exception as e:
                            print(f"Error initializing queued session {session_id} on GPU {free_gpu}: {e}")
                            import traceback
                            traceback.print_exc()
                            
                            # Put the session back in the queue if initialization failed
                            session_queue.insert(0, next_session)
                            print(f"Session {session_id} put back in queue due to initialization failure")
        except Exception as e:
            print(f"Error processing queue: {e}")
            import traceback
            traceback.print_exc()
        
        # Check every 1 second
        await asyncio.sleep(1)

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
                    
                    # 检查是否是基于调度器的会话
                    is_scheduler_based = isinstance(session, dict) and session.get('is_scheduler_based', False)
                    
                    if is_scheduler_based:
                        print(f"Deleting scheduler-based session {session_id}")
                        
                        # 🔥 关键：调用调度器的会话清理功能
                        if global_scheduler:
                            try:
                                user_id = session.get('user_id', session_id)
                                language_pair = session.get('language_pair', 'English -> Chinese')
                                
                                cleanup_success = global_scheduler.cleanup_session(user_id, language_pair)
                                if cleanup_success:
                                    print(f"✅ 调度器会话 {session_id} 清理成功，KV cache页面已释放")
                                else:
                                    print(f"⚠️ 调度器会话 {session_id} 清理失败或会话不存在")
                                    
                            except Exception as e:
                                print(f"❌ 调度器会话清理出错: {e}")
                        else:
                            print(f"⚠️ 全局调度器不可用，无法清理会话KV cache")
                    else:
                        print(f"Deleting traditional session {session_id}")
                        # 传统会话需要清理GPU资源
                        if hasattr(session, 'cleanup'):
                            session.cleanup()
                    
                    # Remove from active sessions
                    del active_sessions[session_id]
                    
                    # Remove from activity tracking
                    if session_id in session_last_activity:
                        del session_last_activity[session_id]
                        
                    # Remove from ping tracking
                    if session_id in session_last_ping:
                        del session_last_ping[session_id]
                    
                    # Remove from GPU mapping (仅对传统会话)
                    if session_id in session_gpu_map:
                        gpu_id = session_gpu_map[session_id]
                        del session_gpu_map[session_id]
                        print(f"Released GPU {gpu_id} from session {session_id}")
                    
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
            print(f"GPU Usage: {len(session_gpu_map)}/{len(gpus)} GPUs in use")
            
            # Print GPU allocation
            gpu_allocation = {}
            for gpu_id in gpus:
                gpu_allocation[gpu_id] = []
            
            for session_id, gpu_id in session_gpu_map.items():
                if gpu_id in gpu_allocation:
                    gpu_allocation[gpu_id].append(session_id)
            
            for gpu_id, sessions in gpu_allocation.items():
                if sessions:
                    print(f"  GPU {gpu_id}: {len(sessions)} sessions - {', '.join(sessions)}")
                else:
                    print(f"  GPU {gpu_id}: Free")
            
            # Print active sessions
            print("\nActive Sessions:")
            for session_id, session in active_sessions.items():
                last_activity = session_last_activity.get(session_id, current_time)
                inactivity_time = current_time - last_activity
                
                # 区分调度器会话和传统会话
                is_scheduler_based = isinstance(session, dict) and session.get('is_scheduler_based', False)
                
                if is_scheduler_based:
                    # 调度器会话
                    agent_type = session.get('agent_type', 'Unknown')
                    language_pair = session.get('language_pair', 'Unknown')
                    latency_multiplier = session.get('latency_multiplier', 'Unknown')
                    gpu_id = "Scheduler"
                    process_id = "Scheduler"
                    
                    print(f"  - {session_id}: {agent_type} | {language_pair} | "
                          f"Latency: {latency_multiplier}x | GPU: {gpu_id} | "
                          f"Process: {process_id} | Inactive for: {inactivity_time:.1f}s | Type: Scheduler")
                else:
                    # 传统会话
                    gpu_id = session_gpu_map.get(session_id, "Unknown")
                    process_id = session.process.pid if hasattr(session, 'process') else "Unknown"
                    
                    print(f"  - {session_id}: {session.agent_type} | {session.language_pair} | "
                          f"Latency: {session.args.latency_multiplier}x | GPU: {gpu_id} | "
                          f"Process: {process_id} | Inactive for: {inactivity_time:.1f}s | Type: Traditional")
            
            # Print queue information
            if session_queue:
                print(f"\nQueue: {len(session_queue)} sessions waiting")
                for i, queued_session in enumerate(session_queue):
                    wait_time = current_time - queued_session['timestamp']
                    print(f"  {i+1}. {queued_session['session_id']}: {queued_session['agent_type']} | "
                          f"{queued_session['language_pair']} | Waiting for: {wait_time:.1f}s")
            
            print("=============================================\n")
        
        # Log every 30 seconds
        await asyncio.sleep(30)

def get_logical_index_from_physical_id(physical_id: int) -> int:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    mapping = [int(x.strip()) for x in visible_devices.split(",") if x.strip().isdigit()]
    if physical_id in mapping:
        return mapping.index(physical_id)
    else:
        raise ValueError(f"GPU ID {physical_id} not in CUDA_VISIBLE_DEVICES: {mapping}")

@app.on_event("startup")
async def startup_event():
    """Start background tasks when the application starts"""
    global global_scheduler, global_inference_engine
    
    # 初始化 scheduler 和 inference engine（如果可用）
    if SCHEDULER_AVAILABLE and len(gpus) > 0:
        try:
            print("🚀 初始化集成调度系统...")
            
            # 创建 GPU 语言映射 - 使用逻辑GPU ID
            # 将物理GPU ID转换为逻辑GPU ID
            logical_gpu_ids = []
            for physical_gpu_id in gpus[:1]:  # 只使用第一个GPU
                try:
                    logical_id = get_logical_index_from_physical_id(physical_gpu_id)
                    logical_gpu_ids.append(logical_id)
                except ValueError as e:
                    print(f"⚠️ GPU ID转换失败: {e}")
                    logical_gpu_ids.append(0)  # 默认使用逻辑GPU 0
            
            gpu_language_map = {logical_id: "English -> Chinese" for logical_id in logical_gpu_ids}
            print(f"GPU语言映射 (物理->逻辑): {gpus[:1]} -> {logical_gpu_ids}")
            print(f"GPU语言映射: {gpu_language_map}")
            
            # 创建推理引擎
            model_args_map = {gpu_id: {} for gpu_id in gpu_language_map.keys()}
            global_inference_engine = MultiGPUInferenceEngine(
                gpu_language_map=gpu_language_map,
                model_args_map=model_args_map
            )
            
            # 🔥 重要：尝试加载模型到推理引擎
            print("📥 开始加载模型到推理引擎...")
            model_load_success = global_inference_engine.load_all_models()
            if model_load_success:
                print("✅ 推理引擎模型加载成功")
                # 启动推理引擎
                global_inference_engine.start_all()
                print("✅ 推理引擎已启动")
            
            # 创建调度器
            class Args:
                def __init__(self):
                    self.max_batch_size = 32  #jiaxuanluo
                    self.batch_timeout = 0.1
                    self.session_timeout = 3600
            
            args_obj = Args()
            global_scheduler = LLMScheduler(gpu_language_map, args_obj)
            
            # 连接推理引擎到调度器
            global_scheduler.set_inference_engine(global_inference_engine)
            
            # 启动调度器
            global_scheduler.start()
            
            print("✅ 集成调度系统初始化完成")
            print(f"   - 调度器运行状态: {global_scheduler.is_running}")
            print(f"   - 支持的语言: {global_scheduler.get_supported_languages()}")
            print(f"   - 推理引擎状态: {len(global_inference_engine.engines)} 个引擎")
            
        except Exception as e:
            print(f"❌ 调度系统初始化失败: {e}")
            import traceback
            traceback.print_exc()
            global_scheduler = None
            global_inference_engine = None
    else:
        print("⚠️ 跳过调度系统初始化（Scheduler不可用或无GPU）")
        print(f"   - SCHEDULER_AVAILABLE: {SCHEDULER_AVAILABLE}")
        print(f"   - 可用GPU数量: {len(gpus)}")
    
    # 启动原有的后台任务
    asyncio.create_task(check_orphaned_sessions())
    asyncio.create_task(log_active_sessions())
    asyncio.create_task(process_queue())

@app.post("/init")
async def initialize_translation(agent_type: str, language_pair: str, latency_multiplier: int = 2, client_id: str = None):
    global args, global_scheduler
    
    # Generate a unique session ID that includes the client ID to ensure different browser tabs have independent sessions
    timestamp = int(time.time() * 1000)  # Use timestamp for uniqueness
    client_suffix = f"_{client_id}" if client_id else f"_{timestamp}"
    session_id = f"{agent_type}_{language_pair}_{len(active_sessions) + len(session_queue)}{client_suffix}"
    
    print(f"Initializing new session {session_id} with {agent_type} model for {language_pair}, latency: {latency_multiplier}x")
    
    # 优先使用调度器系统（如果可用）
    if global_scheduler and SCHEDULER_AVAILABLE:
        try:
            print(f"🚀 使用调度器系统创建会话 {session_id}")
            
            # 创建基于调度器的会话
            scheduler_session = {
                'session_id': session_id,
                'agent_type': agent_type,
                'language_pair': language_pair,
                'latency_multiplier': latency_multiplier,
                'user_id': client_id or session_id,
                'created_at': time.time(),
                'is_scheduler_based': True,  # 标记这是基于调度器的会话
                'pending_results': {},  # 存储异步结果
                'result_callback_map': {}  # 结果回调映射
            }
            
            # 添加到活跃会话
            active_sessions[session_id] = scheduler_session
            session_last_activity[session_id] = time.time()
            session_last_ping[session_id] = time.time()
            
            print(f"✅ 调度器会话 {session_id} 创建成功")
            return {"session_id": session_id, "queued": False, "queue_position": 0, "scheduler_based": True}
            
        except Exception as e:
            print(f"❌ 调度器会话创建失败: {e}")
            # 如果调度器失败，回退到原始系统
            pass
    
    # 回退到原始的TranslationSession系统
    print(f"🔄 回退到原始TranslationSession系统 {session_id}")
    
    # Check if there's a free GPU
    free_gpu = find_free_gpu()
    
    if free_gpu is not None:
        # Initialize the session immediately on the free GPU
        session_args = copy.deepcopy(args)
        session_args.latency_multiplier = latency_multiplier
        session_args.max_new_tokens = 10 * latency_multiplier
        
        try:
            # Create the session with the specified GPU
            print(f"Creating session {session_id} on GPU {free_gpu}")
            session = TranslationSession(agent_type, language_pair, session_args, gpu_id=free_gpu)
            
            # Add the session to active sessions immediately, but mark it as initializing
            active_sessions[session_id] = session
            session_last_activity[session_id] = time.time()
            session_last_ping[session_id] = time.time()
            
            # Map the session to the GPU
            session_gpu_map[session_id] = free_gpu
            
            # 异步等待工作进程准备就绪，但不阻塞API响应
            # 创建一个后台任务来等待工作进程准备就绪
            asyncio.create_task(session.wait_for_ready())
            
            print(f"Session {session_id} initialization started on GPU {free_gpu}")
            
            return {"session_id": session_id, "queued": False, "queue_position": 0, "initializing": True, "scheduler_based": False}
        except Exception as e:
            print(f"Error initializing session {session_id} on GPU {free_gpu}: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Failed to initialize session: {str(e)}", "queued": False, "queue_position": 0}
    else:
        # No free GPU, add to queue
        try:
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
                
                print(f"Session {session_id} added to queue at position {queue_position} (no free GPUs available)")
                
                return {"session_id": session_id, "queued": True, "queue_position": queue_position, "scheduler_based": False}
        except Exception as e:
            print(f"Error adding session {session_id} to queue: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Failed to queue session: {str(e)}", "queued": False, "queue_position": 0}

@app.get("/queue_status/{session_id}")
async def get_queue_status(session_id: str):
    """Get the current status of a queued session"""
    # Check if the session is already active
    if session_id in active_sessions:
        session = active_sessions[session_id]
        # 检查会话是否已准备就绪
        if session.is_ready:
            return {"session_id": session_id, "status": "active", "queued": False, "queue_position": 0}
        else:
            return {"session_id": session_id, "status": "initializing", "queued": False, "queue_position": 0}
    
    # Check if the session is in the queue
    queue_position = get_queue_position(session_id)
    if queue_position is not None:
        return {"session_id": session_id, "status": "queued", "queued": True, "queue_position": queue_position}
    
    # Session not found
    return {"session_id": session_id, "status": "not_found", "error": "Session not found in queue or active sessions"}

# receives audio segment from webpage
@app.websocket("/wss/{session_id}")
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
    
    # 检查是否是基于调度器的会话
    is_scheduler_based = isinstance(session, dict) and session.get('is_scheduler_based', False)
    
    if is_scheduler_based:
        print(f"🚀 WebSocket 连接到调度器会话 {session_id}")
        await _handle_scheduler_websocket(websocket, session_id, session)
    else:
        print(f"🔄 WebSocket 连接到传统会话 {session_id}")
        await _handle_traditional_websocket(websocket, session_id, session)

async def _handle_scheduler_websocket(websocket: WebSocket, session_id: str, session: dict):
    """处理基于调度器的WebSocket连接"""
    global global_scheduler
    
    if not global_scheduler:
        await websocket.send_text("ERROR: Scheduler not available")
        await websocket.close(code=4002, reason="Scheduler not available")
        return
    
    await websocket.send_text("READY: Scheduler system ready")
    
    chunk_count = 0
    # 创建结果队列用于异步处理
    result_queue = asyncio.Queue()
    
    # 获取当前事件循环（用于线程安全的队列操作）
    loop = asyncio.get_event_loop()
    
    # 后台任务：检查和发送结果
    async def result_sender():
        while True:
            try:
                # 等待结果（带超时）
                result_text = await asyncio.wait_for(result_queue.get(), timeout=0.1)
                await websocket.send_text(result_text)
                print(f"📤 发送调度器结果到 {session_id}: {result_text}")
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error sending result to {session_id}: {e}")
                break
    
    # 启动结果发送任务
    sender_task = asyncio.create_task(result_sender())
    
    try:
        while True:
            try:
                # Receive data from the WebSocket
                message = await websocket.receive()
                
                # Update activity timestamp
                update_session_activity(session_id)
                update_session_ping(session_id)
                
                # Check if this is a control message (text) or audio data (bytes)
                if "text" in message:
                    control_message = message["text"]
                    print(f"Received control message for scheduler session {session_id}: {control_message}")
                    
                    if control_message == "EOF":
                        print(f"Received EOF signal for scheduler session {session_id}")
                        await websocket.send_text("PROCESSING_COMPLETE: File processing finished")
                        continue
                        
                elif "bytes" in message:
                    # This is audio data
                    data = message["bytes"]
                    
                    # 🔍 调试：检查接收到的原始音频数据
                    print(f"🎤 [DEBUG] WebSocket received audio data:")
                    print(f"   - Raw bytes length: {len(data)}")
                    print(f"   - First 10 bytes: {data[:10] if len(data) >= 10 else data}")
                    
                    # Convert bytes to numpy array
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    chunk_count += 1
                    
                    # 🔍 调试：检查转换后的音频数据
                    print(f"   - Converted to numpy: shape={audio_data.shape}, dtype={audio_data.dtype}")
                    print(f"   - Audio samples: min={audio_data.min():.6f}, max={audio_data.max():.6f}, mean={audio_data.mean():.6f}")
                    print(f"   - Chunk {chunk_count}, size: {len(audio_data)}")
                    
                    # 🔍 检查是否全为零
                    non_zero_count = np.count_nonzero(audio_data)
                    print(f"   - Non-zero samples: {non_zero_count}/{len(audio_data)} ({100*non_zero_count/len(audio_data):.1f}%)")
                    
                    if len(audio_data) == 0:
                        print(f"⚠️  [WARNING] Received empty audio data in chunk {chunk_count}")
                        continue
                    
                    if non_zero_count == 0:
                        print(f"⚠️  [WARNING] Received all-zero audio data in chunk {chunk_count}")
                    
                    # 提交请求到调度器
                    try:
                        user_id = session['user_id']
                        language_pair = session['language_pair']
                        
                        print(f"📤 [DEBUG] Submitting to scheduler:")
                        print(f"   - User ID: {user_id}")
                        print(f"   - Language: {language_pair}")
                        print(f"   - Audio shape: {audio_data.shape}")
                        
                        # 创建结果回调函数（使用线程安全的队列）
                        def result_callback(result):
                            """处理调度器返回的结果"""
                            try:
                                if result.get('success', False):
                                    # 🔥 修复：优先使用完整翻译历史，然后fallback到generated_text
                                    full_translation = result.get('full_translation', '')
                                    generated_text = result.get('generated_text', '')
                                    
                                    # 决定发送什么内容到前端
                                    text_to_send = full_translation if full_translation else generated_text
                                    
                                    if text_to_send:
                                        # 使用线程安全的方式添加结果到队列
                                        loop.call_soon_threadsafe(result_queue.put_nowait, text_to_send)
                                        
                                        # 🔥 增强调试信息
                                        if full_translation:
                                            print(f"📥 调度器完整翻译入队 {session_id}: {text_to_send}")
                                            print(f"   - 段落数: {result.get('segment_count', 'unknown')}")
                                            new_segment = result.get('new_segment', '')
                                            if new_segment:
                                                print(f"   - 新增内容: '{new_segment}'")
                                        else:
                                            print(f"📥 调度器单次结果入队 {session_id}: {text_to_send}")
                                else:
                                    error_msg = result.get('error', 'Unknown error')
                                    loop.call_soon_threadsafe(result_queue.put_nowait, f"ERROR: {error_msg}")
                                    print(f"📥 调度器错误入队 {session_id}: {error_msg}")
                            except Exception as e:
                                print(f"Error in result callback for {session_id}: {e}")
                                # 尝试发送错误信息
                                try:
                                    loop.call_soon_threadsafe(result_queue.put_nowait, f"ERROR: Callback failed - {str(e)}")
                                except:
                                    pass
                        
                        from serve.scheduler import RequestStage

                        # 初始化计数器
                        if "request_count" not in session:
                            session["request_count"] = 0
                        if "request_count_by_stage" not in session:
                            session["request_count_by_stage"] = {}

                        # 更新计数
                        session["request_count"] += 1
                        stage_str = RequestStage.PREFILL.name  # 或者 decode 阶段写成 RequestStage.DECODE.name
                        session["request_count_by_stage"][stage_str] = session["request_count_by_stage"].get(stage_str, 0) + 1

                        # 提交请求
                        request_id = global_scheduler.submit_request(
                            user_id=user_id,
                            language_id=language_pair,
                            speech_data=audio_data,
                            stage=RequestStage.PREFILL,
                            is_final=False,
                            max_new_tokens=session.get('latency_multiplier', 2) * 10,
                            result_callback=result_callback
                        )

                        # 打印结构化日志
                        print(f"[{time.strftime('%H:%M:%S')}] [Session: {session_id}] [Stage: {stage_str}] ✅ 提交请求 {request_id}，累计 {session['request_count']} 次（本阶段: {session['request_count_by_stage'][stage_str]}）")
                        
                    except Exception as e:
                        print(f"❌ 提交调度器请求失败 {session_id}: {e}")
                        import traceback
                        traceback.print_exc()
                        await websocket.send_text(f"ERROR: {str(e)}")
                        
            except starlette.websockets.WebSocketDisconnect:
                print(f"WebSocket disconnected for scheduler session {session_id}")
                break
            except Exception as e:
                print(f"Error in scheduler WebSocket for session {session_id}: {e}")
                await websocket.send_text(f"ERROR: {str(e)}")
                break
                
    except Exception as e:
        print(f"Fatal error in scheduler WebSocket for session {session_id}: {e}")
    finally:
        # 清理任务
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass
        print(f"Scheduler WebSocket connection closed for session {session_id}")

async def _handle_traditional_websocket(websocket: WebSocket, session_id: str, session):
    """处理传统TranslationSession的WebSocket连接"""
    
    # 确保工作进程已准备就绪
    if not session.is_ready:
        print(f"WebSocket connected for session {session_id}, waiting for worker process to be ready...")
        if websocket.client_state.name == "CONNECTED":
            await websocket.send_text("INITIALIZING: Worker process is starting, please wait...")
        else:
            print(f"Client disconnected before INITIALIZING message for session {session_id}")
            return
        # 等待工作进程准备就绪，最多等待180秒
        if not await session.wait_for_ready(timeout=180):
            # Guard: avoid sending WebSocket message if client disconnected
            if websocket.client_state.name != "CONNECTED":
                print(f"Client disconnected before worker ready for session {session_id}")
                return
            await websocket.send_text("ERROR: Worker process initialization timeout")
            await websocket.close(code=4001, reason="Worker process initialization timeout")
            return

        # Guard: avoid sending WebSocket message if client disconnected
        if websocket.client_state.name == "CONNECTED":
            await websocket.send_text("READY: Worker process is ready")
        else:
            print(f"Client disconnected before READY message for session {session_id}")

    try:
        # Create a task to continuously check for translations from the worker process
        async def check_translations():
            while True:
                try:
                    # Check if there's any translation output from the worker process
                    if not session.output_queue.empty():
                        translation = await asyncio.get_event_loop().run_in_executor(
                            None, session.output_queue.get_nowait
                        )
                        
                        # Check if it's an error message
                        if isinstance(translation, tuple) and translation[0] == "ERROR":
                            print(f"Error in worker process: {translation[1]}")
                            await websocket.send_text(f"ERROR: {translation[1]}")
                        else:
                            print(f"Got translation: {translation}")
                            await websocket.send_text(translation)
                except Empty:
                    pass
                except Exception as e:
                    print(f"Error checking translations: {e}")
                
                # Short sleep to avoid busy waiting
                await asyncio.sleep(0.01)
        
        # Start the translation checking task
        translation_task = asyncio.create_task(check_translations())
        
        # Process incoming audio data
        while True:
            try:
                # Receive data from the WebSocket
                message = await websocket.receive()

                # Update activity timestamp
                update_session_activity(session_id)
                # Update ping timestamp when data is received
                update_session_ping(session_id)

                # Check if this is a control message (text) or audio data (bytes)
                if "text" in message:
                    # Handle control messages
                    control_message = message["text"]
                    print(f"Received control message for session {session_id}: {control_message}")

                    if control_message == "EOF":
                        # This is an explicit End-Of-File signal from the client
                        # Process an empty segment with is_last=True to signal completion
                        print(f"Received EOF signal for session {session_id}, marking processing as complete")

                        # Send an empty audio segment with is_last=True to indicate completion
                        empty_segment = np.array([], dtype=np.float32)
                        await session.process_segment(empty_segment, is_last=True)

                        # Send a confirmation message to the client
                        await websocket.send_text("PROCESSING_COMPLETE: File processing finished")
                        continue

                    # Handle other potential control messages here
                    elif control_message.startswith("LATENCY:"):
                        # Example of another control message to dynamically adjust latency
                        try:
                            latency_value = int(control_message.split(":")[1])
                            session.control_queue.put(f"update_latency:{latency_value}")
                            await websocket.send_text(f"LATENCY_UPDATED: Set to {latency_value}")
                        except (ValueError, IndexError):
                            await websocket.send_text("ERROR: Invalid latency format")
                        continue

                elif "bytes" in message:
                    # This is audio data
                    data = message["bytes"]

                    # Convert bytes to numpy array
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    chunk_count += 1
                    print(f"Received chunk {chunk_count}, size: {len(audio_data)}")

                    # Process the segment (send to worker process)
                    # For regular chunks, is_last is always False
                    await session.process_segment(audio_data, is_last=False)
                    
            except starlette.websockets.WebSocketDisconnect:
                print(f"WebSocket disconnected for session {session_id}")
                break
            except RuntimeError as e:
                if "disconnect message has been received" in str(e):
                    print(f"WebSocket already disconnected for session {session_id}")
                    break
                else:
                    print(f"Runtime error in WebSocket: {str(e)}")
                    break
            except Exception as e:
                print(f"Error processing WebSocket message for session {session_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Send error message to client if connection is still open
                try:
                    await websocket.send_text(f"ERROR: {str(e)}")
                except:
                    pass
                break

    except starlette.websockets.WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        print(f"Error in WebSocket connection: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # This block will execute when the WebSocket connection is closed
        print(f"WebSocket connection closed for session {session_id}")
        # Don't immediately delete the session, let the idle cleanup handle it
        # This allows reconnection if the page is refreshed
        
        # Cancel the translation checking task
        if 'translation_task' in locals():
            translation_task.cancel()
            try:
                await translation_task
            except asyncio.CancelledError:
                pass

@app.post("/download_youtube")
async def download_youtube(request: Request, background_tasks: BackgroundTasks):
    try:
        query_params = dict(request.query_params)
        url = query_params.get("url")
        session_id = query_params.get("session_id")

        if not url:
            return {"error": "Missing URL parameter"}

        # (Optional) log session_id for debugging
        if session_id:
            print(f"Download request received for session: {session_id}")

        output_path = f"/mnt/aries/data6/jiaxuanluo/tmp/video_{session_id}.mp4"

        import subprocess
        cmd = [
            "yt-dlp",
            #"--cookies-from-browser", "chrome",
            "--cookies=/mnt/aries/data6/jiaxuanluo/cookies.txt",
            #"-f", "best[ext=mp4]/best",
            "-f", "bestvideo+bestaudio/best",
            "--merge-output-format", "mp4",
            "--no-continue",
            "--no-part",
            "-o", output_path,
            url
        ]

        print("Running command:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("yt-dlp failed:", result.stderr)
            raise Exception(f"yt-dlp error: {result.stderr}")
        else:
            print("yt-dlp success:", result.stdout)

        print("Download completed.")
        print(f"Checking file after download: {output_path}")
        if os.path.exists(output_path):
            print(f"File exists. Size: {os.path.getsize(output_path)} bytes")
        else:
            print("Download failed: File not found.")

        background_tasks.add_task(os.remove, output_path)
        return FileResponse(output_path, media_type='video/mp4', filename=f'video_{session_id}.mp4')
    except Exception as e:
        return {"error": str(e)}

@app.post("/update_latency")
async def update_latency(session_id: str, latency_multiplier: int):
    """Update the latency multiplier for a session."""
    try:
        if session_id not in active_sessions:
            return {"success": False, "error": "Invalid session ID"}
        
        # Get the session
        session = active_sessions[session_id]
        
        # 确保工作进程已准备就绪
        if not session.is_ready:
            # 尝试等待工作进程准备就绪，最多等待10秒
            if not await session.wait_for_ready(timeout=10):
                return {"success": False, "error": "Worker process not ready, try again later"}
        
        # Update the latency multiplier in the session args
        session.args.latency_multiplier = latency_multiplier
        
        # Send a command to the worker process to update the latency multiplier
        session.control_queue.put(f"update_latency:{latency_multiplier}")
        
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
    """重置翻译会话，清空历史翻译内容"""
    try:
        # 从session_id中提取user_id和language_pair
        if session_id.startswith("InfiniSST_"):
            # 调度器会话：InfiniSST_English -> Chinese_1_client_xxx
            parts = session_id.split("_")
            if len(parts) >= 4:
                language_pair = f"{parts[1]} -> {parts[3]}"
                client_id = "_".join(parts[4:])  # client_xxx
                user_id = client_id
                
                # 重置调度器会话
                if global_scheduler:
                    success = global_scheduler.reset_session(user_id, language_pair)
                    if success:
                        logger.info(f"✅ 调度器会话重置成功: {session_id}")
                        return {
                            "status": "success", 
                            "message": f"调度器会话 {session_id} 重置成功",
                            "session_type": "scheduler"
                        }
                    else:
                        return {
                            "status": "error", 
                            "message": f"调度器会话 {session_id} 不存在或重置失败",
                            "session_type": "scheduler"
                        }
                else:
                    return {"status": "error", "message": "调度器不可用"}
        
        # 传统会话处理
        if session_id in active_sessions:
            session = active_sessions[session_id]
            
            if session:
                # 重置传统会话
                session.reset()
                logger.info(f"✅ 传统会话重置成功: {session_id}")
                return {
                    "status": "success", 
                    "message": f"传统会话 {session_id} 重置成功",
                    "session_type": "traditional"
                }
            else:
                return {"status": "error", "message": f"会话 {session_id} 未找到有效的session对象"}
        
        return {"status": "error", "message": f"会话 {session_id} 不存在"}
        
    except Exception as e:
        logger.error(f"重置会话 {session_id} 时出错: {e}")
        return {"status": "error", "message": f"重置会话失败: {str(e)}"}

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
        
        # 检查是否是基于调度器的会话
        is_scheduler_based = isinstance(session, dict) and session.get('is_scheduler_based', False)
        
        if is_scheduler_based:
            print(f"Deleting scheduler-based session {session_id}")
            
            # 🔥 关键：调用调度器的会话清理功能
            if global_scheduler:
                try:
                    user_id = session.get('user_id', session_id)
                    language_pair = session.get('language_pair', 'English -> Chinese')
                    
                    cleanup_success = global_scheduler.cleanup_session(user_id, language_pair)
                    if cleanup_success:
                        print(f"✅ 调度器会话 {session_id} 清理成功，KV cache页面已释放")
                    else:
                        print(f"⚠️ 调度器会话 {session_id} 清理失败或会话不存在")
                        
                except Exception as e:
                    print(f"❌ 调度器会话清理出错: {e}")
            else:
                print(f"⚠️ 全局调度器不可用，无法清理会话KV cache")
        else:
            print(f"Deleting traditional session {session_id}")
            # 传统会话需要清理GPU资源
            if hasattr(session, 'cleanup'):
                session.cleanup()
        
        # Remove the session from active sessions
        del active_sessions[session_id]
        
        # Remove the session from last activity tracking
        if session_id in session_last_activity:
            del session_last_activity[session_id]
        
        # Remove the session from ping tracking
        if session_id in session_last_ping:
            del session_last_ping[session_id]
            
        # Remove the session from GPU mapping (仅对传统会话)
        if session_id in session_gpu_map:
            gpu_id = session_gpu_map[session_id]
            del session_gpu_map[session_id]
            print(f"Released GPU {gpu_id} from session {session_id}")
        
        # Force garbage collection to free up memory
        gc.collect()
        
        print(f"Session {session_id} deleted, {len(active_sessions)} active sessions remaining")
        
        return {"success": True}
    except Exception as e:
        print(f"Error deleting session: {e}")
        return {"success": False, "error": str(e)}

@app.get("/health")
async def health_check():
    try:
        return {
            "status": "healthy",
            "time": int(time.time()),
            "scheduler_available": global_scheduler is not None,
            "active_sessions": len(active_sessions),
            "scheduler_sessions": global_scheduler.stats['active_sessions'] if global_scheduler else 0
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/diagnose")
async def diagnose_system():
    """🔍 诊断系统状态，特别是卡住的session"""
    try:
        diagnosis = {
            "timestamp": time.time(),
            "traditional_sessions": {},
            "scheduler_diagnosis": None
        }
        
        # 诊断传统session
        current_time = time.time()
        for session_id, session_info in active_sessions.items():
            session = session_info.get('session')
            if session:
                last_activity = session_info.get('last_activity', 0)
                inactive_time = current_time - last_activity
                
                diagnosis["traditional_sessions"][session_id] = {
                    "session_type": "traditional",
                    "language_pair": session_info.get('language_pair', 'unknown'),
                    "is_ready": getattr(session, 'is_ready', False),
                    "worker_alive": session.process.is_alive() if hasattr(session, 'process') else False,
                    "inactive_seconds": inactive_time,
                    "status": "normal" if inactive_time < 30 else "potentially_stuck"
                }
        
        # 诊断scheduler系统
        if global_scheduler:
            scheduler_diagnosis = global_scheduler.diagnose_stuck_sessions()
            diagnosis["scheduler_diagnosis"] = scheduler_diagnosis
            
            # 添加队列统计
            queue_stats = global_scheduler.get_queue_stats()
            diagnosis["queue_stats"] = queue_stats
        
        return diagnosis
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }

@app.post("/load_models")
async def load_models():
    """Load models to inference engine"""
    try:
        if not global_inference_engine:
            return {"success": False, "error": "Inference engine not available"}
        
        # 加载所有模型
        success = global_inference_engine.load_all_models()
        
        if success:
            return {
                "success": True,
                "message": "所有模型加载成功",
                "loaded_gpus": list(global_inference_engine.engines.keys())
            }
        else:
            return {
                "success": False,
                "error": "部分或全部模型加载失败"
            }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/ping")
async def ping_session(session_id: str):
    """Update the last ping timestamp for a session to indicate the webpage is still open."""
    import time
    start_time = time.time()
    
    try:
        # Log system resource usage occasionally
        import psutil
        import torch
        
        if hasattr(ping_session, '_call_count'):
            ping_session._call_count += 1
        else:
            ping_session._call_count = 1
        
        # Log detailed system stats every 20 pings for debugging 503 errors
        if ping_session._call_count % 20 == 0:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            disk = psutil.disk_usage('/')
            
            gpu_info = ""
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_mem = torch.cuda.memory_stats(i)
                    allocated = gpu_mem.get('allocated_bytes.all.current', 0) / 1024**3
                    reserved = gpu_mem.get('reserved_bytes.all.current', 0) / 1024**3
                    gpu_info += f" GPU{i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved"
            
            print(f"[PING-{ping_session._call_count}] System status - CPU: {cpu}%, Memory: {memory.percent}%, Disk: {disk.percent}%{gpu_info}")
            print(f"[PING-{ping_session._call_count}] Active sessions: {len(active_sessions)}, Queue: {len(session_queue)}")
        
        # Check if session exists
        if session_id not in active_sessions:
            print(f"[PING ERROR] Session {session_id} not found in active sessions")
            return {"success": False, "error": "Invalid session ID"}
        
        # Check if session worker process is still alive
        session = active_sessions[session_id]
        if hasattr(session, 'process') and not session.process.is_alive():
            print(f"[PING ERROR] Worker process for session {session_id} is dead (PID: {session.process.pid})")
            return {"success": False, "error": "Worker process terminated"}
            
        # Update the last ping timestamp
        update_session_ping(session_id)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        if ping_session._call_count % 10 == 0 or processing_time > 100:  # Log slow pings
            print(f"[PING] Session {session_id} - Processing time: {processing_time:.1f}ms")
        
        return {"success": True, "processing_time_ms": round(processing_time, 1)}
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        print(f"[PING EXCEPTION] Session {session_id} - Error after {processing_time:.1f}ms: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e), "processing_time_ms": round(processing_time, 1)}

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Explicit root path handling
@app.get("/")
async def read_index():
    """Return index.html"""
    return FileResponse('static/index.html')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InfiniSST Translation API Server")
    InfiniSST.add_args(parser)
    
    # Add server-specific arguments
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                       help="Host to bind the server to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001, 
                       help="Port to bind the server to (default: 8001)")
    parser.add_argument("--reload", action="store_true", 
                       help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print(f"Starting server with {len(gpus)} GPUs available")
    print(f"Each translation session will run in its own worker process")
    print(f"Server will be available at http://{args.host}:{args.port}")

    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        reload=args.reload if hasattr(args, 'reload') else False,
        workers=1,  # 单个worker避免进程间通信问题
        limit_concurrency=100,  # 限制并发连接数
        limit_max_requests=1000,  # 最大请求数后重启worker
        timeout_keep_alive=30,  # Keep-alive超时
        access_log=True,  # 启用访问日志
    )