import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from threading import Lock, Thread
from enum import Enum

import torch
import numpy as np

logger = logging.getLogger(__name__)

class RequestStage(Enum):
    """Request processing stage"""
    PREFILL = "prefill"
    DECODE = "decode"

@dataclass
class UserSession:
    """
    Maintains state for a user session including all necessary information
    Based on the api.py structure and infinisst.py requirements
    """
    user_id: str
    language_id: str
    session_id: str
    
    # Speech processing state
    source: List[float] = field(default_factory=list)  # Audio samples
    source_finished: bool = False
    source_sample_rate: int = 16000
    src_len: int = 0
    
    # Translation state
    target: List[str] = field(default_factory=list)
    target_ids: List[int] = field(default_factory=list)
    segment_idx: int = 0
    
    # Cache state
    speech_cache: Optional[Any] = None
    past_key_values: Optional[Any] = None
    
    # Session management
    last_activity: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    
    # Translation parameters
    latency_multiplier: int = 2
    max_new_tokens: int = 20
    
    def reset(self):
        """Reset session state for new translation"""
        self.source = []
        self.source_finished = False
        self.src_len = 0
        self.target = []
        self.target_ids = []
        self.segment_idx = 0
        self.speech_cache = None
        self.past_key_values = None
        self.last_activity = time.time()

@dataclass
class InferenceRequest:
    """
    Single inference request 
    """
    request_id: str
    user_id: str
    language_id: str
    session_id: str
    stage: RequestStage
    
    # Input data 
    speech_batch: torch.Tensor  # Speech input tensor
    input_ids: torch.Tensor     # Text input token IDs
    
    # Generation parameters 
    max_new_tokens: int = 20
    beam_size: int = 1
    do_sample: bool = False
    top_p: float = 0.9
    top_k: int = 50
    temperature: float = 1.0
    repetition_penalty: float = 1.0
    
    # State management
    speech_cache: Optional[Any] = None
    past_key_values: Optional[Any] = None
    encoder_input_ids: Optional[torch.Tensor] = None
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # Higher number = higher priority
    
    # Result handling
    result_callback: Optional[callable] = None
    is_processing: bool = False
    is_completed: bool = False
    result: Optional[Dict[str, Any]] = None

class LLMScheduler:
    """
    - Maintains two FCFS queues (PREFILL and DECODE)
    - Prioritizes PREFILL queue over DECODE queue
    - Maximum batch size of 32 requests
    """
    
    def __init__(self, gpu_language_map: Dict[int, str], args=None):
        """
        Initialize scheduler with GPU-to-language mapping
        
        Args:
            gpu_language_map: Dict mapping GPU ID to language pair (e.g., {0: "en-zh", 1: "en-de"})
            args: Additional configuration arguments
        """
        self.gpu_language_map = gpu_language_map  # {gpu_id: language_id}
        self.language_gpu_map = {v: k for k, v in gpu_language_map.items()}  # {language_id: gpu_id}
        
        # Configuration
        self.max_batch_size = getattr(args, 'max_batch_size', 32) if args else 32
        self.batch_timeout = getattr(args, 'batch_timeout', 0.1) if args else 0.1  # seconds
        self.session_timeout = getattr(args, 'session_timeout', 3600) if args else 3600  # 1 hour
        
        # FCFS queues - separate queues for each GPU/language
        self.prefill_queues: Dict[int, deque] = {gpu_id: deque() for gpu_id in gpu_language_map.keys()}
        self.decode_queues: Dict[int, deque] = {gpu_id: deque() for gpu_id in gpu_language_map.keys()}
        
        # User session management
        self.user_sessions: Dict[str, Dict[str, UserSession]] = {}  # {language_id: {user_id: session}}
        
        # Thread safety
        self.queue_lock = Lock()
        self.session_lock = Lock()
        
        # Processing state
        self.is_running = False
        self.processing_threads: Dict[int, Thread] = {}  # One thread per GPU
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'active_sessions': 0,
            'queue_sizes': {gpu_id: {'prefill': 0, 'decode': 0} for gpu_id in gpu_language_map.keys()}
        }
        
        logger.info(f"LLMScheduler initialized with GPU mapping: {gpu_language_map}")
        logger.info(f"Max batch size: {self.max_batch_size}")
    
    def start(self):
        """Start the scheduler processing loops for all GPUs"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        
        # Start one processing thread per GPU
        for gpu_id in self.gpu_language_map.keys():
            thread = Thread(target=self._processing_loop, args=(gpu_id,), daemon=True)
            thread.start()
            self.processing_threads[gpu_id] = thread
            logger.info(f"Started processing thread for GPU {gpu_id} (language: {self.gpu_language_map[gpu_id]})")
        
        logger.info("Scheduler started")
    
    def stop(self):
        """Stop all scheduler processing loops"""
        self.is_running = False
        
        # Wait for all threads to finish
        for gpu_id, thread in self.processing_threads.items():
            thread.join(timeout=5.0)
            logger.info(f"Stopped processing thread for GPU {gpu_id}")
        
        self.processing_threads.clear()
        logger.info("Scheduler stopped")
    
    def get_or_create_session(self, user_id: str, language_id: str) -> UserSession:
        """Get existing session or create new one"""
        with self.session_lock:
            if language_id not in self.user_sessions:
                self.user_sessions[language_id] = {}
            
            if user_id not in self.user_sessions[language_id]:
                session_id = f"{user_id}_{language_id}_{int(time.time())}"
                session = UserSession(
                    user_id=user_id,
                    language_id=language_id,
                    session_id=session_id
                )
                self.user_sessions[language_id][user_id] = session
                self.stats['active_sessions'] += 1
                logger.info(f"Created new session {session_id} for user {user_id}, language {language_id}")
            else:
                session = self.user_sessions[language_id][user_id]
                session.last_activity = time.time()
            
            return self.user_sessions[language_id][user_id]
    
    def submit_request(self, 
                      user_id: str,
                      language_id: str,
                      speech_data: Union[torch.Tensor, np.ndarray, List[float]],
                      stage: RequestStage = RequestStage.PREFILL,
                      is_final: bool = False,
                      max_new_tokens: int = 20,
                      result_callback: Optional[callable] = None) -> str:
        """
        Submit a request to the appropriate queue based on language and stage
        
        Args:
            user_id: Unique identifier for the user
            language_id: Language pair identifier (e.g., "en-zh")
            speech_data: Audio data (will be converted to tensor)
            stage: PREFILL or DECODE stage
            is_final: Whether this is the final segment
            max_new_tokens: Maximum tokens to generate
            result_callback: Callback function for results
            
        Returns:
            request_id: Unique identifier for this request
        """
        # Validate language support
        if language_id not in self.language_gpu_map:
            raise ValueError(f"Unsupported language pair: {language_id}. Supported: {list(self.language_gpu_map.keys())}")
        
        gpu_id = self.language_gpu_map[language_id]
        
        # Get or create user session
        session = self.get_or_create_session(user_id, language_id)
        
        # Update session with new speech data
        if isinstance(speech_data, (list, np.ndarray)):
            speech_data = torch.tensor(speech_data, dtype=torch.float32)
        elif not isinstance(speech_data, torch.Tensor):
            raise ValueError("speech_data must be list, numpy array, or torch tensor")
        
        # Update session state 
        session.source.extend(speech_data.tolist() if speech_data.dim() == 1 else speech_data.flatten().tolist())
        session.source_finished = is_final
        session.last_activity = time.time()
        
        # Prepare input data 
        request_id = str(uuid.uuid4())
        
        # placeholder that matches the format
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)  # Placeholder
        
        # Prepare speech batch 
        speech_batch = speech_data.unsqueeze(0) if speech_data.dim() == 1 else speech_data
        
        request = InferenceRequest(
            request_id=request_id,
            user_id=user_id,
            language_id=language_id,
            session_id=session.session_id,
            stage=stage,
            speech_batch=speech_batch,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            speech_cache=session.speech_cache,
            past_key_values=session.past_key_values,
            result_callback=result_callback
        )
        
        # Add to appropriate queue
        with self.queue_lock:
            if stage == RequestStage.PREFILL:
                self.prefill_queues[gpu_id].append(request)
                self.stats['queue_sizes'][gpu_id]['prefill'] += 1
            else:
                self.decode_queues[gpu_id].append(request)
                self.stats['queue_sizes'][gpu_id]['decode'] += 1
            
            self.stats['total_requests'] += 1
        
        logger.debug(f"Submitted {stage.value} request {request_id} for user {user_id}, language {language_id}, GPU {gpu_id}")
        return request_id
    
    def _processing_loop(self, gpu_id: int):
        """
        Main processing loop for a specific GPU
        Implements the scheduling policy: PREFILL queue has priority over DECODE queue
        """
        language_id = self.gpu_language_map[gpu_id]
        logger.info(f"Starting processing loop for GPU {gpu_id} (language: {language_id})")
        
        while self.is_running:
            try:
                # Get batch of requests following the priority rule
                batch = self._get_request_batch(gpu_id)
                
                if not batch:
                    time.sleep(0.001)  
                    continue
                
                # Process the batch
                self._process_batch(batch, gpu_id)
                
                # Clean up old sessions periodically
                if time.time() % 60 < 1:  # Every minute
                    self._cleanup_sessions()
                
            except Exception as e:
                logger.error(f"Error in processing loop for GPU {gpu_id}: {e}")
                time.sleep(0.1)  # Brief pause on error
        
        logger.info(f"Processing loop stopped for GPU {gpu_id}")
    
    def _get_request_batch(self, gpu_id: int) -> List[InferenceRequest]:
        """
        Get a HOMOGENEOUS batch of requests (either all PREFILL or all DECODE)
        
        Scheduling Policy:
        1. If PREFILL queue has requests: Create pure PREFILL batch (up to 32 requests)
        2. If PREFILL queue is empty: Create pure DECODE batch (up to 32 requests)
        3. NEVER mix PREFILL and DECODE in the same batch
        """
        batch = []
        
        with self.queue_lock:
            prefill_queue = self.prefill_queues[gpu_id]
            decode_queue = self.decode_queues[gpu_id]
            
            # Priority 1: Create PREFILL batch
            if prefill_queue:
                while len(batch) < self.max_batch_size and prefill_queue:
                    request = prefill_queue.popleft()
                    batch.append(request)
                    self.stats['queue_sizes'][gpu_id]['prefill'] -= 1
                # decouple PD
                assert all(req.stage == RequestStage.PREFILL for req in batch)
                logger.debug(f"Created PREFILL batch of size {len(batch)} for GPU {gpu_id}")
            
            # Priority 2: Create  DECODE batch ( if no PREFILL requests)
            elif decode_queue:
                while len(batch) < self.max_batch_size and decode_queue:
                    request = decode_queue.popleft()
                    batch.append(request)
                    self.stats['queue_sizes'][gpu_id]['decode'] -= 1
                
                assert all(req.stage == RequestStage.DECODE for req in batch)
                logger.debug(f"Created DECODE batch of size {len(batch)} for GPU {gpu_id}")
        
        return batch
    
    def _process_batch(self, batch: List[InferenceRequest], gpu_id: int):
        """
        Process a batch of requests using inference engine or fallback to simulation
        """
        if not batch:
            return
        
        language_id = self.gpu_language_map[gpu_id]
        logger.debug(f"Processing batch of {len(batch)} requests on GPU {gpu_id} for language {language_id}")
        
        # Mark requests as processing
        for request in batch:
            request.is_processing = True
        
        try:
            # 尝试使用实际的推理引擎
            if hasattr(self, 'inference_engine') and self.inference_engine:
                results = self.inference_engine.process_batch(gpu_id, batch)
                
                # 处理推理结果
                for i, request in enumerate(batch):
                    if i < len(results):
                        result = results[i]
                        self._update_session_with_result(request, result)
                        logger.debug(f"Request {request.request_id} completed with inference engine")
                    else:
                        # 处理缺失的结果
                        self._handle_failed_request(request, "Missing inference result")
            else:
                # 使用模拟推理
                logger.warning(f"No inference engine available, using simulation for GPU {gpu_id}")
                self._simulate_inference(batch, gpu_id)
                
        except Exception as e:
            logger.error(f"Batch processing failed on GPU {gpu_id}: {e}")
            # 处理所有请求的错误
            for request in batch:
                self._handle_failed_request(request, f"Batch processing failed: {str(e)}")
    
    def _update_session_with_result(self, request: InferenceRequest, result: Dict[str, Any]):
        """使用推理结果更新用户会话"""
        try:
            # 更新用户会话
            session = self.user_sessions[request.language_id][request.user_id]
            
            if result.get('success', False):
                generated_text = result.get('generated_text', '')
                generated_tokens = result.get('generated_tokens', [])
                
                if generated_text:
                    session.target.append(generated_text)
                if generated_tokens:
                    session.target_ids.extend(generated_tokens)
                
                session.segment_idx += 1
                
                # 更新缓存状态
                if 'speech_cache' in result:
                    session.speech_cache = result['speech_cache']
                if 'past_key_values' in result:
                    session.past_key_values = result['past_key_values']
            
            session.last_activity = time.time()
            
            # 标记请求完成
            request.result = result
            request.is_completed = True
            request.is_processing = False
            
            # 调用回调函数
            if request.result_callback:
                try:
                    request.result_callback(result)
                except Exception as e:
                    logger.error(f"Error in result callback for request {request.request_id}: {e}")
            
            self.stats['completed_requests'] += 1
            
        except Exception as e:
            logger.error(f"Error updating session for request {request.request_id}: {e}")
            self._handle_failed_request(request, f"Session update failed: {str(e)}")
    
    def _handle_failed_request(self, request: InferenceRequest, error_msg: str):
        """处理失败的请求"""
        error_result = {
            'request_id': request.request_id,
            'success': False,
            'error': error_msg,
            'generated_text': '',
            'generated_tokens': [],
            'stage': request.stage.value
        }
        
        request.result = error_result
        request.is_completed = True
        request.is_processing = False
        
        if request.result_callback:
            try:
                request.result_callback(error_result)
            except Exception as e:
                logger.error(f"Error in error callback for request {request.request_id}: {e}")
        
        self.stats['failed_requests'] = self.stats.get('failed_requests', 0) + 1
    
    def _simulate_inference(self, batch: List[InferenceRequest], gpu_id: int):
        """模拟推理处理（用于测试和开发）"""
        for request in batch:
            try:
                # 模拟处理时间
                processing_time = 0.1 if request.stage == RequestStage.PREFILL else 0.05
                time.sleep(processing_time)
                
                # 生成模拟结果
                result = {
                    'request_id': request.request_id,
                    'success': True,
                    'generated_text': f"模拟翻译结果 {request.request_id}",
                    'generated_tokens': [100, 101, 102],  # 模拟token IDs
                    'finished': True,
                    'stage': request.stage.value,
                    'processing_time': processing_time
                }
                
                self._update_session_with_result(request, result)
                logger.debug(f"Request {request.request_id} completed with simulation")
                
            except Exception as e:
                logger.error(f"Error in simulation for request {request.request_id}: {e}")
                self._handle_failed_request(request, f"Simulation failed: {str(e)}")
    
    def set_inference_engine(self, inference_engine):
        """设置推理引擎"""
        self.inference_engine = inference_engine
        logger.info("推理引擎已设置到调度器")
    
    def _cleanup_sessions(self):
        """Clean up old/inactive sessions"""
        current_time = time.time()
        sessions_to_remove = []
        
        with self.session_lock:
            for language_id, user_sessions in self.user_sessions.items():
                for user_id, session in list(user_sessions.items()):
                    if current_time - session.last_activity > self.session_timeout:
                        sessions_to_remove.append((language_id, user_id))
            
            # Remove expired sessions
            for language_id, user_id in sessions_to_remove:
                if language_id in self.user_sessions and user_id in self.user_sessions[language_id]:
                    del self.user_sessions[language_id][user_id]
                    self.stats['active_sessions'] -= 1
                    logger.info(f"Cleaned up expired session for user {user_id}, language {language_id}")
    
    def get_session_info(self, user_id: str, language_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a user session"""
        with self.session_lock:
            if language_id in self.user_sessions and user_id in self.user_sessions[language_id]:
                session = self.user_sessions[language_id][user_id]
                return {
                    'session_id': session.session_id,
                    'user_id': session.user_id,
                    'language_id': session.language_id,
                    'source_length': len(session.source),
                    'source_finished': session.source_finished,
                    'target_segments': len(session.target),
                    'segment_idx': session.segment_idx,
                    'last_activity': session.last_activity,
                    'created_at': session.created_at
                }
        return None
    
    def reset_session(self, user_id: str, language_id: str) -> bool:
        """Reset a user session"""
        with self.session_lock:
            if language_id in self.user_sessions and user_id in self.user_sessions[language_id]:
                session = self.user_sessions[language_id][user_id]
                session.reset()
                logger.info(f"Reset session for user {user_id}, language {language_id}")
                return True
        return False
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get current queue and system statistics"""
        with self.queue_lock:
            current_stats = self.stats.copy()
            current_stats['gpu_language_map'] = self.gpu_language_map.copy()
            current_stats['timestamp'] = time.time()
            return current_stats
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language pairs"""
        return list(self.language_gpu_map.keys()) 