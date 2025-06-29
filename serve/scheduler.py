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
    
    # 🔍 内存使用追踪
    memory_usage: Dict[str, int] = field(default_factory=lambda: {
        'speech_pages': 0,
        'llm_prefill_pages': 0, 
        'llm_decode_pages': 0,
        'total_pages': 0,
        'peak_pages': 0,
        'allocation_count': 0
    })
    
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
        
        # 重置内存使用追踪
        self.memory_usage = {
            'speech_pages': 0,
            'llm_prefill_pages': 0, 
            'llm_decode_pages': 0,
            'total_pages': 0,
            'peak_pages': 0,
            'allocation_count': 0
        }
    
    def update_memory_usage(self, cache_type: str, pages_used: int):
        """更新内存使用统计"""
        if cache_type in self.memory_usage:
            self.memory_usage[cache_type] = pages_used
        
        # 更新总页面数
        total = (self.memory_usage.get('speech_pages', 0) + 
                self.memory_usage.get('llm_prefill_pages', 0) + 
                self.memory_usage.get('llm_decode_pages', 0))
        self.memory_usage['total_pages'] = total
        
        # 更新峰值
        if total > self.memory_usage.get('peak_pages', 0):
            self.memory_usage['peak_pages'] = total
        
        self.memory_usage['allocation_count'] += 1
        
        print(f"🔍 [SESSION-MEMORY] {self.session_id} 内存使用:")
        print(f"   - Speech: {self.memory_usage.get('speech_pages', 0)} 页")
        print(f"   - LLM Prefill: {self.memory_usage.get('llm_prefill_pages', 0)} 页")
        print(f"   - LLM Decode: {self.memory_usage.get('llm_decode_pages', 0)} 页")
        print(f"   - 总计: {total} 页 (峰值: {self.memory_usage.get('peak_pages', 0)} 页)")
        print(f"   - 分配次数: {self.memory_usage.get('allocation_count', 0)}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取内存使用摘要"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'language_id': self.language_id,
            'memory_usage': self.memory_usage.copy(),
            'session_age_seconds': time.time() - self.created_at,
            'inactive_seconds': time.time() - self.last_activity
        }

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
    segment_idx: int = 0
    translations_list: List[str] = field(default_factory=list)
    session_src_len: int = 0  # 🔥 添加：会话的已处理音频长度
    
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
        
        # 检查音频数据长度
        audio_length = speech_data.numel() if speech_data.dim() > 0 else 0
        print(f"🔍 [SCHEDULER] Audio data length: {audio_length}, shape: {speech_data.shape}")
        print(f"🔍 [SCHEDULER] Audio stats: min={speech_data.min().item() if audio_length > 0 else 0:.6f}, max={speech_data.max().item() if audio_length > 0 else 0:.6f}")
        
        # 如果音频数据为空或太短，记录警告但不填充
        MIN_AUDIO_LENGTH = 160  # 0.01秒 @ 16kHz，更宽松的阈值
        if audio_length == 0:
            print(f"⚠️ [SCHEDULER] Received empty audio data for user {user_id}, skipping request")
            raise ValueError("Empty audio data received")
        elif audio_length < MIN_AUDIO_LENGTH:
            print(f"⚠️ [SCHEDULER] Audio data too short ({audio_length} samples), but processing anyway")
        
        # Update session state 
        session.source.extend(speech_data.tolist() if speech_data.dim() == 1 else speech_data.flatten().tolist())
        session.source_finished = is_final
        session.last_activity = time.time()
        
        print(f"🔍 [SCHEDULER] Session source now has {len(session.source)} total samples")
        print(f"🔍 [SCHEDULER] Session src_len (already processed): {session.src_len} samples")
        print(f"🔍 [SCHEDULER] New samples to process: {len(session.source) - session.src_len}")
        
        # 🔥 修改设计：传递完整的音频历史，让推理引擎处理增量逻辑
        # 这样模型的 _prepare_speech 方法能正确使用 src_len 进行增量处理
        full_audio_data = session.source
        speech_batch_for_processing = torch.tensor(full_audio_data, dtype=torch.float32)
        
        # 检查是否有新数据需要处理
        new_samples_count = len(session.source) - session.src_len
        print(f"🔍 [SCHEDULER] Passing full audio history: {len(full_audio_data)} samples")
        print(f"🔍 [SCHEDULER] New samples in this batch: {new_samples_count}")
        
        if new_samples_count <= 0:
            print(f"⚠️ [SCHEDULER] No new audio data to process for user {user_id}")
            raise ValueError("No new audio data to process")
        elif new_samples_count < MIN_AUDIO_LENGTH:
            print(f"⚠️ [SCHEDULER] New audio data too short ({new_samples_count} samples), but processing anyway")
        
        # Prepare input data 
        request_id = str(uuid.uuid4())
        
        # 修复input_ids的维度 - 确保是2D tensor [batch_size, seq_len]
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)  # [1, 4] - 2D tensor
        
        # 修复speech batch的维度处理 - 使用完整的音频历史
        if speech_batch_for_processing.dim() == 1:
            speech_batch = speech_batch_for_processing.unsqueeze(0)  # [seq_len] -> [1, seq_len]
        else:
            speech_batch = speech_batch_for_processing
        
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
            result_callback=result_callback,
            # 🔥 传递会话状态信息
            segment_idx=session.segment_idx,
            translations_list=session.target,
            session_src_len=session.src_len
        )
        
        print(f"🔍 [SCHEDULER] Created request with session_src_len={session.src_len}")
        
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
                    try:
                        request = prefill_queue.popleft()
                        batch.append(request)
                        self.stats['queue_sizes'][gpu_id]['prefill'] -= 1
                    except IndexError:
                        # 队列为空，退出循环
                        print(f"⚠️ [SCHEDULER] Prefill queue empty during pop for GPU {gpu_id}")
                        break
                # decouple PD
                if batch:
                    assert all(req.stage == RequestStage.PREFILL for req in batch)
                    logger.debug(f"Created PREFILL batch of size {len(batch)} for GPU {gpu_id}")
            
            # Priority 2: Create  DECODE batch ( if no PREFILL requests)
            elif decode_queue:
                while len(batch) < self.max_batch_size and decode_queue:
                    try:
                        request = decode_queue.popleft()
                        batch.append(request)
                        self.stats['queue_sizes'][gpu_id]['decode'] -= 1
                    except IndexError:
                        # 队列为空，退出循环
                        print(f"⚠️ [SCHEDULER] Decode queue empty during pop for GPU {gpu_id}")
                        break
                
                if batch:
                    assert all(req.stage == RequestStage.DECODE for req in batch)
                    logger.debug(f"Created DECODE batch of size {len(batch)} for GPU {gpu_id}")
        
        return batch
    
    def _process_batch(self, batch: List[InferenceRequest], gpu_id: int):
        """
        Process a batch of requests using inference engine only (no simulation)
        """
        if not batch:
            return
        
        language_id = self.gpu_language_map[gpu_id]
        logger.debug(f"Processing batch of {len(batch)} requests on GPU {gpu_id} for language {language_id}")
        
        # Mark requests as processing
        for request in batch:
            request.is_processing = True
        
        try:
            # 🔥 只使用真实推理引擎，不再使用模拟推理
            if hasattr(self, 'inference_engine') and self.inference_engine:
                try:
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
                            
                except Exception as e:
                    logger.error(f"Inference engine failed for GPU {gpu_id}: {e}")
                    
                    # 🔥 智能错误处理：根据错误类型决定处理策略
                    if "页面池耗尽" in str(e) or "page" in str(e).lower() or "memory" in str(e).lower():
                        logger.warning(f"GPU内存不足，将请求重新排队等待...")
                        
                        # 将请求重新放回队列等待
                        self._requeue_requests_for_memory_wait(batch, gpu_id)
                        
                        # 尝试清理不活跃的会话
                        self._emergency_cleanup_sessions()
                        
                    else:
                        # 其他错误：标记所有请求失败
                        for request in batch:
                            self._handle_failed_request(request, f"Inference engine error: {str(e)}")
            else:
                # 没有推理引擎可用
                logger.error(f"No inference engine available for GPU {gpu_id}")
                for request in batch:
                    self._handle_failed_request(request, "Inference engine not available")
                
        except Exception as e:
            logger.error(f"Batch processing failed on GPU {gpu_id}: {e}")
            # 处理所有请求的错误
            for request in batch:
                self._handle_failed_request(request, f"Batch processing failed: {str(e)}")
    
    def _requeue_requests_for_memory_wait(self, batch: List[InferenceRequest], gpu_id: int):
        """将内存不足的请求重新放回队列等待"""
        with self.queue_lock:
            for request in batch:
                # 重置请求状态
                request.is_processing = False
                request.is_completed = False
                
                # 添加重试标记
                if not hasattr(request, 'retry_count'):
                    request.retry_count = 0
                request.retry_count += 1
                
                # 限制重试次数，避免无限重试
                max_retries = 3
                if request.retry_count <= max_retries:
                    # 重新放回对应的队列
                    if request.stage == RequestStage.PREFILL:
                        self.prefill_queues[gpu_id].appendleft(request)  # 放到队列前面，优先处理
                        self.stats['queue_sizes'][gpu_id]['prefill'] += 1
                        logger.info(f"Request {request.request_id} requeued for memory wait (retry {request.retry_count}/{max_retries})")
                    else:
                        self.decode_queues[gpu_id].appendleft(request)
                        self.stats['queue_sizes'][gpu_id]['decode'] += 1
                        logger.info(f"Request {request.request_id} requeued for memory wait (retry {request.retry_count}/{max_retries})")
                else:
                    # 超过重试次数，标记失败
                    logger.error(f"Request {request.request_id} exceeded max retries ({max_retries}) due to memory issues")
                    self._handle_failed_request(request, f"GPU memory exhausted after {max_retries} retries")
    
    def _simulate_inference(self, batch: List[InferenceRequest], gpu_id: int):
        """🚫 移除模拟推理功能 - 不再提供假的翻译结果"""
        logger.error("模拟推理已禁用 - 不提供假的翻译结果")
        
        for request in batch:
            self._handle_failed_request(request, "Simulation disabled - no fake results provided")
    
    def _update_session_with_result(self, request: InferenceRequest, result: Dict[str, Any]):
        """使用推理结果更新用户会话"""
        try:
            # 更新用户会话
            session = self.user_sessions[request.language_id][request.user_id]
            
            if result.get('success', False):
                generated_text = result.get('generated_text', '')
                generated_tokens = result.get('generated_tokens', [])
                
                # 🔥 修复：对于真实推理，需要正确处理翻译历史
                if generated_text:
                    # 添加新的翻译片段到session历史
                    session.target.append(generated_text)
                    print(f"🔍 [SCHEDULER] 添加翻译片段到session.target: {generated_text}")
                    
                    # 🔥 关键修复：根据目标语言决定连接方式
                    # 检查是否是中文翻译（根据language_id判断）
                    is_chinese_translation = "Chinese" in request.language_id or "zh" in request.language_id.lower()
                    
                    if is_chinese_translation:
                        # 中文翻译直接连接，不加空格
                        full_translation_history = ''.join(session.target)
                        print(f"📤 [SCHEDULER] 中文翻译历史连接（无空格）: {full_translation_history}")
                    else:
                        # 其他语言用空格连接
                        full_translation_history = ' '.join(session.target)
                        print(f"📤 [SCHEDULER] 其他语言翻译历史连接（空格）: {full_translation_history}")
                    
                    # 修改result中的generated_text为完整历史
                    result['generated_text'] = full_translation_history
                    result['new_segment'] = generated_text  # 保留原始片段
                    result['segment_count'] = len(session.target)
                    
                    print(f"📤 [SCHEDULER] 发送完整翻译历史 ({len(session.target)} 片段): {full_translation_history}")
                
                if generated_tokens:
                    session.target_ids.extend(generated_tokens)
                
                session.segment_idx += 1
                
                # 更新缓存状态
                if 'speech_cache' in result:
                    session.speech_cache = result['speech_cache']
                if 'past_key_values' in result:
                    session.past_key_values = result['past_key_values']
                
                # 🔥 关键更新：标记当前处理的音频数据已完成
                # 更新 src_len 到当前 session.source 的长度
                session.src_len = len(session.source)
                print(f"🔍 [SCHEDULER] Updated session src_len to {session.src_len} (marked as processed)")
            
            session.last_activity = time.time()
            
            # 标记请求完成
            request.result = result
            request.is_completed = True
            request.is_processing = False
            
            # 调用回调函数（现在传递的是包含完整历史的结果）
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
    
    def _emergency_cleanup_sessions(self):
        """紧急清理不活跃的会话以释放GPU内存"""
        current_time = time.time()
        cleaned_sessions = 0
        total_freed_pages = 0
        
        # 紧急清理阈值：5分钟不活跃
        emergency_timeout = 300  # 5 minutes
        
        with self.session_lock:
            sessions_to_remove = []
            
            for language_id, user_sessions in self.user_sessions.items():
                for user_id, session in list(user_sessions.items()):
                    inactive_time = current_time - session.last_activity
                    if inactive_time > emergency_timeout:
                        sessions_to_remove.append((language_id, user_id, session))
            
            # 移除超时的会话
            for language_id, user_id, session in sessions_to_remove:
                try:
                    memory_summary = session.get_memory_summary()
                    freed_pages = memory_summary['memory_usage']['total_pages']
                    total_freed_pages += freed_pages
                    
                    # 🔥 关键：调用推理引擎清理KV cache页面
                    self._cleanup_session_pages(session)
                    
                    if language_id in self.user_sessions and user_id in self.user_sessions[language_id]:
                        del self.user_sessions[language_id][user_id]
                        self.stats['active_sessions'] -= 1
                        cleaned_sessions += 1
                        
                        logger.info(f"🧹 紧急清理会话 {session.session_id}，释放 {freed_pages} 页内存")
                        logger.info(f"   - 不活跃时间: {inactive_time:.1f}s")
                        logger.info(f"   - 用户: {user_id}, 语言: {language_id}")
                        
                except Exception as e:
                    logger.error(f"Error during emergency cleanup of session {session.session_id}: {e}")
        
        if cleaned_sessions > 0:
            logger.info(f"🧹 紧急清理完成：清理了 {cleaned_sessions} 个会话，释放 {total_freed_pages} 页内存")
            
            # 🔥 最后手段：如果还是内存不足，强制重置所有页面表
            if total_freed_pages < 100:  # 如果释放的页面太少
                logger.warning("🚨 释放的页面不足，执行强制页面表重置")
                self._force_reset_all_pagetables()
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 清理GPU缓存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("🧹 清理GPU缓存完成")
            except Exception as e:
                logger.error(f"Error clearing GPU cache: {e}")
        else:
            logger.warning("🧹 紧急清理：没有找到可清理的会话")
            
            # 🔥 即使没有会话可清理，也可能需要重置页面表
            logger.warning("🚨 没有会话可清理但内存不足，执行强制页面表重置")
            self._force_reset_all_pagetables()
    
    def _cleanup_session_pages(self, session: UserSession):
        """清理单个session的KV cache页面"""
        try:
            if hasattr(self, 'inference_engine') and self.inference_engine:
                # 导入torch
                import torch
                
                # 创建一个临时的InferenceRequest用于清理
                cleanup_request = InferenceRequest(
                    request_id=f"cleanup_{session.session_id}",
                    user_id=session.user_id,
                    language_id=session.language_id,
                    session_id=session.session_id,
                    stage=RequestStage.PREFILL,
                    speech_batch=torch.empty(0),
                    input_ids=torch.empty(0, dtype=torch.long),
                    speech_cache=session.speech_cache,
                    past_key_values=session.past_key_values,
                    is_final=True  # 标记为最终请求，触发清理
                )
                
                # 从推理引擎获取对应GPU的引擎实例
                gpu_id = self.language_gpu_map.get(session.language_id)
                if gpu_id is not None:
                    engine = self.inference_engine.get_engine(gpu_id)
                    if engine:
                        logger.info(f"🧹 调用推理引擎清理session {session.session_id} 的KV cache页面")
                        engine._cleanup_session_kv_cache(cleanup_request)
                        
                        # 更新session的内存使用统计为0
                        session.memory_usage = {
                            'speech_pages': 0,
                            'llm_prefill_pages': 0, 
                            'llm_decode_pages': 0,
                            'total_pages': 0,
                            'peak_pages': session.memory_usage.get('peak_pages', 0),
                            'allocation_count': session.memory_usage.get('allocation_count', 0)
                        }
                        
                        logger.info(f"✅ Session {session.session_id} KV cache页面清理完成")
                    else:
                        logger.warning(f"⚠️ 无法获取GPU {gpu_id} 的推理引擎")
                else:
                    logger.warning(f"⚠️ 无法找到语言 {session.language_id} 对应的GPU")
            else:
                logger.warning("⚠️ 推理引擎不可用，跳过KV cache页面清理")
                
        except Exception as e:
            logger.error(f"清理session {session.session_id} 页面时出错: {e}")
    
    def _force_reset_all_pagetables(self):
        """强制重置所有GPU的页面表（最后手段）"""
        try:
            if hasattr(self, 'inference_engine') and self.inference_engine:
                logger.warning("🚨 执行强制页面表重置")
                
                for gpu_id in self.gpu_language_map.keys():
                    engine = self.inference_engine.get_engine(gpu_id)
                    if engine:
                        logger.warning(f"🚨 强制重置GPU {gpu_id} 的所有页面表")
                        engine.force_cleanup_all_sessions()
                    else:
                        logger.error(f"❌ 无法获取GPU {gpu_id} 的推理引擎进行重置")
                
                logger.info("✅ 强制页面表重置完成")
            else:
                logger.error("❌ 推理引擎不可用，无法执行强制重置")
                
        except Exception as e:
            logger.error(f"强制重置页面表时出错: {e}")
    
    def cleanup_session(self, user_id: str, language_id: str) -> bool:
        """手动清理指定的用户会话"""
        try:
            with self.session_lock:
                if language_id in self.user_sessions and user_id in self.user_sessions[language_id]:
                    session = self.user_sessions[language_id][user_id]
                    
                    logger.info(f"🧹 手动清理会话: {session.session_id}")
                    
                    # 清理KV cache页面
                    self._cleanup_session_pages(session)
                    
                    # 从会话字典中移除
                    del self.user_sessions[language_id][user_id]
                    self.stats['active_sessions'] -= 1
                    
                    logger.info(f"✅ 会话 {session.session_id} 清理完成")
                    return True
                else:
                    logger.warning(f"⚠️ 会话不存在: user_id={user_id}, language_id={language_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"手动清理会话时出错: {e}")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取所有会话的内存使用统计"""
        memory_stats = {
            'total_sessions': 0,
            'total_pages_used': 0,
            'sessions_by_language': {},
            'top_memory_users': [],
            'memory_distribution': {
                'speech_pages': 0,
                'llm_prefill_pages': 0,
                'llm_decode_pages': 0
            }
        }
        
        all_sessions = []
        
        with self.session_lock:
            for language_id, user_sessions in self.user_sessions.items():
                language_stats = {
                    'session_count': len(user_sessions),
                    'total_pages': 0,
                    'sessions': []
                }
                
                for user_id, session in user_sessions.items():
                    session_summary = session.get_memory_summary()
                    all_sessions.append(session_summary)
                    language_stats['sessions'].append(session_summary)
                    
                    pages = session_summary['memory_usage']['total_pages']
                    language_stats['total_pages'] += pages
                    memory_stats['total_pages_used'] += pages
                    
                    # 累计各类型页面使用
                    memory_stats['memory_distribution']['speech_pages'] += session_summary['memory_usage'].get('speech_pages', 0)
                    memory_stats['memory_distribution']['llm_prefill_pages'] += session_summary['memory_usage'].get('llm_prefill_pages', 0)
                    memory_stats['memory_distribution']['llm_decode_pages'] += session_summary['memory_usage'].get('llm_decode_pages', 0)
                
                memory_stats['sessions_by_language'][language_id] = language_stats
                memory_stats['total_sessions'] += language_stats['session_count']
        
        # 找出内存使用最多的会话
        memory_stats['top_memory_users'] = sorted(
            all_sessions, 
            key=lambda x: x['memory_usage']['total_pages'], 
            reverse=True
        )[:10]  # 前10个 