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
    
    # 🔥 添加：Beam search状态
    beam_state: Optional[Any] = None
    
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
        self.beam_state = None  # 🔥 添加：重置beam_state
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
    
    # 🔥 添加：Beam search状态
    beam_state: Optional[Any] = None
    
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
        
        # Update session with new speech data - 验证但不做滑动窗口
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
        
        # Update session state - 简化版，移除滑动窗口
        new_audio_data = speech_data.tolist() if speech_data.dim() == 1 else speech_data.flatten().tolist()
        session.source.extend(new_audio_data)
        session.source_finished = is_final
        session.last_activity = time.time()
        
        print(f"🔍 [SCHEDULER] Session source now has {len(session.source)} total samples ({len(session.source)/16000:.1f}s)")
        print(f"🔍 [SCHEDULER] Session src_len (already processed): {session.src_len} samples")
        print(f"🔍 [SCHEDULER] New samples to process: {len(session.source) - session.src_len}")
        
        # 🔍 估算页面使用量（仅用于诊断）
        audio_duration_s = len(session.source) / 16000
        estimated_speech_pages = max(1, int(audio_duration_s / 2))  # 估算：每2秒需要1个speech页面
        estimated_llm_pages = len(session.target) * 2  # 估算：每个翻译段落需要2个LLM页面
        total_estimated_pages = estimated_speech_pages + estimated_llm_pages
        
        print(f"📊 [SCHEDULER] 估算页面使用: Speech={estimated_speech_pages}, LLM={estimated_llm_pages}, 总计={total_estimated_pages}")
        print(f"📊 [SCHEDULER] 当前翻译段落数: {len(session.target)}")
        print(f"📊 [SCHEDULER] Engine cache管理: 依赖模型max_cache_size进行自动滑动窗口")
        
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
        
        # 🔥 简化：scheduler只提供placeholder，让inference engine调用model的_prepare_inputs处理
        # 这样保持了与原始infinisst_faster.policy()完全一致的行为
        input_ids = torch.tensor([[1]], dtype=torch.long)  # 简单的placeholder
        
        print(f"🔧 [SCHEDULER] 使用placeholder input_ids，inference engine将调用model._prepare_inputs")
        
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
            session_src_len=session.src_len,
            beam_state=session.beam_state
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
        
        # 🔥 添加：卡住检测计时器
        last_diagnosis_time = time.time()
        diagnosis_interval = 60  # 每60秒诊断一次
        
        while self.is_running:
            try:
                # Get batch of requests following the priority rule
                batch = self._get_request_batch(gpu_id)
                
                if not batch:
                    time.sleep(0.001)  
                    # 🔥 添加：在空闲时检查是否需要诊断
                    current_time = time.time()
                    if current_time - last_diagnosis_time > diagnosis_interval:
                        self._auto_diagnose_stuck_sessions(gpu_id)
                        last_diagnosis_time = current_time
                    continue
                
                # Process the batch
                self._process_batch(batch, gpu_id)
                
                # Clean up old sessions periodically
                if time.time() % 60 < 1:  # Every minute
                    self._cleanup_sessions()
                
                # 🔥 添加：定期诊断检查
                current_time = time.time()
                if current_time - last_diagnosis_time > diagnosis_interval:
                    self._auto_diagnose_stuck_sessions(gpu_id)
                    last_diagnosis_time = current_time
                
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
                    # 🔍 处理前记录页面池状态
                    print(f"📊 [SCHEDULER] GPU {gpu_id} 开始处理 {len(batch)} 个请求")
                    for i, req in enumerate(batch):
                        audio_len = req.speech_batch.shape[-1] if hasattr(req.speech_batch, 'shape') else len(req.speech_batch)
                        print(f"   - Request {i+1}: {audio_len} samples, stage={req.stage.value}")
                    
                    results = self.inference_engine.process_batch(gpu_id, batch)
                    
                    # 🔍 处理后记录结果
                    print(f"📊 [SCHEDULER] GPU {gpu_id} 完成处理，返回 {len(results)} 个结果")
                    
                    # 处理推理结果
                    for i, request in enumerate(batch):
                        if i < len(results):
                            result = results[i]
                            success = result.get('success', False)
                            error = result.get('error', 'None')
                            print(f"   - Request {i+1} 结果: success={success}, error={error}")
                            self._update_session_with_result(request, result)
                            logger.debug(f"Request {request.request_id} completed with inference engine")
                        else:
                            # 处理缺失的结果
                            print(f"   - Request {i+1} 缺失结果")
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
    
    def _update_session_with_result(self, request: InferenceRequest, result: Dict[str, Any]):
        """使用推理结果更新用户会话 - ORCA风格分步处理"""
        try:
            # 更新用户会话
            session = self.user_sessions[request.language_id][request.user_id]
            
            if result.get('success', False):
                # 🔥 ORCA风格：根据处理阶段更新状态
                prefill_finished = result.get('prefill_finished', False)
                decode_finished = result.get('decode_finished', False)
                
                if prefill_finished and not hasattr(request, '_prefill_done'):
                    # Prefill阶段刚完成
                    print(f"🔍 [ORCA-SCHEDULER] Request {request.request_id} prefill完成")
                    request._prefill_done = True
                    
                    # 将request状态切换到DECODE
                    request.stage = RequestStage.DECODE
                    
                    # Prefill阶段通常不生成最终文本，只是准备beam状态
                    generated_text = result.get('generated_text', '')
                    if generated_text:
                        print(f"🔍 [ORCA-SCHEDULER] Prefill生成初始文本: '{generated_text}'")
                    
                    # 🔥 关键修复：更新session和request的缓存状态
                    if 'speech_cache' in result:
                        session.speech_cache = result['speech_cache']
                        request.speech_cache = result['speech_cache']  # 🔥 同步更新request
                        print(f"🔍 [ORCA-CACHE] 更新speech_cache引用")
                    
                    if 'past_key_values' in result:
                        session.past_key_values = result['past_key_values']
                        request.past_key_values = result['past_key_values']  # 🔥 同步更新request
                        print(f"🔍 [ORCA-CACHE] 更新past_key_values引用")
                        
                    # 🔥 关键修复：保存beam_state
                    if hasattr(request, 'beam_state') and request.beam_state is not None:
                        session.beam_state = request.beam_state
                        print(f"🔍 [ORCA-CACHE] 保存beam_state到session")
                        
                    # 🔥 关键：将request重新放回DECODE队列继续处理
                    with self.queue_lock:
                        gpu_id = self.language_gpu_map[request.language_id]
                        self.decode_queues[gpu_id].append(request)
                        self.stats['queue_sizes'][gpu_id]['decode'] += 1
                        print(f"🔄 [ORCA-SCHEDULER] Request {request.request_id} 已放回DECODE队列 (cache已更新)")
                
                elif request.stage == RequestStage.DECODE:
                    # Decode阶段 - 生成了新的token
                    generated_text = result.get('generated_text', '')
                    generated_tokens = result.get('generated_tokens', [])
                    finished = result.get('finished', False)
                    
                    print(f"🔍 [ORCA-SCHEDULER] Decode step: '{generated_text}', finished={finished}")
                    
                    # 🔥 修复：累积式更新翻译历史
                    if generated_text:
                        # 获取当前完整翻译历史
                        is_chinese_translation = "Chinese" in request.language_id or "zh" in request.language_id.lower()
                        
                        if is_chinese_translation:
                            current_full_text = ''.join(session.target)
                        else:
                            current_full_text = ' '.join(session.target)
                        
                        # 🔥 关键修复：基于src_len判断是否为新音频片段
                        if generated_text.strip() != current_full_text.strip():
                            # 检查是否处理了新的音频数据
                            if not hasattr(session, 'last_processed_src_len'):
                                session.last_processed_src_len = 0
                            
                            current_src_len = request.session_src_len
                            is_new_audio_segment = current_src_len > session.last_processed_src_len
                            
                            if not session.target:
                                # 第一个翻译片段
                                session.target = [generated_text]
                                session.last_processed_src_len = current_src_len
                                print(f"🔍 [ORCA-SCHEDULER] 开始新翻译: '{generated_text}' (src_len: {current_src_len})")
                            elif is_new_audio_segment:
                                # 新的音频片段，添加新的翻译段落
                                session.target.append(generated_text)
                                session.last_processed_src_len = current_src_len
                                print(f"🔍 [ORCA-SCHEDULER] 新音频片段翻译: '{generated_text}' (src_len: {session.last_processed_src_len} -> {current_src_len})")
                                print(f"🔍 [ORCA-SCHEDULER] 翻译历史共 {len(session.target)} 个段落")
                            else:
                                # 同一音频片段的翻译扩展，替换最后一个翻译
                                session.target[-1] = generated_text
                                print(f"🔍 [ORCA-SCHEDULER] 扩展当前翻译: '{generated_text}' (同一音频片段, src_len: {current_src_len})")
                                print(f"🔍 [ORCA-SCHEDULER] 翻译历史共 {len(session.target)} 个段落")
                            
                            # 计算发送给前端的完整翻译
                            if is_chinese_translation:
                                new_full_text = ''.join(session.target)
                            else:
                                new_full_text = ' '.join(session.target)
                            
                            # 计算新增的内容（相对于上次发送的）
                            if current_full_text:
                                new_segment = new_full_text.replace(current_full_text, "").strip()
                            else:
                                new_segment = new_full_text.strip()
                            
                            result['new_segment'] = new_segment
                            result['segment_count'] = len(session.target)
                            result['full_translation'] = new_full_text
                        else:
                            print(f"🔍 [ORCA-SCHEDULER] 翻译未变化，跳过更新")
                            result['new_segment'] = ""
                            result['segment_count'] = len(session.target)
                            # 返回当前完整翻译历史
                            if is_chinese_translation:
                                result['full_translation'] = ''.join(session.target)
                            else:
                                result['full_translation'] = ' '.join(session.target)
                    
                    # 更新token序列
                    if generated_tokens:
                        session.target_ids = generated_tokens.copy()  # 完全替换
                        print(f"🔍 [ORCA-SCHEDULER] 更新token序列: {len(session.target_ids)} tokens")
                    
                    # 🔥 关键修复：更新session和request的缓存状态
                    if 'speech_cache' in result:
                        session.speech_cache = result['speech_cache']
                        request.speech_cache = result['speech_cache']  # 🔥 同步更新request
                        print(f"🔍 [ORCA-CACHE] Decode阶段更新speech_cache引用")
                    
                    if 'past_key_values' in result:
                        session.past_key_values = result['past_key_values']
                        request.past_key_values = result['past_key_values']  # 🔥 同步更新request
                        print(f"🔍 [ORCA-CACHE] Decode阶段更新past_key_values引用")
                    
                    # 🔥 关键修复：保存beam_state
                    if hasattr(request, 'beam_state') and request.beam_state is not None:
                        session.beam_state = request.beam_state
                        print(f"🔍 [ORCA-CACHE] 保存beam_state到session")
                        
                    # 🔥 关键：如果还没完成，继续放回DECODE队列
                    if not finished and not decode_finished:
                        with self.queue_lock:
                            gpu_id = self.language_gpu_map[request.language_id]
                            self.decode_queues[gpu_id].append(request)
                            self.stats['queue_sizes'][gpu_id]['decode'] += 1
                            print(f"🔄 [ORCA-SCHEDULER] Request {request.request_id} 继续DECODE，已重新入队 (cache已更新)")
                    else:
                        print(f"✅ [ORCA-SCHEDULER] Request {request.request_id} 翻译完成")
                        # 更新 src_len 到当前 session.source 的长度
                        session.src_len = len(session.source)
                        print(f"🔍 [ORCA-SCHEDULER] Final src_len updated to {session.src_len}")
            
            session.last_activity = time.time()
            
            # 🔥 关键：只在真正完成时才标记request完成和调用回调
            finished = result.get('finished', False) or result.get('decode_finished', False)
            
            if finished:
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
                print(f"📤 [ORCA-SCHEDULER] 发送最终结果到客户端: '{result.get('generated_text', '')}'")
            else:
                # 中间步骤，不调用回调，继续处理
                print(f"🔄 [ORCA-SCHEDULER] 中间步骤完成，继续处理...")
            
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
            
            # 🔥 添加：详细的队列诊断信息
            current_stats['detailed_queue_info'] = {}
            for gpu_id in self.gpu_language_map.keys():
                prefill_count = len(self.prefill_queues[gpu_id])
                decode_count = len(self.decode_queues[gpu_id])
                
                current_stats['detailed_queue_info'][gpu_id] = {
                    'language': self.gpu_language_map[gpu_id],
                    'prefill_queue_size': prefill_count,
                    'decode_queue_size': decode_count,
                    'total_queue_size': prefill_count + decode_count
                }
            
            # 🔥 添加：活跃session的最后活动时间检查
            current_time = time.time()
            inactive_sessions = []
            
            with self.session_lock:
                for language_id, user_sessions in self.user_sessions.items():
                    for user_id, session in user_sessions.items():
                        inactive_time = current_time - session.last_activity
                        if inactive_time > 60:  # 超过1分钟不活跃
                            inactive_sessions.append({
                                'session_id': session.session_id,
                                'user_id': user_id,
                                'language_id': language_id,
                                'inactive_seconds': inactive_time,
                                'source_length': len(session.source),
                                'target_segments': len(session.target)
                            })
            
            current_stats['inactive_sessions'] = inactive_sessions
            current_stats['inactive_session_count'] = len(inactive_sessions)
            
            return current_stats
    
    def diagnose_stuck_sessions(self) -> Dict[str, Any]:
        """🔍 诊断可能卡住的session"""
        current_time = time.time()
        stuck_sessions = []
        
        with self.session_lock:
            for language_id, user_sessions in self.user_sessions.items():
                for user_id, session in user_sessions.items():
                    inactive_time = current_time - session.last_activity
                    
                    # 检查是否可能卡住：超过30秒不活跃且有音频数据
                    if inactive_time > 30 and len(session.source) > 0:
                        gpu_id = self.language_gpu_map.get(language_id)
                        
                        with self.queue_lock:
                            prefill_queue_size = len(self.prefill_queues.get(gpu_id, []))
                            decode_queue_size = len(self.decode_queues.get(gpu_id, []))
                        
                        stuck_info = {
                            'session_id': session.session_id,
                            'user_id': user_id,
                            'language_id': language_id,
                            'gpu_id': gpu_id,
                            'inactive_seconds': inactive_time,
                            'source_length_samples': len(session.source),
                            'source_length_seconds': len(session.source) / 16000,
                            'target_segments': len(session.target),
                            'src_len_processed': session.src_len,
                            'unprocessed_samples': len(session.source) - session.src_len,
                            'prefill_queue_size': prefill_queue_size,
                            'decode_queue_size': decode_queue_size,
                            'total_queue_size': prefill_queue_size + decode_queue_size,
                            'has_speech_cache': session.speech_cache is not None,
                            'has_past_key_values': session.past_key_values is not None,
                            'has_beam_state': session.beam_state is not None
                        }
                        
                        stuck_sessions.append(stuck_info)
        
        diagnosis = {
            'timestamp': current_time,
            'stuck_sessions': stuck_sessions,
            'stuck_session_count': len(stuck_sessions),
            'analysis': []
        }
        
        # 分析原因
        for session in stuck_sessions:
            analysis = []
            
            if session['unprocessed_samples'] > 0:
                analysis.append(f"有 {session['unprocessed_samples']} 个样本未处理")
            
            if session['total_queue_size'] == 0:
                analysis.append("队列为空 - 可能没有新请求提交")
            elif session['total_queue_size'] > 10:
                analysis.append(f"队列积压严重 ({session['total_queue_size']} 个请求)")
            
            if not session['has_speech_cache']:
                analysis.append("缺少speech_cache")
            
            if not session['has_past_key_values']:
                analysis.append("缺少past_key_values")
            
            session['possible_causes'] = analysis
        
        return diagnosis
    
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
                    past_key_values=session.past_key_values
                    # 移除了 is_final 参数，因为 InferenceRequest 没有这个参数
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
    
    def _partial_page_cleanup(self, session: UserSession, request: InferenceRequest):
        """部分页面清理：释放一些不再需要的页面，但保持session活跃"""
        try:
            # 🔥 修复：不要模拟页面清理，避免状态不一致
            # 只记录需要清理，不实际修改内存使用统计
            
            # 🔥 策略1：只在音频历史真的过长时才进行清理
            SPEECH_CLEANUP_THRESHOLD_SECONDS = 60  # 提高到60秒阈值
            speech_samples_threshold = SPEECH_CLEANUP_THRESHOLD_SECONDS * session.source_sample_rate
            
            if len(session.source) > speech_samples_threshold:
                logger.info(f"🧹 [PARTIAL-CLEANUP] Session {session.session_id} 音频历史过长 ({len(session.source)/16000:.1f}s)，记录需要清理")
                
                # 🔥 修复：只调用真实的推理引擎清理，不模拟
                if hasattr(self, 'inference_engine') and self.inference_engine:
                    gpu_id = self.language_gpu_map.get(session.language_id)
                    if gpu_id is not None:
                        engine = self.inference_engine.get_engine(gpu_id)
                        if engine and hasattr(engine, '_partial_cleanup_speech_cache'):
                            # 只清理早期的speech cache，不修改session统计
                            cleanup_ratio = 0.1  # 减少到10%
                            engine._partial_cleanup_speech_cache(request, cleanup_ratio)
                            logger.info(f"🧹 [PARTIAL-CLEANUP] 调用引擎清理了 {cleanup_ratio*100}% 的speech cache页面")
                        else:
                            logger.info(f"🧹 [PARTIAL-CLEANUP] 推理引擎不支持部分清理，跳过")
                    else:
                        logger.warning(f"⚠️ [PARTIAL-CLEANUP] 无法找到GPU ID for language {session.language_id}")
                else:
                    logger.warning(f"⚠️ [PARTIAL-CLEANUP] 推理引擎不可用")
            
            # 🔥 修复：完全移除模拟的页面统计修改
            # 不再修改 session.memory_usage，避免状态不一致
            logger.debug(f"🔍 [PARTIAL-CLEANUP] Session {session.session_id} 当前内存使用保持不变: {session.memory_usage.get('total_pages', 0)} 页")
                
        except Exception as e:
            logger.error(f"部分页面清理时出错: {e}")
    
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
    
    def _auto_diagnose_stuck_sessions(self, gpu_id: int):
        """🔍 自动诊断当前GPU的卡住session"""
        try:
            language_id = self.gpu_language_map[gpu_id]
            diagnosis = self.diagnose_stuck_sessions()
            
            # 过滤只看当前GPU的session
            gpu_stuck_sessions = [
                session for session in diagnosis['stuck_sessions'] 
                if session['gpu_id'] == gpu_id
            ]
            
            if gpu_stuck_sessions:
                logger.warning(f"🚨 [AUTO-DIAGNOSIS] GPU {gpu_id} ({language_id}) 发现 {len(gpu_stuck_sessions)} 个可能卡住的session:")
                
                for session in gpu_stuck_sessions:
                    logger.warning(f"   - Session {session['session_id'][:8]}...")
                    logger.warning(f"     用户: {session['user_id']}")
                    logger.warning(f"     不活跃: {session['inactive_seconds']:.1f}s")
                    logger.warning(f"     未处理音频: {session['unprocessed_samples']} 样本")
                    logger.warning(f"     队列状态: P{session['prefill_queue_size']} + D{session['decode_queue_size']}")
                    
                    for cause in session['possible_causes']:
                        logger.warning(f"     可能原因: {cause}")
                
                # 🔥 添加：自动修复尝试
                self._attempt_auto_fix_stuck_sessions(gpu_stuck_sessions, gpu_id)
            else:
                logger.debug(f"✅ [AUTO-DIAGNOSIS] GPU {gpu_id} ({language_id}) 所有session正常运行")
                
        except Exception as e:
            logger.error(f"自动诊断时出错: {e}")
    
    def _attempt_auto_fix_stuck_sessions(self, stuck_sessions: List[Dict], gpu_id: int):
        """🔧 尝试自动修复卡住的session"""
        for session_info in stuck_sessions:
            try:
                session_id = session_info['session_id']
                user_id = session_info['user_id']
                language_id = session_info['language_id']
                
                logger.info(f"🔧 [AUTO-FIX] 尝试修复卡住的session {session_id[:8]}...")
                
                # 修复策略1：如果有未处理的音频数据，尝试重新提交请求
                if session_info['unprocessed_samples'] > 0:
                    logger.info(f"🔧 [AUTO-FIX] 检测到未处理音频，尝试重新生成请求...")
                    
                    # 获取session对象
                    with self.session_lock:
                        if language_id in self.user_sessions and user_id in self.user_sessions[language_id]:
                            session = self.user_sessions[language_id][user_id]
                            
                            # 创建一个新的prefill请求来处理未处理的音频
                            try:
                                import torch
                                
                                # 计算需要处理的音频片段
                                unprocessed_audio = session.source[session.src_len:]
                                if len(unprocessed_audio) > 160:  # 至少0.01秒的音频
                                    audio_tensor = torch.tensor(unprocessed_audio, dtype=torch.float32)
                                    
                                    # 创建一个临时的处理请求
                                    def temp_callback(result):
                                        logger.info(f"🔧 [AUTO-FIX] 自动修复请求完成: {result.get('success', False)}")
                                    
                                    request_id = self.submit_request(
                                        user_id=user_id,
                                        language_id=language_id,
                                        speech_data=audio_tensor,
                                        stage=RequestStage.PREFILL,
                                        is_final=False,
                                        max_new_tokens=10,
                                        result_callback=temp_callback
                                    )
                                    
                                    logger.info(f"🔧 [AUTO-FIX] 重新提交请求 {request_id} 处理 {len(unprocessed_audio)} 个未处理样本")
                                    
                            except Exception as e:
                                logger.error(f"🔧 [AUTO-FIX] 重新提交请求失败: {e}")
                
                # 修复策略2：如果队列为空但session有数据，可能是前端停止发送数据
                elif session_info['total_queue_size'] == 0 and session_info['source_length_samples'] > 0:
                    logger.warning(f"🔧 [AUTO-FIX] Session {session_id[:8]} 可能前端停止发送数据")
                    logger.warning(f"   建议检查前端WebSocket连接状态")
                
            except Exception as e:
                logger.error(f"🔧 [AUTO-FIX] 修复session {session_info['session_id'][:8]} 时出错: {e}") 