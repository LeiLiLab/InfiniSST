import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from threading import Lock, Thread
from enum import Enum
import json
import statistics

import torch
import numpy as np

logger = logging.getLogger(__name__)

class RequestStage(Enum):
    """Request processing stage"""
    PREFILL = "prefill"
    DECODE = "decode"

@dataclass
class CharacterDelay:
    """记录单个字符的延迟信息"""
    char: str
    segment_id: int
    char_index: int
    input_time: float
    output_time: float
    delay: float

@dataclass
class SegmentLog:
    """Simuleval兼容的segment日志"""
    segment_id: int
    src: str
    tgt: str
    tokens: List[str]
    delays: List[float]
    input_start_time: float
    output_time: float
    average_delay: float

class DelayTracker:
    """字符级延迟追踪器，用于计算streamLAAL"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.character_delays: List[CharacterDelay] = []
        self.segment_logs: List[SegmentLog] = []
        self.current_segment_id = 0
        self.current_input_buffer = ""
        self.current_input_start_time = 0.0
        self.char_input_times: List[float] = []  # 记录每个输入字符的时间
        
    def record_input_segment(self, text: str, timestamp: float):
        """记录输入segment和每个字符的输入时间"""
        self.current_input_buffer += text
        # 为新输入的每个字符记录时间戳
        for char in text:
            self.char_input_times.append(timestamp)
        
        if not self.current_input_start_time:
            self.current_input_start_time = timestamp
    
    def record_output_segment(self, output_text: str, timestamp: float, is_final: bool = False):
        """记录输出segment并计算字符级延迟"""
        if not output_text or not self.char_input_times:
            return
            
        # 分析输出文本的每个字符
        output_chars = list(output_text)
        delays = []
        tokens = []
        
        # 对于每个输出字符，计算与对应输入字符的延迟
        input_char_index = 0
        for i, output_char in enumerate(output_chars):
            # 找到对应的输入字符时间（简化匹配策略）
            if input_char_index < len(self.char_input_times):
                input_time = self.char_input_times[input_char_index]
                delay = timestamp - input_time
                
                char_delay = CharacterDelay(
                    char=output_char,
                    segment_id=self.current_segment_id,
                    char_index=i,
                    input_time=input_time,
                    output_time=timestamp,
                    delay=delay
                )
                
                self.character_delays.append(char_delay)
                delays.append(delay)
                tokens.append(output_char)
                
                input_char_index += 1
        
        # 创建segment日志
        if delays:
            segment_log = SegmentLog(
                segment_id=self.current_segment_id,
                src=self.current_input_buffer[:len(output_chars)],  # 对应的输入文本
                tgt=output_text,
                tokens=tokens,
                delays=delays,
                input_start_time=self.current_input_start_time,
                output_time=timestamp,
                average_delay=statistics.mean(delays)
            )
            
            self.segment_logs.append(segment_log)
            
            logger.info(f"🎯 [DELAY-TRACKER] Segment {self.current_segment_id}: {len(delays)} chars, avg delay: {segment_log.average_delay:.3f}s")
        
        if is_final:
            self.current_segment_id += 1
            self.current_input_buffer = ""
            self.current_input_start_time = 0.0
            self.char_input_times = []
    
    def calculate_stream_laal(self) -> float:
        """计算streamLAAL（所有字符延迟的平均值）"""
        if not self.character_delays:
            return 0.0
            
        all_delays = [cd.delay for cd in self.character_delays]
        stream_laal = statistics.mean(all_delays)
        
        logger.info(f"📊 [STREAM-LAAL] Session {self.session_id}: {stream_laal:.3f}s (from {len(all_delays)} characters)")
        return stream_laal
    
    def export_simuleval_log(self, filepath: str):
        """导出simuleval兼容的instance.log格式"""
        simuleval_data = []
        
        for segment_log in self.segment_logs:
            entry = {
                "segment_id": segment_log.segment_id,
                "src": segment_log.src,
                "tgt": segment_log.tgt,
                "tokens": segment_log.tokens,
                "delays": segment_log.delays,
                "input_start_time": segment_log.input_start_time,
                "output_time": segment_log.output_time,
                "average_delay": segment_log.average_delay
            }
            simuleval_data.append(entry)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for entry in simuleval_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        logger.info(f"📝 [EXPORT] Simuleval log exported to {filepath} ({len(simuleval_data)} segments)")
        return filepath
    
    def get_statistics(self, include_character_details: bool = False) -> Dict[str, Any]:
        """获取延迟统计信息"""
        if not self.character_delays:
            return {"stream_laal": 0.0, "total_characters": 0, "segments": 0}
            
        delays = [cd.delay for cd in self.character_delays]
        
        result = {
            "stream_laal": statistics.mean(delays),
            "min_delay": min(delays),
            "max_delay": max(delays),
            "median_delay": statistics.median(delays),
            "std_delay": statistics.stdev(delays) if len(delays) > 1 else 0.0,
            "total_characters": len(delays),
            "segments": len(self.segment_logs),
            "session_id": self.session_id
        }
        
        # 如果需要详细信息，包含字符级延迟数据
        if include_character_details:
            result["character_delays"] = [
                {
                    "char": cd.char,
                    "segment_id": cd.segment_id,
                    "char_index": cd.char_index,
                    "input_time": cd.input_time,
                    "output_time": cd.output_time,
                    "delay": cd.delay
                }
                for cd in self.character_delays
            ]
            
            result["segment_logs"] = [
                {
                    "segment_id": sl.segment_id,
                    "src": sl.src,
                    "tgt": sl.tgt,
                    "tokens": sl.tokens,
                    "delays": sl.delays,
                    "input_start_time": sl.input_start_time,
                    "output_time": sl.output_time,
                    "average_delay": sl.average_delay
                }
                for sl in self.segment_logs
            ]
        
        return result

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
    
    # 🔥 新增：延迟追踪
    delay_tracker: Optional[DelayTracker] = None
    evaluation_mode: bool = False  # 是否启用评估模式
    
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
    
    def __post_init__(self):
        # 初始化延迟追踪器
        if self.evaluation_mode:
            self.delay_tracker = DelayTracker(self.session_id)
    
    def enable_evaluation_mode(self):
        """启用评估模式，开始记录延迟"""
        self.evaluation_mode = True
        if not self.delay_tracker:
            self.delay_tracker = DelayTracker(self.session_id)
        logger.info(f"🎯 [EVAL] Session {self.session_id} evaluation mode enabled")
    
    def record_input(self, text: str, timestamp: Optional[float] = None):
        """记录输入文本用于延迟计算"""
        if self.delay_tracker:
            if timestamp is None:
                timestamp = time.time()
            self.delay_tracker.record_input_segment(text, timestamp)
    
    def record_output(self, text: str, timestamp: Optional[float] = None, is_final: bool = False):
        """记录输出文本并计算延迟"""
        if self.delay_tracker:
            if timestamp is None:
                timestamp = time.time()
            self.delay_tracker.record_output_segment(text, timestamp, is_final)
    
    def get_stream_laal(self) -> float:
        """获取当前session的streamLAAL"""
        if self.delay_tracker:
            return self.delay_tracker.calculate_stream_laal()
        return 0.0
    
    def export_delays(self, filepath: str) -> str:
        """导出延迟数据"""
        if self.delay_tracker:
            return self.delay_tracker.export_simuleval_log(filepath)
        return ""
    
    def get_delay_statistics(self, include_character_details: bool = False) -> Dict[str, Any]:
        """获取延迟统计"""
        if self.delay_tracker:
            return self.delay_tracker.get_statistics(include_character_details)
        return {"stream_laal": 0.0, "total_characters": 0, "segments": 0}

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
    
    beam_state: Optional[Any] = None
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # Higher number = higher priority
    
    # 🔥 动态属性支持
    retry_count: Optional[int] = None  # 重试次数
    queue_enter_time: Optional[float] = None  # 入队时间
    queue_exit_time: Optional[float] = None  # 出队时间
    queue_wait_time: Optional[float] = None  # 等待时间
    process_start_time: Optional[float] = None  # 处理开始时间
    
    # 🔥 添加动态属性支持
    def __post_init__(self):
        # 允许动态添加属性
        pass
    
    def __setattr__(self, name, value):
        # 允许动态设置属性
        super().__setattr__(name, value)
    
    # Result handling
    result_callback: Optional[Callable] = None
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
        
        # 🔥 Task 3: Dynamic Scheduling Configuration
        self.use_dynamic_schedule = getattr(args, 'use_dynamic_schedule', False) if args else False
        self.dynamic_wait_threshold = getattr(args, 'dynamic_wait_threshold', 0.05) if args else 0.05  # 50ms
        self.dynamic_batch_min_size = getattr(args, 'dynamic_batch_min_size', 1) if args else 1
        
        # FCFS queues - separate queues for each GPU/language
        self.prefill_queues: Dict[int, deque] = {gpu_id: deque() for gpu_id in gpu_language_map.keys()}
        self.decode_queues: Dict[int, deque] = {gpu_id: deque() for gpu_id in gpu_language_map.keys()}
        
        # 🔥 关键修改：细粒度锁 - 每个GPU的prefill和decode队列分别有独立的锁
        self.prefill_locks: Dict[int, Lock] = {gpu_id: Lock() for gpu_id in gpu_language_map.keys()}
        self.decode_locks: Dict[int, Lock] = {gpu_id: Lock() for gpu_id in gpu_language_map.keys()}
        
        # User session management
        self.user_sessions: Dict[str, Dict[str, UserSession]] = {}  # {language_id: {user_id: session}}
        
        # 🔥 Session锁保持独立，因为跨GPU访问
        self.session_lock = Lock()
        
        # Processing state
        self.is_running = False
        self.processing_threads: Dict[int, Thread] = {}  # One thread per GPU
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'active_sessions': 0,
            'queue_sizes': {gpu_id: {'prefill': 0, 'decode': 0} for gpu_id in gpu_language_map.keys()},
            # 🔥 Task 3: Dynamic Scheduling Statistics
            'dynamic_triggers': {
                'timeout_triggers': 0,
                'batch_size_triggers': 0,
                'total_dispatches': 0
            }
        }
        
        # 🔥 新增：队列监控统计
        self.queue_stats = {
            gpu_id: {
                'prefill': {
                    'total_processed': 0,
                    'total_wait_time': 0.0,
                    'total_process_time': 0.0,
                    'max_wait_time': 0.0,
                    'max_process_time': 0.0,
                    'avg_wait_time': 0.0,
                    'avg_process_time': 0.0,
                    'current_queue_size': 0,
                    'max_queue_size': 0,
                    'last_process_time': 0.0,
                    'throughput_per_sec': 0.0
                },
                'decode': {
                    'total_processed': 0,
                    'total_wait_time': 0.0,
                    'total_process_time': 0.0,
                    'max_wait_time': 0.0,
                    'max_process_time': 0.0,
                    'avg_wait_time': 0.0,
                    'avg_process_time': 0.0,
                    'current_queue_size': 0,
                    'max_queue_size': 0,
                    'last_process_time': 0.0,
                    'throughput_per_sec': 0.0
                }
            } for gpu_id in gpu_language_map.keys()
        }
        
        logger.info(f"LLMScheduler initialized with GPU mapping: {gpu_language_map}")
        logger.info(f"Max batch size: {self.max_batch_size}")
        logger.info(f"🔒 Fine-grained locks initialized: {len(self.prefill_locks)} prefill locks, {len(self.decode_locks)} decode locks")
    
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
    
    def get_or_create_session(self, user_id: str, language_id: str, session_id: Optional[str] = None, evaluation_mode: bool = False) -> UserSession:
        """Get existing session or create new one"""
        with self.session_lock:
            if language_id not in self.user_sessions:
                self.user_sessions[language_id] = {}
            
            if user_id not in self.user_sessions[language_id]:
                # 使用提供的session_id，如果没有提供则生成新的
                if session_id is None:
                    session_id = f"{user_id}_{language_id}_{int(time.time())}"
                
                session = UserSession(
                    user_id=user_id,
                    language_id=language_id,
                    session_id=session_id,
                    evaluation_mode=evaluation_mode  # 🔥 传递评估模式参数
                )
                self.user_sessions[language_id][user_id] = session
                
                # 🔥 无锁统计更新（容忍短暂不一致）
                self.stats['active_sessions'] += 1
                    
                logger.info(f"Created new session {session_id} for user {user_id}, language {language_id}, evaluation_mode={evaluation_mode}")
            else:
                session = self.user_sessions[language_id][user_id]
                session.last_activity = time.time()
                # 🔥 如果现有session的evaluation_mode与请求不同，更新它
                if evaluation_mode and not session.evaluation_mode:
                    session.enable_evaluation_mode()
                    logger.info(f"Enabled evaluation mode for existing session {session.session_id}")
            
            return self.user_sessions[language_id][user_id]
    
    def submit_request(self, 
                      user_id: str,
                      language_id: str,
                      speech_data: Union[torch.Tensor, np.ndarray, List[float]],
                      stage: RequestStage = RequestStage.PREFILL,
                      is_final: bool = False,
                      max_new_tokens: int = 20,
                      result_callback: Optional[Callable] = None,
                      api_session_id: Optional[str] = None,
                      evaluation_mode: bool = False) -> str:
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
            api_session_id: Session ID from API layer (optional, for consistency)
            
        Returns:
            request_id: Unique identifier for this request
        """
        # Validate language support
        if language_id not in self.language_gpu_map:
            raise ValueError(f"Unsupported language pair: {language_id}. Supported: {list(self.language_gpu_map.keys())}")
        
        gpu_id = self.language_gpu_map[language_id]
        
        # Get or create user session
        session = self.get_or_create_session(user_id, language_id, api_session_id, evaluation_mode)
        
        if isinstance(speech_data, (list, np.ndarray)):
            speech_data = torch.tensor(speech_data, dtype=torch.float32)
        elif not isinstance(speech_data, torch.Tensor):
            raise ValueError("speech_data must be list, numpy array, or torch tensor")
        
        session.source = speech_data.tolist() if speech_data.dim() == 1 else speech_data.flatten().tolist()
        session.source_finished = is_final
        session.last_activity = time.time()
        
        # 🎯 记录输入延迟（用于streamLAAL计算）
        input_timestamp = time.time()
        if session.evaluation_mode and session.delay_tracker:
            # 简化：将音频数据转换为模拟文本以便延迟计算
            input_text = f"[Audio segment {len(session.source)} samples]"
            session.record_input(input_text, input_timestamp)
            logger.info(f"🎯 [DELAY] Recorded input for session {session.session_id}: {len(session.source)} audio samples")

        
        # Prepare input data 
        request_id = str(uuid.uuid4())
        input_ids = torch.tensor([[1]], dtype=torch.long)  # 简单的placeholder

        request = InferenceRequest(
            request_id=request_id,
            user_id=user_id,
            language_id=language_id,
            session_id=session.session_id,
            stage=stage,
            speech_batch=session.source,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            speech_cache=session.speech_cache,
            past_key_values=session.past_key_values,
            result_callback=result_callback,
            # 🔥 传递会话状态信息
            segment_idx=session.segment_idx,
            translations_list=session.target,
            beam_state=session.beam_state
        )
        
        # 🔥 关键修改：使用细粒度锁添加到相应队列
        queue_enter_time = time.time()
        request.queue_enter_time = queue_enter_time  # 动态设置属性
        
        print(f"🔒 [FINE-LOCK] Submitting {stage.value} request to GPU {gpu_id}")
        
        if stage.name == RequestStage.PREFILL.name:
            # 🔥 只锁定prefill队列，不影响decode队列
            with self.prefill_locks[gpu_id]:
                self.prefill_queues[gpu_id].append(request)
                current_size = len(self.prefill_queues[gpu_id])
                
                # 🔥 更新队列统计（无锁，容忍短暂不一致）
                self.stats['queue_sizes'][gpu_id]['prefill'] += 1
                self.queue_stats[gpu_id]['prefill']['current_queue_size'] = current_size
                if current_size > self.queue_stats[gpu_id]['prefill']['max_queue_size']:
                    self.queue_stats[gpu_id]['prefill']['max_queue_size'] = current_size
                    print(f"📊 [QUEUE-STATS] GPU {gpu_id} Prefill队列新峰值: {current_size}")
                        
                print(f"🔒 [FINE-LOCK] ✅ Prefill request added to GPU {gpu_id}, queue size: {current_size}")
        else:
            # 🔥 只锁定decode队列，不影响prefill队列
            with self.decode_locks[gpu_id]:
                self.decode_queues[gpu_id].append(request)
                current_size = len(self.decode_queues[gpu_id])
                
                # 🔥 更新队列统计（无锁，容忍短暂不一致）
                self.stats['queue_sizes'][gpu_id]['decode'] += 1
                self.queue_stats[gpu_id]['decode']['current_queue_size'] = current_size
                if current_size > self.queue_stats[gpu_id]['decode']['max_queue_size']:
                    self.queue_stats[gpu_id]['decode']['max_queue_size'] = current_size
                    print(f"📊 [QUEUE-STATS] GPU {gpu_id} Decode队列新峰值: {current_size}")
                        
                print(f"🔒 [FINE-LOCK] ✅ Decode request added to GPU {gpu_id}, queue size: {current_size}")
        
        # 🔥 更新总统计（无锁，容忍短暂不一致）
        self.stats['total_requests'] += 1
        
        logger.info(f"🔒 [FINE-LOCK] Submitted {stage.value} request {request_id} for user {user_id}, language {language_id}, GPU {gpu_id}")
        return request_id
    
    def _processing_loop(self, gpu_id: int):
        """
        Main processing loop for a specific GPU
        Implements the scheduling policy: PREFILL queue has priority over DECODE queue
        """
        language_id = self.gpu_language_map[gpu_id]
        logger.info(f"Starting processing loop for GPU {gpu_id} (language: {language_id})")
        
        # 🔥 添加：队列状态报告计时器
        last_queue_report_time = time.time()
        queue_report_interval = 30  # 每30秒报告队列状态
        
        while self.is_running:
            try:
                # Get batch of requests following the priority rule
                batch = self._get_request_batch(gpu_id)
                
                if not batch:
                    time.sleep(0.001)  
                    # 🔥 添加：在空闲时检查是否需要诊断和报告
                    current_time = time.time()
                    
                    # 🔥 添加：周期性队列状态报告
                    if current_time - last_queue_report_time > queue_report_interval:
                        self._report_queue_performance(gpu_id)
                        last_queue_report_time = current_time
                    continue
                
                # 🔥 关键：process_batch不需要锁，因为每个GPU单线程处理
                self._process_batch(batch, gpu_id)
                
                # Clean up old sessions periodically
                if time.time() % 60 < 1:  # Every minute
                    self._cleanup_sessions()
                
                # 🔥 添加：定期诊断检查
                current_time = time.time()
                
            except Exception as e:
                logger.error(f"Error in processing loop for GPU {gpu_id}: {e}")
                
                # 🔥 添加：发生错误时自动打印诊断信息
                print(f"🚨 [SCHEDULER-ERROR] GPU {gpu_id} 处理循环发生错误，打印诊断信息:")
                self.print_diagnosis()
                
                time.sleep(0.1)  # Brief pause on error
        
        logger.info(f"Processing loop stopped for GPU {gpu_id}")
    
    def _get_request_batch(self, gpu_id: int) -> List[InferenceRequest]:
        """
        Get a HOMOGENEOUS batch of requests (either all PREFILL or all DECODE)
        
        🔥 关键改进：使用细粒度锁，减少锁竞争
        🔥 Task 3: Added dynamic scheduling support
        
        Scheduling Policy:
        1. If PREFILL queue has requests: Create pure PREFILL batch (up to 32 requests)
        2. If PREFILL queue is empty: Create pure DECODE batch (up to 32 requests)
        3. NEVER mix PREFILL and DECODE in the same batch
        4. Dynamic scheduling: Dispatch based on wait time or batch size thresholds
        """
        batch = []
        current_time = time.time()
        
        # 🔥 Task 3: Check dynamic scheduling conditions
        should_dispatch = False
        trigger_reason = None
        
        if self.use_dynamic_schedule:
            should_dispatch, trigger_reason = self._check_dynamic_dispatch_conditions(gpu_id, current_time)
            if should_dispatch:
                print(f"🚀 [DYNAMIC-SCHEDULE] GPU {gpu_id} dispatch triggered: {trigger_reason}")
        
        # 🔥 关键修改：先检查prefill队列（优先级更高）

        prefill_queue = self.prefill_queues[gpu_id]
        if prefill_queue:
            with self.prefill_locks[gpu_id]:
                print(f"🔒 [FINE-LOCK] GPU {gpu_id} checking prefill queue... {len(prefill_queue)}")
                # 有prefill请求，创建prefill batch
                batch_exit_time = time.time()
                while len(batch) < self.max_batch_size and prefill_queue:
                    try:
                        request = prefill_queue.popleft()
                        # 🔥 记录出队时间和等待时间
                        request.queue_exit_time = batch_exit_time
                        if hasattr(request, 'queue_enter_time'):
                            wait_time = batch_exit_time - request.queue_enter_time
                            request.queue_wait_time = wait_time
                            # 更新统计
                            stats = self.queue_stats[gpu_id]['prefill']
                            stats['total_wait_time'] += wait_time
                            if wait_time > stats['max_wait_time']:
                                stats['max_wait_time'] = wait_time
                        batch.append(request)
                        
                        # 🔥 更新队列大小统计（出队时减1）
                        self.stats['queue_sizes'][gpu_id]['prefill'] -= 1
                        self.queue_stats[gpu_id]['prefill']['current_queue_size'] = len(prefill_queue)
                    except IndexError:
                        # 队列为空，退出循环
                        print(f"⚠️ [SCHEDULER] Prefill queue empty during pop for GPU {gpu_id}")
                        break
                
                if batch:
                    assert all(req.stage.name == RequestStage.PREFILL.name for req in batch)
                    # 🔥 更新队列大小统计
                    self.queue_stats[gpu_id]['prefill']['current_queue_size'] = len(prefill_queue)
                    logger.info(f"🔒 [FINE-LOCK] Created PREFILL batch of size {len(batch)} for GPU {gpu_id}")
                    return batch
        
        decode_queue = self.decode_queues[gpu_id]
        if decode_queue:
            with self.decode_locks[gpu_id]:
                print(f"🔒 [FINE-LOCK] GPU {gpu_id} checking decode queue... {len(decode_queue)}")
                # 有decode请求，创建decode batch
                batch_exit_time = time.time()
                while len(batch) < self.max_batch_size and decode_queue:
                    try:
                        request = decode_queue.popleft()
                        # 🔥 记录出队时间和等待时间
                        request.queue_exit_time = batch_exit_time
                        if hasattr(request, 'queue_enter_time'):
                            wait_time = batch_exit_time - request.queue_enter_time
                            request.queue_wait_time = wait_time
                            # 更新统计
                            stats = self.queue_stats[gpu_id]['decode']
                            stats['total_wait_time'] += wait_time
                            if wait_time > stats['max_wait_time']:
                                stats['max_wait_time'] = wait_time
                        batch.append(request)
                        
                        # 🔥 更新队列大小统计（出队时减1）
                        self.stats['queue_sizes'][gpu_id]['decode'] -= 1
                        self.queue_stats[gpu_id]['decode']['current_queue_size'] = len(decode_queue)
                    except IndexError:
                        # 队列为空，退出循环
                        print(f"⚠️ [SCHEDULER] Decode queue empty during pop for GPU {gpu_id}")
                        break
                
                if batch:
                    assert all(req.stage.name == RequestStage.DECODE.name for req in batch)
                    # 🔥 更新队列大小统计
                    self.queue_stats[gpu_id]['decode']['current_queue_size'] = len(decode_queue)
                    logger.info(f"🔒 [FINE-LOCK] Created DECODE batch of size {len(batch)} for GPU {gpu_id}")
        
        print(f"🔒 [FINE-LOCK] GPU {gpu_id} no requests available, returning empty batch")
        return batch
    
    def _process_batch(self, batch: List[InferenceRequest], gpu_id: int):
        """
        Process a batch of requests using inference engine only (no simulation)
        """
        if not batch:
            return
        
        language_id = self.gpu_language_map[gpu_id]
        batch_stage = batch[0].stage.value if batch else "unknown"
        
        # 🔥 开始处理时间记录
        process_start_time = time.time()
        print(f"📊 [BATCH-TIMING] GPU {gpu_id} 开始处理 {batch_stage} batch: {len(batch)} 个请求")
        
        # 记录每个请求的处理开始时间和等待时间统计
        for i, request in enumerate(batch):
            request.is_processing = True
            request.process_start_time = process_start_time
            
            # 🔥 打印等待时间信息
            if hasattr(request, 'queue_wait_time') and request.queue_wait_time is not None:
                wait_time_ms = request.queue_wait_time * 1000
                print(f"   - Request {i+1}: 队列等待 {wait_time_ms:.1f}ms")
        
        logger.info(f"Processing batch of {len(batch)} requests on GPU {gpu_id} for language {language_id}")
        
        # 🔥 添加详细的诊断信息
        print(f"🔍 [SCHEDULER-DEBUG] Processing batch:")
        print(f"   - GPU {gpu_id}, Language: {language_id}")
        print(f"   - Batch size: {len(batch)}")
        print(f"   - Stage: {batch_stage}")
        print(f"   - Has inference_engine: {hasattr(self, 'inference_engine')}")
        if hasattr(self, 'inference_engine'):
            print(f"   - Inference_engine is None: {self.inference_engine is None}")
        
        try:

            if hasattr(self, 'inference_engine') and self.inference_engine:
                print(f"🔍 [SCHEDULER-DEBUG] 推理引擎可用，开始处理...")
                try:
                    # 🔍 处理前记录页面池状态
                    print(f"📊 [SCHEDULER] GPU {gpu_id} 开始处理 {len(batch)} 个请求")
                    for i, req in enumerate(batch):
                        audio_len = req.speech_batch.shape[-1] if hasattr(req.speech_batch, 'shape') else len(req.speech_batch)
                        print(f"   - Request {i+1}: {audio_len} samples, stage={req.stage.value}")
                    
                    batch_inference_start = time.time()
                    print(f"🔍 [SCHEDULER-DEBUG] 调用推理引擎 process_batch...")
                    results = self.inference_engine.process_batch(gpu_id, batch)
                    batch_inference_time = time.time() - batch_inference_start
                    
                    print(f"🔍 [SCHEDULER-DEBUG] 推理引擎调用完成，耗时: {batch_inference_time*1000:.1f}ms")
                    print(f"🔍 [SCHEDULER-DEBUG] 返回结果数量: {len(results) if results else 0}")
                    
                    # 🔍 处理后记录结果
                    print(f"📊 [SCHEDULER] GPU {gpu_id} 完成处理 [{batch_stage}]: {len(batch)} 个请求 → {len(results)} 个结果, 推理耗时: {batch_inference_time*1000:.1f}ms")
                    
                    # 处理推理结果
                    for i, request in enumerate(batch):
                        if i < len(results):
                            result = results[i]
                            success = result.get('success', False)
                            error = result.get('error', 'None')
                            print(f"   - Request {i+1} 结果: success={success}, error={error}")
                            if not success:
                                print(f"🔍 [SCHEDULER-DEBUG] Request {i+1} 失败详情: {result}")
                            self._update_session_with_result(request, result)
                            logger.info(f"Request {request.request_id} completed with inference engine")
                        else:
                            # 处理缺失的结果
                            print(f"   - Request {i+1} 缺失结果")
                            self._handle_failed_request(request, "Missing inference result")
                    
                except Exception as e:
                    logger.error(f"[ERROR] Inference engine failed for GPU {gpu_id}: {e}")
                    print(f"🔍 [SCHEDULER-DEBUG] 推理引擎异常详情:")
                    print(f"   - 异常类型: {type(e).__name__}")
                    print(f"   - 异常消息: {str(e)}")
                    import traceback
                    print(f"   - 异常堆栈: {traceback.format_exc()}")
                
                    # 其他错误：标记所有请求失败
                    for request in batch:
                        self._handle_failed_request(request, f"Inference engine error: {str(e)}")
            else:
                # 没有推理引擎可用
                print(f"🔍 [SCHEDULER-DEBUG] 推理引擎不可用!")
                logger.error(f"No inference engine available for GPU {gpu_id}")
                for request in batch:
                    self._handle_failed_request(request, "Inference engine not available")
                
        except Exception as e:
            logger.error(f"Batch processing failed on GPU {gpu_id}: {e}")
            # 处理所有请求的错误
            for request in batch:
                self._handle_failed_request(request, f"Batch processing failed: {str(e)}")
        
        # 🔥 处理完成后的时间统计
        process_end_time = time.time()
        total_process_time = process_end_time - process_start_time
        
        # 更新队列统计
        stage_stats = self.queue_stats[gpu_id][batch_stage]
        stage_stats['total_processed'] += len(batch)
        stage_stats['total_process_time'] += total_process_time
        stage_stats['last_process_time'] = total_process_time
        
        if total_process_time > stage_stats['max_process_time']:
            stage_stats['max_process_time'] = total_process_time
        
        # 计算平均值
        if stage_stats['total_processed'] > 0:
            stage_stats['avg_process_time'] = stage_stats['total_process_time'] / stage_stats['total_processed']
            stage_stats['avg_wait_time'] = stage_stats['total_wait_time'] / stage_stats['total_processed']
            
            # 计算吞吐量 (requests per second)
            if total_process_time > 0:
                stage_stats['throughput_per_sec'] = len(batch) / total_process_time
        
        # 🔥 打印详细的性能统计
        print(f"📊 [BATCH-TIMING] GPU {gpu_id} {batch_stage} batch完成:")
        print(f"   - 批处理耗时: {total_process_time*1000:.1f}ms")
        print(f"   - 平均每请求: {(total_process_time/len(batch))*1000:.1f}ms")
        print(f"   - 吞吐量: {len(batch)/total_process_time:.1f} req/s")
        print(f"   - 累计处理: {stage_stats['total_processed']} 个{batch_stage}请求")
        print(f"   - 平均处理时间: {stage_stats['avg_process_time']*1000:.1f}ms")
        print(f"   - 平均等待时间: {stage_stats['avg_wait_time']*1000:.1f}ms")
        print(f"   - 最大处理时间: {stage_stats['max_process_time']*1000:.1f}ms")
        print(f"   - 最大等待时间: {stage_stats['max_wait_time']*1000:.1f}ms")
    
    
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
                    gpu_id = self.language_gpu_map[request.language_id]
                    with self.decode_locks[gpu_id]:
                        self.decode_queues[gpu_id].append(request)
                        self.stats['queue_sizes'][gpu_id]['decode'] += 1
                    print(f"🔄 [ORCA-SCHEDULER] Request {request.request_id} 已放回DECODE队列 (cache已更新)")
                
                elif request.stage.name == RequestStage.DECODE.name:
                    # Decode阶段 - 生成了新的token
                    generated_text = result.get('generated_text', '')
                    generated_tokens = result.get('generated_tokens', [])
                    finished = result.get('finished', False)
                    
                    print(f"🔍 [ORCA-SCHEDULER] Decode step: '{generated_text}', finished={finished}")

                    is_chinese_translation = "Chinese" in request.language_id or "zh" in request.language_id.lower()
                    
                    if finished and generated_text:
                        session.target.append(generated_text)
                        
                        # 🎯 记录输出延迟（用于streamLAAL计算）
                        output_timestamp = time.time()
                        if session.evaluation_mode and session.delay_tracker:
                            session.record_output(generated_text, output_timestamp, is_final=True)
                            logger.info(f"🎯 [DELAY] Recorded output for session {session.session_id}: {len(generated_text)} chars")
                    
                    if is_chinese_translation:
                        new_full_text = ''.join(session.target)
                    else:
                        new_full_text = ' '.join(session.target)
                    result['full_translation'] = new_full_text

                    
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
                        gpu_id = self.language_gpu_map[request.language_id]
                        with self.decode_locks[gpu_id]:
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
    
    def diagnose_scheduler_status(self) -> Dict[str, Any]:
        """🔍 诊断调度器状态，帮助排查问题"""
        diagnosis = {
            'timestamp': time.time(),
            'scheduler_running': self.is_running,
            'gpu_language_map': self.gpu_language_map,
            'inference_engine_status': {},
            'queue_status': {},
            'session_status': {},
            'thread_status': {}
        }
        
        # 检查推理引擎状态
        diagnosis['inference_engine_status'] = {
            'has_inference_engine': hasattr(self, 'inference_engine'),
            'inference_engine_is_none': not hasattr(self, 'inference_engine') or self.inference_engine is None,
            'engine_type': type(self.inference_engine).__name__ if hasattr(self, 'inference_engine') and self.inference_engine else 'None'
        }
        
        if hasattr(self, 'inference_engine') and self.inference_engine:
            try:
                # 检查多GPU推理引擎的状态
                if hasattr(self.inference_engine, 'engines'):
                    engine_details = {}
                    for gpu_id in self.gpu_language_map.keys():
                        engine = self.inference_engine.get_engine(gpu_id)
                        engine_details[gpu_id] = {
                            'engine_exists': engine is not None,
                            'is_loaded': engine.is_loaded if engine else False,
                            'is_running': engine.is_running if engine else False,
                            'language': self.gpu_language_map[gpu_id]
                        }
                    diagnosis['inference_engine_status']['gpu_engines'] = engine_details
                else:
                    diagnosis['inference_engine_status']['single_engine'] = {
                        'is_loaded': getattr(self.inference_engine, 'is_loaded', False),
                        'is_running': getattr(self.inference_engine, 'is_running', False)
                    }
            except Exception as e:
                diagnosis['inference_engine_status']['engine_check_error'] = str(e)
        
        # 检查队列状态
        for gpu_id in self.gpu_language_map.keys():
            diagnosis['queue_status'][gpu_id] = {
                'language': self.gpu_language_map[gpu_id],
                'prefill_queue_size': len(self.prefill_queues[gpu_id]),
                'decode_queue_size': len(self.decode_queues[gpu_id]),
                'has_prefill_lock': gpu_id in self.prefill_locks,
                'has_decode_lock': gpu_id in self.decode_locks
            }
        
        # 检查会话状态
        with self.session_lock:
            diagnosis['session_status'] = {
                'total_languages': len(self.user_sessions),
                'languages': list(self.user_sessions.keys()),
                'total_sessions': sum(len(sessions) for sessions in self.user_sessions.values()),
                'sessions_by_language': {
                    lang: len(sessions) for lang, sessions in self.user_sessions.items()
                }
            }
        
        # 检查处理线程状态
        diagnosis['thread_status'] = {
            'total_threads': len(self.processing_threads),
            'thread_details': {
                gpu_id: {
                    'thread_alive': thread.is_alive(),
                    'thread_name': thread.name
                } for gpu_id, thread in self.processing_threads.items()
            }
        }
        
        return diagnosis
    
    def print_diagnosis(self):
        """🔍 打印详细的诊断信息"""
        diagnosis = self.diagnose_scheduler_status()
        
        print("🔍 [SCHEDULER-DIAGNOSIS] 调度器状态诊断:")
        print(f"   📊 调度器运行状态: {diagnosis['scheduler_running']}")
        print(f"   🖥️  GPU语言映射: {diagnosis['gpu_language_map']}")
        
        # 推理引擎状态
        engine_status = diagnosis['inference_engine_status']
        print(f"   🤖 推理引擎状态:")
        print(f"      - 有推理引擎: {engine_status['has_inference_engine']}")
        print(f"      - 引擎为空: {engine_status['inference_engine_is_none']}")
        print(f"      - 引擎类型: {engine_status['engine_type']}")
        
        if 'gpu_engines' in engine_status:
            print(f"      - GPU引擎详情:")
            for gpu_id, details in engine_status['gpu_engines'].items():
                print(f"        GPU {gpu_id} ({details['language']}): 存在={details['engine_exists']}, 已加载={details['is_loaded']}, 运行中={details['is_running']}")
        
        # 队列状态
        print(f"   📥 队列状态:")
        for gpu_id, queue_info in diagnosis['queue_status'].items():
            print(f"      GPU {gpu_id} ({queue_info['language']}): Prefill={queue_info['prefill_queue_size']}, Decode={queue_info['decode_queue_size']}")
        
        # 会话状态
        session_status = diagnosis['session_status']
        print(f"   👥 会话状态: {session_status['total_sessions']} 个会话，{session_status['total_languages']} 种语言")
        
        # 线程状态
        thread_status = diagnosis['thread_status']
        print(f"   🧵 处理线程: {thread_status['total_threads']} 个线程")
        for gpu_id, thread_info in thread_status['thread_details'].items():
            print(f"      GPU {gpu_id}: 线程存活={thread_info['thread_alive']}")
        
        return diagnosis
    
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
        """Get current queue and system statistics with detailed performance monitoring"""
        # 🔥 无锁读取基础统计（容忍短暂不一致）
        current_stats = self.stats.copy()
        current_stats['gpu_language_map'] = self.gpu_language_map.copy()
        current_stats['timestamp'] = time.time()
        
        # 🔥 添加：详细的队列诊断信息（无锁读取）
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
        
        # 🔥 新增：详细的队列性能统计（无锁读取）
        current_stats['queue_performance'] = {}
        for gpu_id in self.gpu_language_map.keys():
            gpu_stats = {}
            for stage in ['prefill', 'decode']:
                stage_stats = self.queue_stats[gpu_id][stage].copy()
                
                # 转换时间单位为毫秒以便阅读
                stage_stats['avg_wait_time_ms'] = stage_stats['avg_wait_time'] * 1000
                stage_stats['avg_process_time_ms'] = stage_stats['avg_process_time'] * 1000
                stage_stats['max_wait_time_ms'] = stage_stats['max_wait_time'] * 1000
                stage_stats['max_process_time_ms'] = stage_stats['max_process_time'] * 1000
                stage_stats['last_process_time_ms'] = stage_stats['last_process_time'] * 1000
                
                gpu_stats[stage] = stage_stats
            
            current_stats['queue_performance'][gpu_id] = gpu_stats
        
        # 🔥 添加：活跃session的最后活动时间检查（只在访问sessions时用锁）
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
    
    
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language pairs"""
        return list(self.language_gpu_map.keys())

    
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
        
        return memory_stats
    
    def _report_queue_performance(self, gpu_id: int):
        """周期性报告队列性能统计"""
        try:
            # 🔥 无锁读取队列大小
            prefill_count = len(self.prefill_queues[gpu_id])
            decode_count = len(self.decode_queues[gpu_id])
            
            # 只在有活动时报告
            if prefill_count == 0 and decode_count == 0:
                prefill_stats = self.queue_stats[gpu_id]['prefill']
                decode_stats = self.queue_stats[gpu_id]['decode']
                
                # 如果没有处理过任何请求，不报告
                if prefill_stats['total_processed'] == 0 and decode_stats['total_processed'] == 0:
                    return
            
            language = self.gpu_language_map[gpu_id]
            print(f"📊 [QUEUE-REPORT] GPU {gpu_id} ({language}) 队列状态:")
            print(f"   📥 当前队列: Prefill={prefill_count}, Decode={decode_count}")
            
            # Prefill性能统计
            prefill_stats = self.queue_stats[gpu_id]['prefill']
            if prefill_stats['total_processed'] > 0:
                print(f"   🔥 Prefill性能:")
                print(f"      - 累计处理: {prefill_stats['total_processed']} 个请求")
                print(f"      - 平均等待: {prefill_stats['avg_wait_time']*1000:.1f}ms")
                print(f"      - 平均处理: {prefill_stats['avg_process_time']*1000:.1f}ms")
                print(f"      - 最大等待: {prefill_stats['max_wait_time']*1000:.1f}ms")
                print(f"      - 最大处理: {prefill_stats['max_process_time']*1000:.1f}ms")
                print(f"      - 当前队列大小: {prefill_stats['current_queue_size']}")
                print(f"      - 峰值队列大小: {prefill_stats['max_queue_size']}")
                if prefill_stats.get('throughput_per_sec', 0) > 0:
                    print(f"      - 吞吐量: {prefill_stats['throughput_per_sec']:.1f} req/s")
            
            # Decode性能统计
            decode_stats = self.queue_stats[gpu_id]['decode']
            if decode_stats['total_processed'] > 0:
                print(f"   🔄 Decode性能:")
                print(f"      - 累计处理: {decode_stats['total_processed']} 个请求")
                print(f"      - 平均等待: {decode_stats['avg_wait_time']*1000:.1f}ms")
                print(f"      - 平均处理: {decode_stats['avg_process_time']*1000:.1f}ms")
                print(f"      - 最大等待: {decode_stats['max_wait_time']*1000:.1f}ms")
                print(f"      - 最大处理: {decode_stats['max_process_time']*1000:.1f}ms")
                print(f"      - 当前队列大小: {decode_stats['current_queue_size']}")
                print(f"      - 峰值队列大小: {decode_stats['max_queue_size']}")
                if decode_stats.get('throughput_per_sec', 0) > 0:
                    print(f"      - 吞吐量: {decode_stats['throughput_per_sec']:.1f} req/s")
            
            # 队列积压警告
            if prefill_count > 5:
                print(f"⚠️  [QUEUE-WARNING] Prefill队列积压: {prefill_count} 个请求")
            if decode_count > 10:
                print(f"⚠️  [QUEUE-WARNING] Decode队列积压: {decode_count} 个请求")
                
        except Exception as e:
            logger.error(f"Error reporting queue performance for GPU {gpu_id}: {e}") 
    
    # 🔥 Task 3: Dynamic Scheduling Methods
    
    def _check_dynamic_dispatch_conditions(self, gpu_id: int, current_time: float) -> Tuple[bool, Optional[str]]:
        """
        检查动态调度的触发条件
        
        Returns:
            (should_dispatch, trigger_reason)
        """
        if not self.use_dynamic_schedule:
            return False, None
        
        # 条件1：检查批处理大小
        prefill_size = len(self.prefill_queues[gpu_id])
        decode_size = len(self.decode_queues[gpu_id])
        total_size = prefill_size + decode_size
        
        if total_size >= self.max_batch_size:
            return True, "batch_size_threshold"
        
        # 条件2：检查等待时间
        oldest_wait_time = self._get_oldest_request_wait_time(gpu_id, current_time)
        if oldest_wait_time is not None and oldest_wait_time > self.dynamic_wait_threshold:
            return True, f"wait_time_threshold_{oldest_wait_time:.3f}s"
        
        # 条件3：检查最小批处理大小
        if total_size >= self.dynamic_batch_min_size:
            # 如果有最小数量的请求，可以考虑dispatch
            return True, f"min_batch_size_{total_size}"
        
        return False, None
    
    def _get_oldest_request_wait_time(self, gpu_id: int, current_time: float) -> Optional[float]:
        """获取队列中最老请求的等待时间"""
        oldest_time = None
        
        # 检查prefill队列
        prefill_queue = self.prefill_queues[gpu_id]
        if prefill_queue:
            first_request = prefill_queue[0]
            if hasattr(first_request, 'queue_enter_time') and first_request.queue_enter_time:
                wait_time = current_time - first_request.queue_enter_time
                oldest_time = wait_time
        
        # 检查decode队列
        decode_queue = self.decode_queues[gpu_id]
        if decode_queue:
            first_request = decode_queue[0]
            if hasattr(first_request, 'queue_enter_time') and first_request.queue_enter_time:
                wait_time = current_time - first_request.queue_enter_time
                if oldest_time is None or wait_time > oldest_time:
                    oldest_time = wait_time
        
        return oldest_time
    
    def _log_dynamic_dispatch(self, gpu_id: int, trigger_reason: str, prefill_size: int, decode_size: int):
        """记录动态调度触发事件"""
        timestamp = time.time()
        
        # 更新统计
        if "batch_size" in trigger_reason:
            self.stats['dynamic_triggers']['batch_size_triggers'] += 1
        elif "wait_time" in trigger_reason:
            self.stats['dynamic_triggers']['timeout_triggers'] += 1
        
        # 记录详细日志
        logger.info(f"🚀 [DYNAMIC-SCHEDULE] GPU {gpu_id} dispatch triggered: {trigger_reason}")
        logger.info(f"   - Timestamp: {timestamp}")
        logger.info(f"   - Prefill queue size: {prefill_size}")
        logger.info(f"   - Decode queue size: {decode_size}")
        logger.info(f"   - Total batch size: {prefill_size + decode_size}")
        logger.info(f"   - Batch size triggers: {self.stats['dynamic_triggers']['batch_size_triggers']}")
        logger.info(f"   - Timeout triggers: {self.stats['dynamic_triggers']['timeout_triggers']}")
        logger.info(f"   - Total dispatches: {self.stats['dynamic_triggers']['total_dispatches']}")
        
        # 可选：写入专门的调度日志文件
        try:
            log_entry = {
                "timestamp": timestamp,
                "gpu_id": gpu_id,
                "trigger_reason": trigger_reason,
                "prefill_queue_size": prefill_size,
                "decode_queue_size": decode_size,
                "total_batch_size": prefill_size + decode_size,
                "cumulative_stats": self.stats['dynamic_triggers'].copy()
            }
            
            # 写入调度日志文件 (可选)
            schedule_log_path = f"dynamic_schedule_gpu_{gpu_id}.log"
            with open(schedule_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.warning(f"Failed to write dynamic schedule log: {e}")
    
    def get_dynamic_schedule_stats(self) -> Dict[str, Any]:
        """获取动态调度统计信息"""
        stats = {
            "enabled": self.use_dynamic_schedule,
            "configuration": {
                "wait_threshold_ms": self.dynamic_wait_threshold * 1000,
                "min_batch_size": self.dynamic_batch_min_size,
                "max_batch_size": self.max_batch_size
            },
            "triggers": self.stats['dynamic_triggers'].copy(),
            "per_gpu_status": {}
        }
        
        # 添加每个GPU的当前状态
        current_time = time.time()
        for gpu_id in self.gpu_language_map.keys():
            prefill_size = len(self.prefill_queues[gpu_id])
            decode_size = len(self.decode_queues[gpu_id])
            oldest_wait = self._get_oldest_request_wait_time(gpu_id, current_time)
            
            should_dispatch, trigger_reason = self._check_dynamic_dispatch_conditions(gpu_id, current_time)
            
            stats["per_gpu_status"][gpu_id] = {
                "language": self.gpu_language_map[gpu_id],
                "prefill_queue_size": prefill_size,
                "decode_queue_size": decode_size,
                "total_queue_size": prefill_size + decode_size,
                "oldest_wait_time_ms": oldest_wait * 1000 if oldest_wait else None,
                "should_dispatch": should_dispatch,
                "trigger_reason": trigger_reason
            }
        
        return stats