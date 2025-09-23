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
class WordDelay:
    """记录单个单词的延迟信息"""
    word: str
    segment_id: int
    word_index: int
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
    """多语言延迟追踪器，支持字符级和单词级对齐，用于计算streamLAAL"""
    
    def __init__(self, session_id: str, language_id: str):
        self.session_id = session_id
        self.language_id = language_id
        
        # 🔥 根据语言确定对齐类型：中文用字符级，意大利语等用单词级
        self.is_character_based = self._is_character_based_language(language_id)
        
        # 通用延迟记录（统一接口）
        self.unit_delays: List[Union[CharacterDelay, WordDelay]] = []
        self.segment_logs: List[SegmentLog] = []
        self.current_segment_id = 0
        self.current_input_start_time = 0.0
        
        # 🔥 修复：添加输入segment记录
        self.input_segments: List[Dict[str, Any]] = []
        
        # 🔥 新增：WebSocket连接时间作为基础时间
        self.websocket_start_time: Optional[float] = None
        
        # 根据语言类型选择不同的输入时间记录策略
        if self.is_character_based:
            self.char_input_times: List[float] = []  # 记录每个输入字符的时间
        else:
            self.word_input_times: List[float] = []  # 记录每个输入单词的时间
            self.input_words: List[str] = []  # 记录输入的单词序列
    
    def _is_character_based_language(self, language_id: str) -> bool:
        """判断语言是否为字符级对齐（中文）还是单词级对齐（意大利语等）"""
        # 中文相关的语言对使用字符级对齐
        chinese_patterns = ['chinese', 'zh', 'Chinese', 'Chinese']
        return any(pattern in language_id for pattern in chinese_patterns)
        
    def record_input_segment(self, text: str, timestamp: float):
        """记录输入segment和每个单元的输入时间（字符或单词）"""
        # 🔥 修复：不累积buffer，每个segment独立处理
        # 保存当前segment的输入信息用于后续匹配
        current_segment_text = text
        
        if self.is_character_based:
            # 🔥 中文：字符级处理
            for char in current_segment_text:
                self.char_input_times.append(timestamp)
        else:
            # 🔥 意大利语等：单词级处理
            import re
            # 提取单词（忽略标点符号）
            words = re.findall(r'\b\w+\b', current_segment_text)
            for word in words:
                self.word_input_times.append(timestamp)
                self.input_words.append(word)
        
        # 🔥 修复：记录当前segment的文本，用于后续输出匹配
        if not hasattr(self, 'input_segments'):
            self.input_segments = []
        self.input_segments.append({
            'text': current_segment_text,
            'timestamp': timestamp,
            'segment_id': len(self.input_segments)
        })
        
        if not self.current_input_start_time:
            self.current_input_start_time = timestamp
    
    def record_output_segment(self, output_text: str, timestamp: float, is_final: bool = False):
        """记录输出segment并计算延迟（字符级或单词级）"""
        if not output_text:
            return
            
        delays = []
        tokens = []
        
        # 🔥 关键修复：获取当前segment对应的输入时间
        current_segment_input_time = self.current_input_start_time
        relative_input_time = None  # 相对于WebSocket连接的时间
        
        if hasattr(self, 'input_segments') and self.input_segments:
            # 使用当前segment对应的输入时间
            segment_index = min(self.current_segment_id, len(self.input_segments) - 1)
            current_segment_input_time = self.input_segments[segment_index]['timestamp']
            
            # 🔥 计算相对于WebSocket连接的时间
            if self.websocket_start_time:
                relative_input_time = current_segment_input_time - self.websocket_start_time
                logger.debug(f"🎯 [DELAY-FIX] Segment {self.current_segment_id}: input_time={current_segment_input_time}, relative_time={relative_input_time:.3f}s")
            else:
                logger.debug(f"🎯 [DELAY-FIX] Segment {self.current_segment_id}: using input_time={current_segment_input_time} from input_segments[{segment_index}]")
        else:
            logger.warning(f"⚠️ [DELAY-FIX] Segment {self.current_segment_id}: No input_segments available, using current_input_start_time={current_segment_input_time}")
        
        if self.is_character_based:
            # 🔥 中文：字符级处理 - 修复延迟计算
            output_chars = list(output_text)
            
            for i, output_char in enumerate(output_chars):
                # 🔥 关键修复：所有字符使用同一个segment的输入时间
                delay = timestamp - current_segment_input_time
                
                char_delay = CharacterDelay(
                    char=output_char,
                    segment_id=self.current_segment_id,
                    char_index=i,
                    input_time=current_segment_input_time,
                    output_time=timestamp,
                    delay=delay
                )
                
                self.unit_delays.append(char_delay)
                delays.append(delay)
                tokens.append(output_char)
            
            # 🔥 改进日志：显示相对时间和绝对延迟
            if self.websocket_start_time and relative_input_time is not None:
                relative_output_time = timestamp - self.websocket_start_time
                logger.info(f"🎯 [DELAY-TRACKER-CHAR] Segment {self.current_segment_id}: {len(delays)} chars")
                logger.info(f"   - Input relative time: {relative_input_time:.3f}s")
                logger.info(f"   - Output relative time: {relative_output_time:.3f}s") 
                logger.info(f"   - Delay: {delays[0]:.3f}s")
            else:
                logger.info(f"🎯 [DELAY-TRACKER-CHAR] Segment {self.current_segment_id}: {len(delays)} chars, delay: {delays[0]:.3f}s")
        
        else:
            # 🔥 意大利语等：单词级处理 - 修复延迟计算
            import re
            output_words = re.findall(r'\b\w+\b', output_text)
            
            for i, output_word in enumerate(output_words):
                # 🔥 关键修复：所有单词使用同一个segment的输入时间
                delay = timestamp - current_segment_input_time
                
                word_delay = WordDelay(
                    word=output_word,
                    segment_id=self.current_segment_id,
                    word_index=i,
                    input_time=current_segment_input_time,
                    output_time=timestamp,
                    delay=delay
                )
                
                self.unit_delays.append(word_delay)
                delays.append(delay)
                tokens.append(output_word)
            
            logger.info(f"🎯 [DELAY-TRACKER-WORD] Segment {self.current_segment_id}: {len(delays)} words, delay: {delays[0]:.3f}s (all same)")
        
        # 创建segment日志
        if delays:
            # 🔥 修复：使用当前segment对应的输入文本，确保src信息完整
            src_text = ""
            if hasattr(self, 'input_segments') and self.input_segments:
                # 🔥 改进：取当前segment对应的输入文本
                segment_index = min(self.current_segment_id, len(self.input_segments) - 1)
                src_text = self.input_segments[segment_index]['text']
                logger.debug(f"🎯 [SRC-MATCH] Segment {self.current_segment_id}: using input_segments[{segment_index}] = '{src_text}'")
            else:
                # 🔥 后备方案：使用合理的音频segment标识
                src_text = f"[Audio segment {self.current_segment_id}] (no input recorded)"
                logger.warning(f"⚠️ [SRC-FALLBACK] Segment {self.current_segment_id}: no input_segments available, using fallback")
                
            segment_log = SegmentLog(
                segment_id=self.current_segment_id,
                src=src_text,
                tgt=output_text,
                tokens=tokens,
                delays=delays,
                input_start_time=current_segment_input_time,  # 🔥 修复：使用正确的输入时间
                output_time=timestamp,
                average_delay=statistics.mean(delays)
            )
            
            self.segment_logs.append(segment_log)
        
        if is_final:
            self.current_segment_id += 1
            # 🔥 修复：不重置input_start_time，让下一个segment使用正确的输入时间
            # self.current_input_start_time = 0.0  # 注释掉这行，避免重置为0
            
            # 🔥 注意：不再清空输入时间数组，因为可能有多个输出segment对应同一输入
            # 只有在真正需要重置时才清空
    
    def calculate_stream_laal(self) -> float:
        """计算streamLAAL（所有延迟单元的平均值）"""
        if not self.unit_delays:
            return 0.0
            
        all_delays = [unit.delay for unit in self.unit_delays]
        stream_laal = statistics.mean(all_delays)
        
        unit_type = "characters" if self.is_character_based else "words"
        logger.info(f"📊 [STREAM-LAAL] Session {self.session_id}: {stream_laal:.3f}s (from {len(all_delays)} {unit_type})")
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
    
    def get_statistics(self, include_unit_details: bool = False) -> Dict[str, Any]:
        """获取延迟统计信息"""
        if not self.unit_delays:
            unit_type = "characters" if self.is_character_based else "words"
            return {
                "stream_laal": 0.0, 
                f"total_{unit_type}": 0, 
                "segments": 0,
                "language_type": unit_type
            }
            
        delays = [unit.delay for unit in self.unit_delays]
        unit_type = "characters" if self.is_character_based else "words"
        
        result = {
            "stream_laal": statistics.mean(delays),
            "min_delay": min(delays),
            "max_delay": max(delays),
            "median_delay": statistics.median(delays),
            "std_delay": statistics.stdev(delays) if len(delays) > 1 else 0.0,
            f"total_{unit_type}": len(delays),
            "segments": len(self.segment_logs),
            "session_id": self.session_id,
            "language_type": unit_type,
            "language_id": self.language_id
        }
        
        # 🔥 向后兼容：为字符级语言保留total_characters字段
        if self.is_character_based:
            result["total_characters"] = len(delays)
        
        # 如果需要详细信息，包含单元级延迟数据
        if include_unit_details:
            if self.is_character_based:
                result["character_delays"] = [
                    {
                        "char": unit.char,
                        "segment_id": unit.segment_id,
                        "char_index": unit.char_index,
                        "input_time": unit.input_time,
                        "output_time": unit.output_time,
                        "delay": unit.delay
                    }
                    for unit in self.unit_delays
                ]
            else:
                result["word_delays"] = [
                    {
                        "word": unit.word,
                        "segment_id": unit.segment_id,
                        "word_index": unit.word_index,
                        "input_time": unit.input_time,
                        "output_time": unit.output_time,
                        "delay": unit.delay
                    }
                    for unit in self.unit_delays
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
    prefill_can_enter: bool = True
    
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
    
    # 🔥 新增：WebSocket连接时间作为基础时间
    websocket_start_time: Optional[float] = None  # WebSocket连接建立的时间
    
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
        
        # 🔥 清理：移除冗余的内存日志，只在必要时记录
        if total > 0 and self.memory_usage['allocation_count'] % 20 == 0:
            logger.debug(f"SESSION-MEMORY {self.session_id}: Total={total} pages, Peak={self.memory_usage.get('peak_pages', 0)} pages")
    
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
            self.delay_tracker = DelayTracker(self.session_id, self.language_id)
    
    def enable_evaluation_mode(self, websocket_start_time: Optional[float] = None):
        """启用评估模式，开始记录延迟"""
        self.evaluation_mode = True
        if not self.delay_tracker:
            self.delay_tracker = DelayTracker(self.session_id, self.language_id)
        
        # 🔥 设置WebSocket连接时间作为基础时间
        if websocket_start_time:
            self.websocket_start_time = websocket_start_time
            self.delay_tracker.websocket_start_time = websocket_start_time
            logger.info(f"🎯 [EVAL] Session {self.session_id} evaluation mode enabled with WebSocket start time: {websocket_start_time}")
        else:
            self.websocket_start_time = time.time()
            self.delay_tracker.websocket_start_time = self.websocket_start_time
            logger.info(f"🎯 [EVAL] Session {self.session_id} evaluation mode enabled with current time as base")
    
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
    
    def get_delay_statistics(self, include_unit_details: bool = False) -> Dict[str, Any]:
        """获取延迟统计"""
        if self.delay_tracker:
            return self.delay_tracker.get_statistics(include_unit_details)
        unit_type = "characters" if "Chinese" in self.language_id else "words"
        return {
            "stream_laal": 0.0, 
            f"total_{unit_type}": 0, 
            "segments": 0, 
            "language_type": unit_type
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
    
    beam_state: Optional[Any] = None
    
    # 🔥 修正：每个request独立的decode步骤序列
    decode_step: int = 0  # 当前request的decode步骤（0=第一步）
    max_decode_steps: int = 20  # 最大decode步骤数
    request_decode_id: str = ""  # 标识同一chunk的decode sequence
    
    session: Optional[UserSession] = None

    
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
        if not self.request_decode_id:
            # 生成唯一的decode序列ID
            self.request_decode_id = f"{self.session_id}_{self.request_id}_{int(time.time()*1000)}"
        
        # 🔥 确保统计相关的属性有合理的默认值
        if self.retry_count is None:
            self.retry_count = 0
        if self.queue_enter_time is None:
            self.queue_enter_time = self.timestamp  # 使用创建时间作为默认值
    
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
        self.session_timeout = getattr(args, 'session_timeout', 3600) if args else 3600  # 5 minutes
        
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
        
        # 🔥 优化：使用hash-based锁池，避免全局锁瓶颈
        self.session_lock_pool_size = 256  # 锁池大小
        self.session_lock_pool = [Lock() for _ in range(self.session_lock_pool_size)]
        
        # Processing state
        self.is_running = False
        self.processing_threads: Dict[int, Thread] = {}  # One thread per GPU
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,  # 🔥 新增：失败请求计数
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
        logger.info(f"🔒 Hash-based session lock pool: {self.session_lock_pool_size} locks (no global lock contention)")
    
    def _get_session_lock(self, session_id: str) -> Lock:
        """使用hash-based锁池，避免全局锁竞争"""
        # 🔥 关键优化：使用session_id的hash选择锁，无需全局锁
        lock_index = hash(session_id) % self.session_lock_pool_size
        return self.session_lock_pool[lock_index]
    
    def _cleanup_session_lock(self, session_id: str):
        """Hash-based锁池无需清理，保留接口兼容性"""
        # 🔥 优化：hash-based锁池是预分配的，无需清理
        # 保留此方法是为了代码兼容性，实际不执行任何操作
        pass
    
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
    
    def get_or_create_session(self, user_id: str, language_id: str, session_id: Optional[str] = None, evaluation_mode: bool = False, websocket_start_time: Optional[float] = None) -> UserSession:
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
                    evaluation_mode=evaluation_mode,  # 🔥 传递评估模式参数
                    websocket_start_time=websocket_start_time  # 🔥 传递WebSocket连接时间
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
                    session.enable_evaluation_mode(websocket_start_time)
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
                      evaluation_mode: bool = False,
                      websocket_start_time: Optional[float] = None) -> str:
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
        session = self.get_or_create_session(user_id, language_id, api_session_id, evaluation_mode, websocket_start_time)
        
        if isinstance(speech_data, (list, np.ndarray)):
            speech_data = torch.tensor(speech_data, dtype=torch.float32)
        elif not isinstance(speech_data, torch.Tensor):
            raise ValueError("speech_data must be list, numpy array, or torch tensor")
        
        session.source = speech_data.tolist() if speech_data.dim() == 1 else speech_data.flatten().tolist()
        session.source_finished = is_final
        session.last_activity = time.time()
        
        # 🎯 记录输入延迟（用于streamLAAL计算）- 使用音频segment标识
        input_timestamp = time.time()
        if session.evaluation_mode and session.delay_tracker:
            # 🔥 修复：使用简洁的音频segment标识，避免重复累积
            audio_duration_seconds = len(session.source) / session.source_sample_rate if session.source_sample_rate > 0 else 0
            current_segment_id = len(session.delay_tracker.input_segments) if hasattr(session.delay_tracker, 'input_segments') else session.segment_idx
            
            # 创建清晰的输入文本标识
            input_text = f"[Audio segment {current_segment_id}] ({audio_duration_seconds:.1f}s)"
            
            session.record_input(input_text, input_timestamp)
            logger.info(f"🎯 [DELAY-INPUT] Recorded input for session {session.session_id}: {input_text}")
            
            # 🔥 新增：详细的音频处理状态日志
            logger.info(f"📊 [AUDIO-STATUS] Session {session.session_id}:")
            logger.info(f"   - Current segment: {current_segment_id}")
            logger.info(f"   - Audio samples: {len(session.source)}")
            logger.info(f"   - Audio duration: {audio_duration_seconds:.2f}s")
            logger.info(f"   - Is final: {is_final}")
            logger.info(f"   - Source finished: {session.source_finished}")
            logger.info(f"   - Stage: {stage.value}")
            logger.info(f"   - Total input segments recorded: {len(session.delay_tracker.input_segments) if hasattr(session.delay_tracker, 'input_segments') else 0}")

        
        # Prepare input data 
        request_id = str(uuid.uuid4())
        input_ids = torch.tensor([[1]], dtype=torch.long)  # 简单的placeholder

        # 🔥 关键：正确设置decode步骤信息
        decode_step = 0
        request_decode_id = ""
        
        # 对于所有新的request，都从decode_step=0开始
        # 🔥 修正：不再依赖session的decode步骤，每个request独立
        if stage == RequestStage.DECODE:
            # 每个decode request都有独立的decode序列
            request_decode_id = f"{session.session_id}_{request_id}_{int(time.time()*1000)}"
            # print(f"🔍 [DECODE-ORDER] 创建独立DECODE请求: {request_decode_id}, decode_step=0")
        else:
            # PREFILL阶段不需要decode_id
            request_decode_id = ""

        request = InferenceRequest(
            request_id=request_id,
            user_id=user_id,
            language_id=language_id,
            session_id=session.session_id,
            session=session,
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
            beam_state=session.beam_state,
            # 🔥 关键：设置decode步骤信息
            decode_step=decode_step,
            max_decode_steps=max_new_tokens,  # 使用max_new_tokens作为最大步骤数
            request_decode_id=request_decode_id
        )
        
        # 🔥 关键修改：使用细粒度锁添加到相应队列
        queue_enter_time = time.time()
        request.queue_enter_time = queue_enter_time  # 动态设置属性
        
        # 🔥 额外防护：确保queue_enter_time不为None
        if request.queue_enter_time is None:
            request.queue_enter_time = queue_enter_time
            logger.warning(f"Fixed None queue_enter_time for request {request_id}")
        
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
                    logger.info(f"GPU {gpu_id} Prefill队列新峰值: {current_size}")
                        
                logger.debug(f"Prefill request added to GPU {gpu_id}, queue size: {current_size}")
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
                    logger.info(f"GPU {gpu_id} Decode队列新峰值: {current_size}")
                        
                logger.debug(f"Decode request added to GPU {gpu_id}, queue size: {current_size}")
        
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
        🔥 性能优化：智能session过滤，最大化并发性能
        
        Scheduling Policy:
        1. If PREFILL queue has requests: Create pure PREFILL batch (up to 32 requests)
        2. If PREFILL queue is empty: Create pure DECODE batch (up to 32 requests)  
        3. NEVER mix PREFILL and DECODE in the same batch
        
        🔥 关键修正：session同步策略
        4a. PREFILL阶段：每个session只能有一个request per batch (因为依赖session状态)
        4b. DECODE阶段：同session可以有多个requests per batch (基于缓存，独立处理)
        
        5. Dynamic scheduling: Dispatch based on wait time or batch size thresholds
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
            # 🔥 性能优化：只使用GPU级别的队列锁，避免全局锁
            with self.prefill_locks[gpu_id]:
                print(f"🔒 [FINE-LOCK] GPU {gpu_id} checking prefill queue... {len(prefill_queue)}")
                # 有prefill请求，创建prefill batch
                batch_exit_time = time.time()
                need_add_back = []
                session_in_batch = set()  # 🔥 关键：跟踪batch中已有的session
                
                # 🔥 智能session过滤：确保batch中每个session只有一个request
                while len(batch) < self.max_batch_size and prefill_queue:
                    try:
                        request = prefill_queue.popleft()
                        
                        # 🔥 第一层过滤：检查session是否已在当前batch中
                        if request.session_id in session_in_batch:
                            logger.debug(f"Session {request.session_id} 已在batch中，放回前面保持顺序")
                            need_add_back.append(request)
                            continue
                        
                        # 🔥 第二层过滤：尝试获取session锁（无阻塞）
                        session_lock = self._get_session_lock(request.session_id)
                        can_process = False
                        skip_reason = "unknown"
                        
                        # 🔥 关键：使用trylock避免阻塞，确保调度器高效运行
                        if session_lock.acquire(blocking=False):  # 非阻塞尝试获取锁
                            try:
                                if request.session and request.session.prefill_can_enter:
                                    request.session.prefill_can_enter = False  # 原子设置
                                    can_process = True
                                    session_in_batch.add(request.session_id)  # 标记session已在batch
                                else:
                                    skip_reason = "prefill_can_enter=False"
                            finally:
                                session_lock.release()  # 立即释放锁
                        else:
                            skip_reason = "session_lock_busy"
                        
                        if not can_process:
                            logger.debug(f"请求 {request.request_id}, session {request.session_id} 暂时无法处理，原因: {skip_reason}")
                            need_add_back.append(request)
                            continue

                        # 🔥 记录出队时间和等待时间
                        request.queue_exit_time = batch_exit_time
                        if hasattr(request, 'queue_enter_time') and request.queue_enter_time is not None:
                            wait_time = batch_exit_time - request.queue_enter_time
                            request.queue_wait_time = wait_time
                            # 🔥 额外防护：确保wait_time是有效数字
                            if wait_time is not None and isinstance(wait_time, (int, float)) and wait_time >= 0:
                                # 更新统计
                                stats = self.queue_stats[gpu_id]['prefill']
                                stats['total_wait_time'] += wait_time
                                if wait_time > stats['max_wait_time']:
                                    stats['max_wait_time'] = wait_time
                            else:
                                logger.warning(f"Invalid wait_time {wait_time} for request {request.request_id}")
                        batch.append(request)
                        
                        # 🔥 更新队列大小统计（出队时减1）
                        self.stats['queue_sizes'][gpu_id]['prefill'] -= 1
                        self.queue_stats[gpu_id]['prefill']['current_queue_size'] = len(prefill_queue)
                    except IndexError:
                        # 队列为空，退出循环
                        logger.warning(f"Prefill queue empty during pop for GPU {gpu_id}")
                        break

                # 🔥 关键修复：将无法处理的请求放回队列前面，保持原有顺序
                if need_add_back:
                    # 🔥 逆序放回，确保原始顺序保持不变
                    for req in reversed(need_add_back):
                        prefill_queue.appendleft(req)  # 放到队列前面
                    logger.debug(f"将 {len(need_add_back)} 个请求放回前面 (保持session顺序)")

                if batch:
                    assert all(req.stage.name == RequestStage.PREFILL.name for req in batch)
                    # 验证batch中每个session都是唯一的
                    session_ids_in_batch = [req.session_id for req in batch]
                    assert len(session_ids_in_batch) == len(set(session_ids_in_batch)), "Batch contains duplicate sessions!"
                    
                    # 🔥 更新队列大小统计
                    self.queue_stats[gpu_id]['prefill']['current_queue_size'] = len(prefill_queue)
                    logger.info(f"Created PREFILL batch: {len(batch)} requests from {len(session_ids_in_batch)} unique sessions")
                    return batch
        
        decode_queue = self.decode_queues[gpu_id]
        if decode_queue:
            with self.decode_locks[gpu_id]:
                logger.debug(f"GPU {gpu_id} checking decode queue... {len(decode_queue)}")
                # 有decode请求，创建decode batch
                batch_exit_time = time.time()
                
                # 🔥 关键修正：Decode阶段不需要session过滤！
                # Decode requests基于缓存独立处理，同session的多个decode可以并行
                while len(batch) < self.max_batch_size and decode_queue:
                    try:
                        request = decode_queue.popleft()
                        
                        # 🔥 记录出队时间和等待时间
                        request.queue_exit_time = batch_exit_time
                        if hasattr(request, 'queue_enter_time') and request.queue_enter_time is not None:
                            wait_time = batch_exit_time - request.queue_enter_time
                            request.queue_wait_time = wait_time
                            # 🔥 额外防护：确保wait_time是有效数字
                            if wait_time is not None and isinstance(wait_time, (int, float)) and wait_time >= 0:
                                # 更新统计
                                stats = self.queue_stats[gpu_id]['decode']
                                stats['total_wait_time'] += wait_time
                                if wait_time > stats['max_wait_time']:
                                    stats['max_wait_time'] = wait_time
                            else:
                                logger.warning(f"Invalid wait_time {wait_time} for request {request.request_id}")
                        batch.append(request)
                        
                        # 🔥 更新队列大小统计（出队时减1）
                        self.stats['queue_sizes'][gpu_id]['decode'] -= 1
                        self.queue_stats[gpu_id]['decode']['current_queue_size'] = len(decode_queue)
                    except IndexError:
                        # 队列为空，退出循环
                        logger.warning(f"Decode queue empty during pop for GPU {gpu_id}")
                        break
                
                if batch:
                    assert all(req.stage.name == RequestStage.DECODE.name for req in batch)
                    
                    # 🔥 修正：Decode batch允许相同session的多个requests
                    session_ids_in_batch = [req.session_id for req in batch]
                    unique_sessions = len(set(session_ids_in_batch))
                    
                    # 🔥 更新队列大小统计
                    self.queue_stats[gpu_id]['decode']['current_queue_size'] = len(decode_queue)
                    logger.info(f"Created DECODE batch: {len(batch)} requests from {unique_sessions} sessions")
        
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
        logger.debug(f"GPU {gpu_id} 开始处理 {batch_stage} batch: {len(batch)} 个请求")
        
        # 记录每个请求的处理开始时间和等待时间统计
        for i, request in enumerate(batch):
            request.is_processing = True
            request.process_start_time = process_start_time
        
        logger.info(f"Processing batch of {len(batch)} requests on GPU {gpu_id} for language {language_id}")
        
        try:
            # 🔥 详细的推理引擎状态检查
            has_inference_engine = hasattr(self, 'inference_engine')
            engine_not_none = has_inference_engine and self.inference_engine is not None
            
            logger.info(f"🔍 [BATCH-DEBUG] GPU {gpu_id} 推理引擎检查:")
            logger.info(f"   - hasattr(self, 'inference_engine'): {has_inference_engine}")
            logger.info(f"   - self.inference_engine is not None: {engine_not_none}")
            
            if engine_not_none:
                logger.info(f"   - Engine type: {type(self.inference_engine).__name__}")
                logger.info(f"   - Engine engines count: {len(getattr(self.inference_engine, 'engines', {}))}")
            
            if has_inference_engine and self.inference_engine:
                logger.info(f"🚀 GPU {gpu_id} 推理引擎可用，开始处理 {len(batch)} 个请求...")
                
                # 🔥 记录每个请求的详细信息
                for i, request in enumerate(batch):
                    logger.info(f"   Request {i+1}: {request.request_id}")
                    logger.info(f"     - Stage: {request.stage.value}")
                    logger.info(f"     - Session ID: {request.session_id}")
                    logger.info(f"     - Session evaluation_mode: {request.session.evaluation_mode if request.session else 'None'}")
                    logger.info(f"     - Speech data size: {len(request.speech_batch) if hasattr(request.speech_batch, '__len__') else 'N/A'}")
                
                try:
                    batch_inference_start = time.time()
                    logger.info(f"🎯 [INFERENCE] Calling inference_engine.process_batch for GPU {gpu_id}...")
                    results = self.inference_engine.process_batch(gpu_id, batch)
                    batch_inference_time = time.time() - batch_inference_start
                    
                    logger.info(f"✅ [INFERENCE] GPU {gpu_id} 推理引擎调用完成，耗时: {batch_inference_time*1000:.1f}ms")
                    logger.info(f"📊 [INFERENCE] 返回结果数量: {len(results) if results else 0}")
                    
                    # 🔥 详细记录每个结果
                    if results:
                        for i, result in enumerate(results):
                            logger.info(f"📝 [RESULT-{i+1}] Result details:")
                            logger.info(f"     - Success: {result.get('success', False)}")
                            logger.info(f"     - Error: {result.get('error', 'None')}")
                            logger.info(f"     - Generated text: '{result.get('generated_text', '')}'")
                            logger.info(f"     - Generated tokens: {len(result.get('generated_tokens', []))}")
                            logger.info(f"     - Finished: {result.get('finished', False)}")
                            logger.info(f"     - Full translation: '{result.get('full_translation', '')}'")
                    else:
                        logger.error(f"❌ [INFERENCE] 推理引擎返回空结果!")
                    
                    # 处理推理结果
                    for i, request in enumerate(batch):
                        if i < len(results):
                            result = results[i]
                            success = result.get('success', False)
                            logger.info(f"🔄 [PROCESSING] Processing result {i+1} for request {request.request_id}")
                            if not success:
                                logger.warning(f"⚠️ Request {i+1} failed: {result.get('error', 'Unknown error')}")
                            self._update_session_with_result(request, result)
                        else:
                            # 处理缺失的结果
                            logger.warning(f"❌ Request {i+1} missing result")
                            self._handle_failed_request(request, "Missing inference result")
                    
                except Exception as e:
                    logger.error(f"Inference engine failed for GPU {gpu_id}: {e}")
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
        
        # 🔥 简化：只在必要时记录性能统计
        if stage_stats['total_processed'] % 10 == 0:  # 每处理10批次记录一次
            logger.info(f"GPU {gpu_id} {batch_stage} 批处理统计: {stage_stats['total_processed']} 次, "
                       f"平均时间 {stage_stats['avg_process_time']*1000:.1f}ms, "
                       f"吞吐量 {stage_stats.get('throughput_per_sec', 0):.1f} req/s")
    
    
    def _update_session_with_result(self, request: InferenceRequest, result: Dict[str, Any]):
        """使用推理结果更新用户会话 - ORCA风格分步处理，request级别的独立decode"""
        try:
            # 更新用户会话
            session = request.session
            
            # 🔥 更新session的内存使用统计（从beam_search中收集的信息）
            self._update_session_memory_from_result(session, result)
            
            if result.get('success', False):
                # 🔥 ORCA风格：根据处理阶段更新状态
                prefill_finished = result.get('prefill_finished', False)
                decode_finished = result.get('decode_finished', False)
                
                if prefill_finished and not hasattr(request, '_prefill_done'):
                    # Prefill阶段刚完成
                    logger.debug(f"Request {request.request_id} prefill完成")
                    request._prefill_done = True
                    
                    # 将request状态切换到DECODE
                    request.stage = RequestStage.DECODE
                    
                    # 🔥 关键：初始化独立的decode步骤
                    request.decode_step = 0  # 从第一步开始
                    if not request.request_decode_id:
                        request.request_decode_id = f"{session.session_id}_{request.request_id}_{int(time.time()*1000)}"
                    
                    # 🔥 关键修复：更新session和request的缓存状态
                    if 'speech_cache' in result:
                        session.speech_cache = result['speech_cache']
                        request.speech_cache = result['speech_cache']  # 🔥 同步更新request
                    
                    if 'past_key_values' in result:
                        session.past_key_values = result['past_key_values']
                        request.past_key_values = result['past_key_values']  # 🔥 同步更新request
                        
                    # 🔥 关键修复：保存beam_state
                    if hasattr(request, 'beam_state') and request.beam_state is not None:
                        session.beam_state = request.beam_state
                        
                    # 🔥 关键：将request重新放回DECODE队列继续处理
                    gpu_id = self.language_gpu_map[request.language_id]
                    with self.decode_locks[gpu_id]:
                        self.decode_queues[gpu_id].append(request)
                        self.stats['queue_sizes'][gpu_id]['decode'] += 1
                    logger.debug(f"Request {request.request_id} 已放回DECODE队列")
                
                elif request.stage.name == RequestStage.DECODE.name:
                    # Decode阶段 - 生成了新的token
                    generated_text = result.get('generated_text', '')
                    generated_tokens = result.get('generated_tokens', [])
                    finished = result.get('finished', False)
                    
                    is_chinese_translation = "Chinese" in request.language_id or "zh" in request.language_id.lower()
                    
                    # 🎯 修复：只记录完成的输出结果（finished=True），包括空结果
                    if finished and session.evaluation_mode and session.delay_tracker:
                        output_timestamp = time.time()
                        logger.info(f"🎯 [DELAY-RECORD] Recording finished output for session {session.session_id}: '{generated_text}' (empty={'empty' if not generated_text else 'non-empty'})")
                        session.record_output(generated_text, output_timestamp, is_final=True)
                    
                    if finished and generated_text:
                        session.target.append(generated_text)
                        # 🔥 新增：segment计数日志
                        segment_count = len(session.target)
                        logger.info(f"📊 [SEGMENT-COUNT] Session {session.session_id}: Added segment {segment_count}, total segments: {segment_count}")
                        
                        # 🔥 每10个segment报告一次详细统计
                        if segment_count % 10 == 0:
                            logger.info(f"📊 [SEGMENT-MILESTONE] Session {session.session_id}: Reached {segment_count} segments")
                            logger.info(f"   - Source length: {len(session.source)}")
                            logger.info(f"   - Source finished: {session.source_finished}")
                            logger.info(f"   - Segment index: {session.segment_idx}")
                            if session.delay_tracker:
                                logger.info(f"   - Delay tracker segments: {len(session.delay_tracker.segment_logs)}")
                                logger.info(f"   - Input segments recorded: {len(session.delay_tracker.input_segments) if hasattr(session.delay_tracker, 'input_segments') else 0}")

                    # 🔥 关键：request级别的decode步骤管理
                    if finished:
                        # 🔥 整个decode过程完成，释放session
                        session_lock = self._get_session_lock(session.session_id)
                        with session_lock:
                            session.prefill_can_enter = True
                    else:
                        # 🔥 当前decode步骤完成，准备下一步
                        request.decode_step += 1  # 递增到下一步
                        next_step = request.decode_step
                        
                        # 检查是否超过最大步骤数
                        if next_step >= request.max_decode_steps:
                            logger.debug(f"Request {request.request_id} 达到最大decode步骤 {request.max_decode_steps}，强制完成")
                            finished = True
                            # 释放session
                            session_lock = self._get_session_lock(session.session_id)
                            with session_lock:
                                session.prefill_can_enter = True
                    
                    if is_chinese_translation:
                        new_full_text = ''.join(session.target)
                    else:
                        new_full_text = ' '.join(session.target)
                    result['full_translation'] = new_full_text

                    
                    # 更新token序列
                    if generated_tokens:
                        session.target_ids = generated_tokens.copy()  # 完全替换
                        # print(f"🔍 [ORCA-SCHEDULER] 更新token序列: {len(session.target_ids)} tokens")
                    
                    # 🔥 关键修复：更新session和request的缓存状态
                    if 'speech_cache' in result:
                        session.speech_cache = result['speech_cache']
                        request.speech_cache = result['speech_cache']  # 🔥 同步更新request
                    
                    if 'past_key_values' in result:
                        session.past_key_values = result['past_key_values']
                        request.past_key_values = result['past_key_values']  # 🔥 同步更新request
                    
                    # 🔥 关键修复：保存beam_state
                    if hasattr(request, 'beam_state') and request.beam_state is not None:
                        session.beam_state = request.beam_state
                        
                    # 🔥 关键：如果还没完成，继续放回DECODE队列
                    if not finished and not decode_finished:
                        gpu_id = self.language_gpu_map[request.language_id]
                        with self.decode_locks[gpu_id]:
                            self.decode_queues[gpu_id].append(request)
                            self.stats['queue_sizes'][gpu_id]['decode'] += 1
                        logger.debug(f"Request {request.request_id} 继续DECODE step {request.decode_step}，已重新入队")
                    else:
                        logger.debug(f"Request {request.request_id} 翻译完成，decode序列结束")
                        # 更新 src_len 到当前 session.source 的长度
                        session.src_len = len(session.source)
            
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
                logger.debug(f"发送最终结果到客户端: '{result.get('generated_text', '')}'")
            else:
                # 中间步骤，不调用回调，继续处理
                logger.debug(f"Request {request.request_id} decode步骤 {request.decode_step} 完成，继续处理...")
            
        except Exception as e:
            logger.error(f"Error updating session for request {request.request_id}: {e}")
            self._handle_failed_request(request, f"Session update failed: {str(e)}")
    
    def _update_session_memory_from_result(self, session: UserSession, result: Dict[str, Any]):
        """从推理结果中更新session的内存使用统计"""
        try:
            # 从result中获取内存统计信息（如果有的话）
            memory_stats = result.get('memory_stats', None)
            if memory_stats:
                # 更新各类型页面使用
                for cache_type, pages_used in memory_stats.items():
                    if cache_type in ['speech_pages', 'llm_prefill_pages', 'llm_decode_pages']:
                        session.update_memory_usage(cache_type, pages_used)
                        
                print(f"🔍 [SCHEDULER-MEMORY] Updated session {session.session_id} memory from result")
            else:
                # 如果result中没有内存统计，可以从其他地方获取或者跳过
                # 这种情况下，内存统计已经在beam_search中直接更新了session
                pass
                
        except Exception as e:
            logger.error(f"Error updating session memory from result: {e}")
    
    def create_memory_update_callback(self, session_id: str):
        """创建内存更新回调函数，用于从beam_search更新session内存"""
        def update_session_memory(session_id: str, memory_stats: Dict[str, int]):
            """内存更新回调函数 - 直接更新session内存统计"""
            try:
                # 通过session_id查找session对象
                session = None
                with self.session_lock:
                    for language_id, user_sessions in self.user_sessions.items():
                        for user_id, user_session in user_sessions.items():
                            if user_session.session_id == session_id:
                                session = user_session
                                break
                        if session:
                            break
                
                if session:
                    # 更新session的内存使用统计
                    for cache_type, pages_used in memory_stats.items():
                        if cache_type in ['speech_pages', 'llm_prefill_pages', 'llm_decode_pages']:
                            session.update_memory_usage(cache_type, pages_used)
                    
                    print(f"🔍 [MEMORY-CALLBACK] Updated session {session_id} memory: {memory_stats}")
                else:
                    print(f"⚠️ [MEMORY-CALLBACK] Session {session_id} not found for memory update")
                    
            except Exception as e:
                logger.error(f"Error in memory update callback for session {session_id}: {e}")
        
        return update_session_memory
    
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
        
        self.stats['failed_requests'] += 1
    
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
                    session = self.user_sessions[language_id][user_id]
                    session_id = session.session_id
                    del self.user_sessions[language_id][user_id]
                    self.stats['active_sessions'] -= 1
                    # 🔥 清理对应的session锁，避免内存泄漏
                    self._cleanup_session_lock(session_id)
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
                    session=session,
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
                    
                    # 🔥 清理对应的session锁
                    self._cleanup_session_lock(session.session_id)
                    
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
