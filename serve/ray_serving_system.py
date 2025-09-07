#!/usr/bin/env python3
"""
Ray-based Serving System for InfiniSST
支持动态batch调度的分布式实时翻译系统
"""

import ray
import asyncio
import time
import uuid
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque

# Import existing components
from scheduler import RequestStage, UserSession, InferenceRequest
from inference_engine import InferenceEngine, EngineConfig

logger = logging.getLogger(__name__)

# ===== Ray Configuration =====

@dataclass
class RayServingConfig:
    """Ray服务系统配置"""
    # Ray集群配置
    ray_address: Optional[str] = None  # Ray集群地址，None表示本地模式
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None
    
    # GPU配置
    gpu_language_map: Dict[int, str] = field(default_factory=dict)
    
    # 批处理配置
    max_batch_size: int = 32
    batch_timeout_ms: float = 100.0  # 批处理超时时间（毫秒）
    min_batch_size: int = 1
    
    # 动态调度配置
    enable_dynamic_batching: bool = True
    load_balance_strategy: str = "least_loaded"  # least_loaded, round_robin, gpu_memory
    
    # 会话管理
    session_timeout: int = 3600  # 会话超时时间（秒）
    cleanup_interval: int = 60  # 清理间隔（秒）
    
    # 性能配置
    max_concurrent_sessions: int = 1000
    prefetch_enabled: bool = True
    async_result_processing: bool = True

# ===== Ray Actors =====

@ray.remote(num_gpus=1)
class ModelActor:
    """
    单GPU模型Actor，负责在特定GPU上进行推理
    """
    
    def __init__(self, gpu_id: int, language_id: str, model_args: Any, config: EngineConfig):
        self.gpu_id = gpu_id
        self.language_id = language_id
        self.model_args = model_args
        self.config = config
        
        # 推理引擎
        self.inference_engine: Optional[InferenceEngine] = None
        
        # 批处理状态
        self.current_batch: List[InferenceRequest] = []
        self.batch_lock = threading.Lock()
        self.last_batch_time = time.time()
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "avg_batch_size": 0.0,
            "gpu_utilization": 0.0,
            "memory_usage": 0.0,
            "processing_time": 0.0
        }
        
        logger.info(f"🚀 ModelActor初始化: GPU {gpu_id}, Language: {language_id}")
    
    def initialize(self) -> bool:
        """初始化模型"""
        try:
            # 创建推理引擎
            self.inference_engine = InferenceEngine(
                model_args=self.model_args,
                config=self.config,
                gpu_id=self.gpu_id,
                language_id=self.language_id
            )
            
            # 加载模型
            success = self.inference_engine.load_model()
            if success:
                self.inference_engine.start()
                logger.info(f"✅ ModelActor GPU {self.gpu_id} 模型加载成功")
                return True
            else:
                logger.error(f"❌ ModelActor GPU {self.gpu_id} 模型加载失败")
                return False
                
        except Exception as e:
            logger.error(f"❌ ModelActor GPU {self.gpu_id} 初始化异常: {e}")
            return False
    
    def process_batch(self, requests: List[InferenceRequest]) -> List[Dict[str, Any]]:
        """处理批处理请求"""
        if not self.inference_engine:
            error_results = []
            for req in requests:
                error_results.append({
                    "request_id": req.request_id,
                    "success": False,
                    "error": "Model not initialized"
                })
            return error_results
        
        start_time = time.time()
        
        try:
            # 使用推理引擎处理批处理
            results = self.inference_engine.process_batch(requests)
            
            # 更新统计信息
            self.stats["total_requests"] += len(requests)
            self.stats["total_batches"] += 1
            self.stats["avg_batch_size"] = self.stats["total_requests"] / self.stats["total_batches"]
            self.stats["processing_time"] = time.time() - start_time
            
            # 更新GPU使用率（如果可用）
            if torch.cuda.is_available():
                memory_info = torch.cuda.memory_stats(self.gpu_id)
                allocated = memory_info.get('allocated_bytes.all.current', 0)
                reserved = memory_info.get('reserved_bytes.all.current', 0)
                self.stats["memory_usage"] = allocated / reserved if reserved > 0 else 0.0
            
            logger.debug(f"📊 ModelActor GPU {self.gpu_id} 处理批次: {len(requests)} 请求, 耗时: {self.stats['processing_time']:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ ModelActor GPU {self.gpu_id} 批处理失败: {e}")
            # 返回错误结果
            error_results = []
            for req in requests:
                error_results.append({
                    "request_id": req.request_id,
                    "success": False,
                    "error": str(e)
                })
            return error_results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取Actor统计信息"""
        return {
            "gpu_id": self.gpu_id,
            "language_id": self.language_id,
            "stats": self.stats.copy(),
            "current_batch_size": len(self.current_batch),
            "engine_stats": self.inference_engine.get_stats() if self.inference_engine else {}
        }
    
    def cleanup(self):
        """清理资源"""
        if self.inference_engine:
            self.inference_engine.stop()
            logger.info(f"🧹 ModelActor GPU {self.gpu_id} 资源清理完成")

@ray.remote
class SessionActor:
    """
    会话Actor，管理单个用户会话的状态和生命周期
    """
    
    def __init__(self, session_id: str, user_id: str, language_id: str, config: RayServingConfig):
        self.session_id = session_id
        self.user_id = user_id
        self.language_id = language_id
        self.config = config
        
        # 会话状态
        self.session = UserSession(
            user_id=user_id,
            language_id=language_id,
            session_id=session_id
        )
        
        # 请求历史
        self.request_history: List[InferenceRequest] = []
        self.result_history: List[Dict[str, Any]] = []
        
        # 性能统计
        self.performance_stats = {
            "requests_processed": 0,
            "avg_latency": 0.0,
            "total_characters": 0,
            "session_start_time": time.time(),
            "last_activity": time.time()
        }
        
        logger.info(f"📱 SessionActor创建: {session_id} (用户: {user_id}, 语言: {language_id})")
    
    def update_session_state(self, speech_data: np.ndarray, is_final: bool = False):
        """更新会话状态"""
        self.session.source.extend(speech_data.tolist())
        if is_final:
            self.session.source_finished = True
        
        self.session.last_activity = time.time()
        self.performance_stats["last_activity"] = time.time()
        
        logger.debug(f"📝 Session {self.session_id} 状态更新: 音频长度 {len(self.session.source)}")
    
    def add_translation_result(self, result: Dict[str, Any]):
        """添加翻译结果"""
        if result.get("success", False):
            translation = result.get("translation", "")
            if translation:
                self.session.target.append(translation)
                self.performance_stats["total_characters"] += len(translation)
        
        self.result_history.append(result)
        self.performance_stats["requests_processed"] += 1
        
        # 计算平均延迟
        if "latency" in result:
            current_avg = self.performance_stats["avg_latency"]
            request_count = self.performance_stats["requests_processed"]
            new_latency = result["latency"]
            self.performance_stats["avg_latency"] = (current_avg * (request_count - 1) + new_latency) / request_count
    
    def get_session_info(self) -> Dict[str, Any]:
        """获取会话信息"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "language_id": self.language_id,
            "session_state": {
                "source_length": len(self.session.source),
                "target_segments": len(self.session.target),
                "source_finished": self.session.source_finished,
                "segment_idx": self.session.segment_idx
            },
            "performance": self.performance_stats.copy(),
            "session_age": time.time() - self.performance_stats["session_start_time"],
            "inactive_time": time.time() - self.performance_stats["last_activity"]
        }
    
    def reset_session(self):
        """重置会话状态"""
        self.session.reset()
        self.request_history.clear()
        self.result_history.clear()
        
        # 重置性能统计（保留会话级别的统计）
        self.performance_stats.update({
            "requests_processed": 0,
            "avg_latency": 0.0,
            "total_characters": 0,
            "last_activity": time.time()
        })
        
        logger.info(f"🔄 Session {self.session_id} 已重置")
    
    def cleanup(self):
        """清理会话资源"""
        logger.info(f"🧹 SessionActor {self.session_id} 清理完成")

@ray.remote
class SchedulerActor:
    """
    全局调度器Actor，负责负载均衡、批处理优化和资源管理
    """
    
    def __init__(self, config: RayServingConfig, model_actors: Dict[int, ray.ObjectRef]):
        self.config = config
        self.model_actors = model_actors
        
        # 请求队列 - 按GPU分组
        self.request_queues: Dict[int, deque] = defaultdict(deque)
        self.queue_locks: Dict[int, threading.Lock] = defaultdict(threading.Lock)
        
        # 会话管理
        self.sessions: Dict[str, ray.ObjectRef] = {}  # session_id -> SessionActor
        self.session_gpu_map: Dict[str, int] = {}  # session_id -> gpu_id
        
        # 负载均衡状态
        self.gpu_load: Dict[int, float] = {gpu_id: 0.0 for gpu_id in self.model_actors.keys()}
        self.last_used_gpu: Dict[str, int] = {}  # language_id -> last_used_gpu_id
        
        # 动态批处理状态
        self.batch_timers: Dict[int, Optional[asyncio.Task]] = {}
        self.processing_flags: Dict[int, bool] = defaultdict(bool)
        
        # 统计信息
        self.global_stats = {
            "total_requests": 0,
            "total_sessions": 0,
            "active_sessions": 0,
            "avg_queue_time": 0.0,
            "throughput": 0.0,
            "gpu_utilization": {}
        }
        
        # 启动后台任务
        self.cleanup_task = None
        self.monitoring_task = None
        
        logger.info(f"🎯 SchedulerActor初始化: {len(model_actors)} GPU, 支持语言: {list(config.gpu_language_map.values())}")
    
    async def start_background_tasks(self):
        """启动后台任务"""
        self.cleanup_task = asyncio.create_task(self._cleanup_sessions_periodically())
        self.monitoring_task = asyncio.create_task(self._monitor_system_performance())
        
        # 为每个GPU启动批处理任务
        for gpu_id in self.model_actors.keys():
            batch_task = asyncio.create_task(self._dynamic_batch_processor(gpu_id))
            self.batch_timers[gpu_id] = batch_task
        
        logger.info("🚀 SchedulerActor后台任务已启动")
    
    async def create_session(self, user_id: str, language_id: str, session_id: Optional[str] = None) -> str:
        """创建新会话"""
        if session_id is None:
            session_id = f"{user_id}_{language_id}_{int(time.time()*1000)}"
        
        # 选择GPU
        gpu_id = self._select_gpu_for_language(language_id)
        if gpu_id is None:
            raise ValueError(f"不支持的语言: {language_id}")
        
        # 创建SessionActor
        session_actor = SessionActor.remote(session_id, user_id, language_id, self.config)
        
        # 注册会话
        self.sessions[session_id] = session_actor
        self.session_gpu_map[session_id] = gpu_id
        
        # 更新统计
        self.global_stats["total_sessions"] += 1
        self.global_stats["active_sessions"] = len(self.sessions)
        
        logger.info(f"✅ 会话创建成功: {session_id} -> GPU {gpu_id}")
        return session_id
    
    async def submit_request(self, 
                           session_id: str,
                           speech_data: np.ndarray,
                           stage: RequestStage = RequestStage.PREFILL,
                           is_final: bool = False,
                           max_new_tokens: int = 20,
                           result_callback: Optional[Callable] = None) -> str:
        """提交推理请求"""
        
        # 检查会话是否存在
        if session_id not in self.sessions:
            raise ValueError(f"会话不存在: {session_id}")
        
        session_actor = self.sessions[session_id]
        gpu_id = self.session_gpu_map[session_id]
        
        # 更新会话状态
        await session_actor.update_session_state.remote(speech_data, is_final)
        
        # 创建请求
        request_id = f"{session_id}_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
        
        # 获取会话信息用于创建请求
        session_info = await session_actor.get_session_info.remote()
        
        # 创建InferenceRequest
        # 注意：这里需要根据实际的InferenceRequest构造函数调整参数
        request = InferenceRequest(
            request_id=request_id,
            user_id=session_info["user_id"],
            language_id=session_info["language_id"],
            session_id=session_id,
            stage=stage,
            speech_batch=torch.tensor(speech_data, dtype=torch.float32),
            input_ids=torch.tensor([], dtype=torch.long),  # 需要根据实际情况设置
            max_new_tokens=max_new_tokens,
            result_callback=result_callback
        )
        
        # 添加到对应GPU的队列
        with self.queue_locks[gpu_id]:
            self.request_queues[gpu_id].append(request)
        
        # 更新统计
        self.global_stats["total_requests"] += 1
        
        logger.debug(f"📤 请求提交: {request_id} -> GPU {gpu_id} 队列")
        return request_id
    
    def _select_gpu_for_language(self, language_id: str) -> Optional[int]:
        """为语言选择最优GPU"""
        
        # 找到支持该语言的GPU
        available_gpus = []
        for gpu_id, supported_lang in self.config.gpu_language_map.items():
            if supported_lang == language_id:
                available_gpus.append(gpu_id)
        
        if not available_gpus:
            return None
        
        # 根据负载均衡策略选择GPU
        if self.config.load_balance_strategy == "least_loaded":
            return min(available_gpus, key=lambda gpu: self.gpu_load[gpu])
        elif self.config.load_balance_strategy == "round_robin":
            last_gpu = self.last_used_gpu.get(language_id, -1)
            next_gpu_idx = (available_gpus.index(last_gpu) + 1) % len(available_gpus) if last_gpu in available_gpus else 0
            selected_gpu = available_gpus[next_gpu_idx]
            self.last_used_gpu[language_id] = selected_gpu
            return selected_gpu
        else:
            # 默认使用第一个可用GPU
            return available_gpus[0]
    
    async def _dynamic_batch_processor(self, gpu_id: int):
        """动态批处理处理器"""
        logger.info(f"🔄 GPU {gpu_id} 动态批处理器启动")
        
        while True:
            try:
                current_time = time.time()
                batch_to_process = []
                
                # 收集批处理请求
                with self.queue_locks[gpu_id]:
                    queue = self.request_queues[gpu_id]
                    
                    # 检查是否应该处理批次
                    should_process = (
                        len(queue) >= self.config.max_batch_size or
                        (len(queue) >= self.config.min_batch_size and 
                         (len(queue) > 0 and current_time - queue[0].timestamp > self.config.batch_timeout_ms / 1000.0))
                    )
                    
                    if should_process and not self.processing_flags[gpu_id]:
                        # 从队列中取出请求
                        batch_size = min(len(queue), self.config.max_batch_size)
                        for _ in range(batch_size):
                            if queue:
                                batch_to_process.append(queue.popleft())
                        
                        if batch_to_process:
                            self.processing_flags[gpu_id] = True
                
                # 处理批次
                if batch_to_process:
                    await self._process_batch_on_gpu(gpu_id, batch_to_process)
                    self.processing_flags[gpu_id] = False
                
                # 短暂等待避免忙等待
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"❌ GPU {gpu_id} 批处理器异常: {e}")
                self.processing_flags[gpu_id] = False
                await asyncio.sleep(1)  # 错误恢复
    
    async def _process_batch_on_gpu(self, gpu_id: int, batch: List[InferenceRequest]):
        """在指定GPU上处理批次"""
        start_time = time.time()
        
        try:
            # 获取ModelActor并处理批次
            model_actor = self.model_actors[gpu_id]
            results = await model_actor.process_batch.remote(batch)
            
            # 处理结果
            for i, result in enumerate(results):
                request = batch[i]
                
                # 更新会话状态
                if request.session_id in self.sessions:
                    session_actor = self.sessions[request.session_id]
                    await session_actor.add_translation_result.remote(result)
                
                # 调用结果回调
                if request.result_callback:
                    try:
                        if asyncio.iscoroutinefunction(request.result_callback):
                            await request.result_callback(result)
                        else:
                            request.result_callback(result)
                    except Exception as e:
                        logger.error(f"结果回调异常: {e}")
            
            # 更新GPU负载
            processing_time = time.time() - start_time
            self.gpu_load[gpu_id] = processing_time  # 简化的负载指标
            
            logger.debug(f"✅ GPU {gpu_id} 批处理完成: {len(batch)} 请求, 耗时: {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ GPU {gpu_id} 批处理失败: {e}")
            
            # 处理失败的请求
            for request in batch:
                error_result = {
                    "request_id": request.request_id,
                    "success": False,
                    "error": str(e),
                    "latency": time.time() - request.timestamp
                }
                
                if request.result_callback:
                    try:
                        if asyncio.iscoroutinefunction(request.result_callback):
                            await request.result_callback(error_result)
                        else:
                            request.result_callback(error_result)
                    except Exception as callback_error:
                        logger.error(f"错误回调异常: {callback_error}")
    
    async def _cleanup_sessions_periodically(self):
        """定期清理过期会话"""
        while True:
            try:
                current_time = time.time()
                sessions_to_remove = []
                
                # 检查所有会话
                for session_id, session_actor in self.sessions.items():
                    session_info = await session_actor.get_session_info.remote()
                    
                    # 检查是否超时
                    if session_info["inactive_time"] > self.config.session_timeout:
                        sessions_to_remove.append(session_id)
                
                # 清理过期会话
                for session_id in sessions_to_remove:
                    await self._cleanup_session(session_id)
                
                # 更新活跃会话统计
                self.global_stats["active_sessions"] = len(self.sessions)
                
                if sessions_to_remove:
                    logger.info(f"🧹 清理了 {len(sessions_to_remove)} 个过期会话")
                
            except Exception as e:
                logger.error(f"会话清理异常: {e}")
            
            await asyncio.sleep(self.config.cleanup_interval)
    
    async def _monitor_system_performance(self):
        """监控系统性能"""
        while True:
            try:
                # 收集GPU统计信息
                gpu_stats = {}
                for gpu_id, model_actor in self.model_actors.items():
                    stats = await model_actor.get_stats.remote()
                    gpu_stats[gpu_id] = stats
                
                self.global_stats["gpu_utilization"] = gpu_stats
                
                # 计算吞吐量
                # 这里需要更复杂的逻辑来计算真实吞吐量
                
                logger.debug(f"📊 系统性能监控: 活跃会话 {self.global_stats['active_sessions']}, GPU利用率: {gpu_stats}")
                
            except Exception as e:
                logger.error(f"性能监控异常: {e}")
            
            await asyncio.sleep(30)  # 每30秒监控一次
    
    async def _cleanup_session(self, session_id: str):
        """清理单个会话"""
        try:
            if session_id in self.sessions:
                session_actor = self.sessions[session_id]
                await session_actor.cleanup.remote()
                
                # 从映射中移除
                del self.sessions[session_id]
                if session_id in self.session_gpu_map:
                    del self.session_gpu_map[session_id]
                
                logger.info(f"🧹 会话清理完成: {session_id}")
                
        except Exception as e:
            logger.error(f"会话清理失败 {session_id}: {e}")
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            "global_stats": self.global_stats.copy(),
            "queue_stats": {
                gpu_id: len(queue) for gpu_id, queue in self.request_queues.items()
            },
            "gpu_load": self.gpu_load.copy(),
            "active_sessions": len(self.sessions),
            "config": {
                "max_batch_size": self.config.max_batch_size,
                "batch_timeout_ms": self.config.batch_timeout_ms,
                "enable_dynamic_batching": self.config.enable_dynamic_batching
            }
        }
    
    async def cleanup(self):
        """清理调度器资源"""
        # 停止后台任务
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        for gpu_id, batch_task in self.batch_timers.items():
            if batch_task:
                batch_task.cancel()
        
        # 清理所有会话
        for session_id in list(self.sessions.keys()):
            await self._cleanup_session(session_id)
        
        logger.info("🧹 SchedulerActor清理完成")

# ===== Ray Serving System =====

class RayServingSystem:
    """
    基于Ray的分布式服务系统主类
    """
    
    def __init__(self, config: RayServingConfig, model_args_map: Dict[int, Any]):
        self.config = config
        self.model_args_map = model_args_map
        
        # Ray Actors
        self.model_actors: Dict[int, ray.ObjectRef] = {}
        self.scheduler_actor: Optional[ray.ObjectRef] = None
        
        # 系统状态
        self.is_initialized = False
        self.is_running = False
        
        logger.info(f"🌟 RayServingSystem初始化: {len(config.gpu_language_map)} GPU")
    
    async def initialize(self) -> bool:
        """初始化Ray集群和Actors"""
        try:
            # 初始化Ray
            if not ray.is_initialized():
                if self.config.ray_address:
                    ray.init(address=self.config.ray_address)
                    logger.info(f"🔗 连接到Ray集群: {self.config.ray_address}")
                else:
                    ray.init(
                        num_cpus=self.config.num_cpus,
                        num_gpus=self.config.num_gpus
                    )
                    logger.info("🏠 启动本地Ray集群")
            
            # 创建ModelActors
            for gpu_id, language_id in self.config.gpu_language_map.items():
                model_args = self.model_args_map.get(gpu_id, {})
                engine_config = EngineConfig()  # 使用默认配置，可以根据需要自定义
                
                model_actor = ModelActor.remote(gpu_id, language_id, model_args, engine_config)
                
                # 初始化模型
                init_success = await model_actor.initialize.remote()
                if not init_success:
                    logger.error(f"❌ ModelActor GPU {gpu_id} 初始化失败")
                    return False
                
                self.model_actors[gpu_id] = model_actor
                logger.info(f"✅ ModelActor GPU {gpu_id} ({language_id}) 创建成功")
            
            # 创建SchedulerActor
            self.scheduler_actor = SchedulerActor.remote(self.config, self.model_actors)
            await self.scheduler_actor.start_background_tasks.remote()
            
            self.is_initialized = True
            logger.info("🎉 RayServingSystem初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ RayServingSystem初始化失败: {e}")
            return False
    
    async def start(self):
        """启动服务系统"""
        if not self.is_initialized:
            logger.error("系统未初始化，请先调用initialize()")
            return False
        
        self.is_running = True
        logger.info("🚀 RayServingSystem已启动")
        return True
    
    async def stop(self):
        """停止服务系统"""
        self.is_running = False
        
        # 清理Scheduler
        if self.scheduler_actor:
            await self.scheduler_actor.cleanup.remote()
        
        # 清理ModelActors
        for gpu_id, model_actor in self.model_actors.items():
            await model_actor.cleanup.remote()
        
        logger.info("⏹️ RayServingSystem已停止")
    
    async def create_session(self, user_id: str, language_id: str, session_id: Optional[str] = None) -> str:
        """创建新的翻译会话"""
        if not self.is_running or not self.scheduler_actor:
            raise RuntimeError("系统未运行")
        
        return await self.scheduler_actor.create_session.remote(user_id, language_id, session_id)
    
    async def submit_translation_request(self,
                                       session_id: str,
                                       speech_data: np.ndarray,
                                       stage: RequestStage = RequestStage.PREFILL,
                                       is_final: bool = False,
                                       max_new_tokens: int = 20,
                                       result_callback: Optional[Callable] = None) -> str:
        """提交翻译请求"""
        if not self.is_running or not self.scheduler_actor:
            raise RuntimeError("系统未运行")
        
        return await self.scheduler_actor.submit_request.remote(
            session_id=session_id,
            speech_data=speech_data,
            stage=stage,
            is_final=is_final,
            max_new_tokens=max_new_tokens,
            result_callback=result_callback
        )
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        if not self.scheduler_actor:
            return {"error": "调度器未初始化"}
        
        return await self.scheduler_actor.get_system_stats.remote()
    
    def __del__(self):
        """析构函数，确保资源清理"""
        if ray.is_initialized():
            ray.shutdown()

# ===== Factory Functions =====

def create_ray_serving_config(
    gpu_ids: List[int],
    language_pairs: List[str],
    max_batch_size: int = 32,
    batch_timeout_ms: float = 100.0,
    **kwargs
) -> RayServingConfig:
    """创建Ray服务配置的工厂函数"""
    
    # 创建GPU语言映射
    gpu_language_map = {}
    for i, gpu_id in enumerate(gpu_ids):
        if i < len(language_pairs):
            gpu_language_map[gpu_id] = language_pairs[i]
        else:
            # 如果GPU多于语言对，循环分配
            gpu_language_map[gpu_id] = language_pairs[i % len(language_pairs)]
    
    return RayServingConfig(
        gpu_language_map=gpu_language_map,
        max_batch_size=max_batch_size,
        batch_timeout_ms=batch_timeout_ms,
        **kwargs
    )

async def create_ray_serving_system(
    gpu_ids: List[int],
    language_pairs: List[str],
    model_args_factory: Callable[[int, str], Any],
    **config_kwargs
) -> RayServingSystem:
    """创建并初始化Ray服务系统的工厂函数"""
    
    # 创建配置
    config = create_ray_serving_config(gpu_ids, language_pairs, **config_kwargs)
    
    # 创建模型参数映射
    model_args_map = {}
    for gpu_id, language_id in config.gpu_language_map.items():
        model_args_map[gpu_id] = model_args_factory(gpu_id, language_id)
    
    # 创建系统实例
    system = RayServingSystem(config, model_args_map)
    
    # 初始化系统
    init_success = await system.initialize()
    if not init_success:
        raise RuntimeError("Ray服务系统初始化失败")
    
    return system

# ===== Example Usage =====

if __name__ == "__main__":
    import asyncio
    
    async def example_usage():
        """示例用法"""
        
        # 定义模型参数工厂函数
        def model_args_factory(gpu_id: int, language_id: str):
            # 根据GPU和语言返回相应的模型参数
            # 这里需要根据实际的模型参数结构来实现
            return {
                "model_path": f"/path/to/model/{language_id}",
                "gpu_id": gpu_id,
                "language": language_id
            }
        
        # 创建并启动系统
        system = await create_ray_serving_system(
            gpu_ids=[0, 1],
            language_pairs=["English -> Chinese", "English -> Italian"],
            model_args_factory=model_args_factory,
            max_batch_size=16,
            batch_timeout_ms=100.0
        )
        
        await system.start()
        
        try:
            # 创建会话
            session_id = await system.create_session("user123", "English -> Chinese")
            print(f"创建会话: {session_id}")
            
            # 模拟音频数据
            audio_data = np.random.randn(16000).astype(np.float32)  # 1秒音频
            
            # 结果回调函数
            def result_callback(result):
                print(f"翻译结果: {result}")
            
            # 提交翻译请求
            request_id = await system.submit_translation_request(
                session_id=session_id,
                speech_data=audio_data,
                result_callback=result_callback
            )
            print(f"提交请求: {request_id}")
            
            # 等待一段时间处理
            await asyncio.sleep(5)
            
            # 获取系统统计
            stats = await system.get_system_stats()
            print(f"系统统计: {stats}")
            
        finally:
            await system.stop()
    
    # 运行示例
    asyncio.run(example_usage()) 