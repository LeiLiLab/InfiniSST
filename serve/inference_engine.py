#!/usr/bin/env python3
"""
InfiniSST 推理引擎 - 整合版
连接scheduler和实际的infinisst_faster模型，实现多请求并发推理
"""

import asyncio
import logging
import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
import threading
from queue import Queue, Empty

# 导入相关模块
from agents.infinisst_faster import InfiniSSTFaster
from agents.infinisst import S2TAgentStates
from scheduler import InferenceRequest, RequestStage

logger = logging.getLogger(__name__)

@dataclass
class EngineConfig:
    """推理引擎配置"""
    max_concurrent_requests: int = 32
    gpu_memory_fraction: float = 0.8
    enable_beam_search: bool = True
    beam_size: int = 4
    max_new_tokens: int = 20
    temperature: float = 1.0
    top_p: float = 0.9

class InferenceEngine:
    """
    InfiniSST推理引擎
    负责实际的模型推理，支持batch处理和并发执行
    """
    
    def __init__(self, 
                 model_args,
                 config: EngineConfig = None,
                 gpu_id: int = 0):
        """
        初始化推理引擎
        
        Args:
            model_args: 模型参数
            config: 引擎配置
            gpu_id: GPU设备ID
        """
        self.model_args = model_args
        self.config = config or EngineConfig()
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        
        # 模型实例
        self.model = None
        self.tokenizer = None
        
        # 处理队列和线程池
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.processing_queue = Queue()
        
        # 状态管理
        self.is_loaded = False
        self.is_running = False
        self.stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'total_tokens_generated': 0,
            'average_latency': 0.0
        }
        
        logger.info(f"推理引擎初始化完成，GPU: {gpu_id}")
    
    def load_model(self) -> bool:
        """加载模型"""
        try:
            logger.info(f"开始加载模型到GPU {self.gpu_id}...")
            
            # 创建InfiniSSTFaster实例
            self.model = InfiniSSTFaster(self.model_args)
            self.tokenizer = self.model.tokenizer
            
            # 确保模型在正确的设备上
            if hasattr(self.model.model, 'to'):
                self.model.model = self.model.model.to(self.device)
            
            self.is_loaded = True
            logger.info(f"模型加载成功，GPU: {self.gpu_id}")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.is_loaded = False
            return False
    
    def start(self):
        """启动推理引擎"""
        if not self.is_loaded:
            logger.error("模型未加载，无法启动引擎")
            return False
        
        if self.is_running:
            logger.warning("推理引擎已在运行")
            return True
        
        self.is_running = True
        logger.info(f"推理引擎已启动，GPU: {self.gpu_id}")
        return True
    
    def stop(self):
        """停止推理引擎"""
        self.is_running = False
        self.executor.shutdown(wait=True)
        logger.info(f"推理引擎已停止，GPU: {self.gpu_id}")
    
    def process_batch(self, requests: List[InferenceRequest]) -> List[Dict[str, Any]]:
        """
        批处理推理请求
        
        Args:
            requests: 推理请求列表
            
        Returns:
            推理结果列表
        """
        if not self.is_running or not self.is_loaded:
            raise RuntimeError("推理引擎未运行或模型未加载")
        
        start_time = time.time()
        results = []
        
        try:
            # 按阶段分组处理
            prefill_requests = [r for r in requests if r.stage == RequestStage.PREFILL]
            decode_requests = [r for r in requests if r.stage == RequestStage.DECODE]
            
            # 处理prefill请求
            if prefill_requests:
                prefill_results = self._process_prefill_batch(prefill_requests)
                results.extend(prefill_results)
            
            # 处理decode请求
            if decode_requests:
                decode_results = self._process_decode_batch(decode_requests)
                results.extend(decode_results)
            
            # 更新统计信息
            self.stats['completed_requests'] += len(results)
            self.stats['total_requests'] += len(requests)
            
            latency = time.time() - start_time
            self.stats['average_latency'] = (
                self.stats['average_latency'] * (self.stats['completed_requests'] - len(results)) + 
                latency * len(results)
            ) / self.stats['completed_requests']
            
            logger.debug(f"批处理完成: {len(requests)}个请求, 耗时: {latency:.3f}s")
            
        except Exception as e:
            logger.error(f"批处理失败: {e}")
            self.stats['failed_requests'] += len(requests)
            # 返回错误结果
            results = [
                {
                    'request_id': req.request_id,
                    'success': False,
                    'error': str(e),
                    'generated_text': '',
                    'generated_tokens': []
                }
                for req in requests
            ]
        
        return results
    
    def _process_prefill_batch(self, requests: List[InferenceRequest]) -> List[Dict[str, Any]]:
        """处理prefill阶段的请求"""
        results = []
        
        for request in requests:
            try:
                # 创建agent状态
                states = self._create_agent_states(request)
                
                # 执行推理
                action = self.model.policy(states)
                
                # 处理结果
                result = self._process_action(request, action, states)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Prefill处理失败 (request_id: {request.request_id}): {e}")
                results.append({
                    'request_id': request.request_id,
                    'success': False,
                    'error': str(e),
                    'generated_text': '',
                    'generated_tokens': []
                })
        
        return results
    
    def _process_decode_batch(self, requests: List[InferenceRequest]) -> List[Dict[str, Any]]:
        """处理decode阶段的请求"""
        results = []
        
        for request in requests:
            try:
                # 创建agent状态
                states = self._create_agent_states(request)
                
                # 执行推理
                action = self.model.policy(states)
                
                # 处理结果
                result = self._process_action(request, action, states)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Decode处理失败 (request_id: {request.request_id}): {e}")
                results.append({
                    'request_id': request.request_id,
                    'success': False,
                    'error': str(e),
                    'generated_text': '',
                    'generated_tokens': []
                })
        
        return results
    
    def _create_agent_states(self, request: InferenceRequest) -> S2TAgentStates:
        """从推理请求创建agent状态"""
        states = S2TAgentStates()
        
        # 设置音频数据
        if isinstance(request.speech_batch, torch.Tensor):
            states.source = request.speech_batch.cpu().numpy().tolist()
        else:
            states.source = request.speech_batch
        
        states.source_sample_rate = 16000  # 默认采样率
        states.source_finished = (request.stage == RequestStage.DECODE)
        
        # 设置文本数据
        if isinstance(request.input_ids, torch.Tensor):
            states.target_ids = request.input_ids.cpu().numpy().tolist()
        else:
            states.target_ids = request.input_ids if request.input_ids else []
        
        # 设置缓存
        states.speech_cache = request.speech_cache
        states.past_key_values = request.past_key_values
        
        return states
    
    def _process_action(self, request: InferenceRequest, action, states: S2TAgentStates) -> Dict[str, Any]:
        """处理模型输出的action"""
        from simuleval.agents.actions import WriteAction, ReadAction
        
        result = {
            'request_id': request.request_id,
            'success': True,
            'generated_text': '',
            'generated_tokens': [],
            'finished': False,
            'speech_cache': states.speech_cache,
            'past_key_values': states.past_key_values
        }
        
        if isinstance(action, WriteAction):
            result['generated_text'] = action.content
            result['finished'] = action.finished
            
            # 如果有新的token，添加到统计中
            if action.content:
                # 简单的token计数（实际应该用tokenizer）
                token_count = len(action.content.split())
                self.stats['total_tokens_generated'] += token_count
                result['generated_tokens'] = list(range(token_count))  # 占位符
        
        elif isinstance(action, ReadAction):
            result['generated_text'] = ''
            result['finished'] = False
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        return {
            'gpu_id': self.gpu_id,
            'is_loaded': self.is_loaded,
            'is_running': self.is_running,
            'stats': self.stats.copy(),
            'config': {
                'max_concurrent_requests': self.config.max_concurrent_requests,
                'beam_size': self.config.beam_size,
                'max_new_tokens': self.config.max_new_tokens
            }
        }

class MultiGPUInferenceEngine:
    """
    多GPU推理引擎管理器
    管理多个GPU上的推理引擎实例
    """
    
    def __init__(self, gpu_language_map: Dict[int, str], model_args_map: Dict[int, Any]):
        """
        初始化多GPU推理引擎
        
        Args:
            gpu_language_map: GPU到语言对的映射
            model_args_map: GPU到模型参数的映射
        """
        self.gpu_language_map = gpu_language_map
        self.model_args_map = model_args_map
        
        # 创建引擎实例
        self.engines: Dict[int, InferenceEngine] = {}
        for gpu_id, language_pair in gpu_language_map.items():
            if gpu_id not in model_args_map:
                logger.warning(f"GPU {gpu_id} 没有对应的模型参数，跳过")
                continue
            
            engine = InferenceEngine(
                model_args=model_args_map[gpu_id],
                gpu_id=gpu_id
            )
            self.engines[gpu_id] = engine
        
        logger.info(f"多GPU推理引擎初始化完成，支持GPU: {list(self.engines.keys())}")
    
    def load_all_models(self) -> bool:
        """加载所有GPU上的模型"""
        success = True
        for gpu_id, engine in self.engines.items():
            if not engine.load_model():
                success = False
                logger.error(f"GPU {gpu_id} 模型加载失败")
        return success
    
    def start_all(self):
        """启动所有推理引擎"""
        for engine in self.engines.values():
            engine.start()
    
    def stop_all(self):
        """停止所有推理引擎"""
        for engine in self.engines.values():
            engine.stop()
    
    def get_engine(self, gpu_id: int) -> Optional[InferenceEngine]:
        """获取指定GPU的推理引擎"""
        return self.engines.get(gpu_id)
    
    def process_batch(self, gpu_id: int, requests: List[InferenceRequest]) -> List[Dict[str, Any]]:
        """在指定GPU上处理批请求"""
        engine = self.get_engine(gpu_id)
        if not engine:
            raise ValueError(f"GPU {gpu_id} 上没有可用的推理引擎")
        
        return engine.process_batch(requests)
    
    def get_all_stats(self) -> Dict[int, Dict[str, Any]]:
        """获取所有引擎的统计信息"""
        return {gpu_id: engine.get_stats() for gpu_id, engine in self.engines.items()} 