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

# 设置logger
logger = logging.getLogger(__name__)

# 导入相关模块
import sys
import os
# 将agents目录添加到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from agents.infinisst_faster import InfiniSSTFaster
    INFINISST_AVAILABLE = True
except ImportError as e:
    logger.warning(f"InfiniSSTFaster不可用: {e}")
    logger.warning("将使用模拟推理模式")
    InfiniSSTFaster = None
    INFINISST_AVAILABLE = False
try:
    from agents.infinisst import S2TAgentStates
    AGENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"agents.infinisst不可用: {e}")
    # 创建占位符类
    class S2TAgentStates:
        def __init__(self, src_len=0, speech_cache=None, past_key_values=None, 
                     target_ids=None, segment_idx=0, translations_list=None):
            self.source = []
            self.target = []
            self.source_finished = False
            self.source_sample_rate = 16000
            self.src_len = src_len or 0
            self.speech_cache = speech_cache
            self.past_key_values = past_key_values
            self.target_ids = target_ids or []
            self.segment_idx = segment_idx or 0
            self.translations_list = translations_list or []
        
        def reset(self):
            self.source = []
            self.target = []
            self.source_finished = False
            self.src_len = 0
            self.speech_cache = None
            self.past_key_values = None
            self.target_ids = []
            self.segment_idx = 0
            self.translations_list = []
    
    AGENTS_AVAILABLE = False
    
from scheduler import InferenceRequest, RequestStage

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
                 gpu_id: int = 0,
                 language_id: str = "en-zh"):
        """
        初始化推理引擎
        
        Args:
            model_args: 模型参数
            config: 引擎配置
            gpu_id: GPU设备ID
            language_id: 语言对ID (例如 "en-zh")
        """
        self.gpu_id = gpu_id
        self.language_id = language_id
        self.device = f"cuda:{gpu_id}"
        self.config = config or EngineConfig()
        
        # 创建完整的模型参数配置
        self.model_args = self._create_model_args(model_args, language_id)
        
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
        
        logger.info(f"推理引擎初始化完成，GPU: {gpu_id}, 语言对: {language_id}")
    
    def _create_model_args(self, base_args, language_id: str):
        """创建完整的模型参数配置"""
        
        # 与 api.py 相同的语言对定义
        LANGUAGE_PAIRS = {
            "English -> Chinese": ("English", "Chinese", "en", "zh"),
            "English -> Italian": ("English", "Italian", "en", "it"),
            "English -> German": ("English", "German", "en", "de"),
            "English -> Spanish": ("English", "Spanish", "en", "es"),
        }
        
        # 模型路径定义（与 api.py 保持一致）
        model_path_de = "/mnt/aries/data6/xixu/demo/en-de/pytorch_model.bin"
        model_path_es = "/mnt/aries/data6/xixu/demo/en-es/pytorch_model.bin"
        model_path = "/mnt/aries/data6/jiaxuanluo/demo/{}-{}/pytorch_model.bin"
        lora_path = "/mnt/aries/data6/jiaxuanluo/demo/{}-{}/lora.bin"
        
        # 解析语言对（与 api.py 中的逻辑完全一致）
        if language_id in LANGUAGE_PAIRS:
            source_lang, target_lang, src_code, tgt_code = LANGUAGE_PAIRS[language_id]
        else:
            # 默认配置
            source_lang, target_lang, src_code, tgt_code = "English", "Chinese", "en", "zh"
        
        # 条件性模型和LoRA加载（与 api.py 逻辑一致）
        if language_id == "English -> German":
            state_dict_path = model_path_de
            lora_path_final = None
        elif language_id == "English -> Spanish":
            state_dict_path = model_path_es
            lora_path_final = None
        else:
            state_dict_path = model_path.format(src_code, tgt_code)
            lora_path_final = lora_path.format(src_code, tgt_code)
        
        # 默认参数配置（与 api.sh 中的参数完全一致）
        default_args = {
            # 基础模型
            'model_name': '/mnt/aries/data6/jiaxuanluo/Qwen2.5-7B-Instruct',
            
            # 语音编码器
            'w2v2_path': '/mnt/aries/data6/xixu/demo/wav2_vec_vox_960h_pl.pt',
            'w2v2_type': 'w2v2',
            'ctc_finetuned': True,
            
            # 模型配置
            'model_type': 'w2v2_qwen25',
            'length_shrink_cfg': "[(1024,2,2)] * 2",
            'block_size': 48,
            'max_cache_size': 576,
            'rope': 1,
            'audio_normalize': 0,
            
            # Stage1/Stage2 模型路径（动态设置）
            'state_dict_path': state_dict_path,
            
            # LoRA配置（动态设置）
            'lora_path': lora_path_final,
            'lora_rank': 32,
            
            # 缓存配置
            'max_llm_cache_size': 1000,
            'always_cache_system_prompt': True,
            
            # 生成参数（与 api.sh 一致）
            'max_len_a': 10,
            'max_len_b': 20,
            'max_new_tokens': 10,
            'beam': 4,
            'repetition_penalty': 1.2,
            'length_penalty': 1.0,
            
            # 运行参数
            'pseudo_batch_size': 1,
            'min_start_sec': 0,
            'latency_multiplier': 2,
            'max_latency_multiplier': 4,
            
            # 生成控制参数（与 api.sh 一致）
            'no_repeat_ngram_size': 5,
            'no_repeat_ngram_lookback': '100d',
            'suppress_non_language': True,
            'do_sample': False,
            'top_p': 0.9,
            'top_k': 50,
            'epsilon_cutoff': 0.0,
            'temperature': 1.0,
            'dpo_sampling': False,
            
            # 语言配置（动态设置）
            'source_lang': source_lang,
            'target_lang': target_lang
        }
        
        # 合并用户提供的参数
        final_args = {**default_args, **(base_args or {})}
        
        # 🔥 确保gpu_id传递给模型参数，用于设备绑定
        final_args['gpu_id'] = self.gpu_id
        
        # 创建一个类似于argparse.Namespace的对象
        class ModelArgs:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        return ModelArgs(**final_args)
    
    def load_model(self) -> bool:
        """加载模型"""
        try:
            logger.info(f"开始加载模型到GPU {self.gpu_id}...")
            
            if not INFINISST_AVAILABLE:
                logger.warning("InfiniSSTFaster不可用，跳过实际模型加载")
                self.model = None
                self.tokenizer = None
                self.is_loaded = False
                return False
            
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
            raise RuntimeError("inference engine not running or model not loaded not running or model not loaded")
        
        start_time = time.time()
        results = []
        
        try:
            # 按阶段分组处理
            prefill_requests = [r for r in requests if r.stage.name == RequestStage.PREFILL.name]
            decode_requests = [r for r in requests if r.stage.name == RequestStage.DECODE.name]
            
            # 处理prefill请求
            if prefill_requests:
                prefill_results = self._process_prefill_batch(prefill_requests)
                results.extend(prefill_results)
            
            # 处理decode请求
            if decode_requests:
                decode_results = self._process_decode_batch(decode_requests)
                results.extend(decode_results)
            
            # 处理完成后检查并清理结束的session
            self._cleanup_finished_sessions(requests, results)
            
            # 更新统计信息
            self.stats['completed_requests'] += len(results)
            self.stats['total_requests'] += len(requests)
            
            latency = time.time() - start_time
            self.stats['average_latency'] = (
                self.stats['average_latency'] * (self.stats['completed_requests'] - len(results)) + 
                latency * len(results)
            ) / self.stats['completed_requests']
            stage = requests[0].stage.value if requests else "unknown"
            prefill_count = len([r for r in requests if r.stage == RequestStage.PREFILL])
            decode_count = len([r for r in requests if r.stage == RequestStage.DECODE])
            logger.info(f"[IMPORTANT] 批处理完成 [{stage}]: {len(requests)} 个请求 (Prefill:{prefill_count}, Decode:{decode_count}), 耗时: {latency*1000:.1f}ms")
            
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
        """处理prefill阶段的请求 - ORCA风格，一次只做prefill步骤"""
        #todo :try
        # 🔥 ORCA架构：为batch中的每个request分别构造beam_search.Request
        beam_requests = []
        for req in requests:
            beam_req = self._create_beam_request(req)
            beam_requests.append(beam_req)
        
        print(f"🔍 [ORCA-PREFILL] 处理batch: {len(beam_requests)} 个requests")
        
        # 直接调用beam_search的prefill函数
        from model.flashinfer.beam_search import prefill
        
        processed_requests, speech_pagetable, llm_prefill_pagetable, llm_decode_pagetable = prefill(
            requests=beam_requests,
            model=self.model.model,  # 使用内部的模型
            tokenizer=self.tokenizer,
            num_beams=self.config.beam_size,
            length_penalty=1.0,
            speech_pagetable=self.model.speech_pagetable,
            llm_prefill_pagetable=self.model.llm_prefill_pagetable,
            llm_decode_pagetable=self.model.llm_decode_pagetable
        )
        
        # 🔥 关键修复：更新pagetable状态并验证连续性
        self.model.speech_pagetable = speech_pagetable
        self.model.llm_prefill_pagetable = llm_prefill_pagetable
        self.model.llm_decode_pagetable = llm_decode_pagetable
        
        # 验证pagetable状态
        self._verify_pagetable_consistency("Prefill", speech_pagetable, llm_prefill_pagetable, llm_decode_pagetable)
        
        # 转换结果并更新每个request的cache引用
        results = []
        for i, (orig_req, processed_req) in enumerate(zip(requests, processed_requests)):
            result = self._convert_beam_result_to_inference_result(orig_req, processed_req, is_prefill=True)
            
            # 🔥 ORCA关键：立即更新原始request的cache引用，转换为列表格式
            # 根据infinisst_faster.py，cache应该是列表格式
            orig_req.speech_cache = [processed_req.speech_cache]  # 转换为列表
            
            # 🔥 关键修复：prefill完成后，llm_cache应该已经是beam cache列表
            # 不需要再包装一层列表
            if isinstance(processed_req.llm_cache, list):
                # prefill返回的已经是beam cache列表，直接使用
                orig_req.past_key_values = [processed_req.llm_cache]  # 外层列表用于session管理
                print(f"🔍 [ORCA-CACHE] Request {orig_req.request_id} prefill完成，保存beam cache列表 (共{len(processed_req.llm_cache)}个beam)")
            else:
                # 如果不是列表，按单个cache处理（不应该发生）
                orig_req.past_key_values = [[processed_req.llm_cache]]
                print(f"⚠️ [ORCA-CACHE] Request {orig_req.request_id} prefill返回单个cache，包装为beam列表")
            
            # 🔥 关键修复：保存beam_state到原始request
            if hasattr(processed_req, 'beam_state'):
                orig_req.beam_state = processed_req.beam_state
                print(f"🔍 [ORCA-CACHE] 保存beam_state到request {orig_req.request_id}")
            
            results.append(result)
        
        print(f"🔍 [ORCA-PREFILL] Batch完成: {len(results)} 个结果")
        return results
            
        # except Exception as e:
        #     logger.error(f"Prefill batch处理失败: {e}")
        #     # 返回错误结果
        #     return [
        #         {
        #             'request_id': req.request_id,
        #             'success': False,
        #             'error': str(e),
        #             'generated_text': '',
        #             'generated_tokens': [],
        #             'prefill_finished': False
        #         }
        #         for req in requests
        #     ]
    
    def _process_decode_batch(self, requests: List[InferenceRequest]) -> List[Dict[str, Any]]:
        """处理decode阶段的请求 - ORCA风格，一次只生成一个token"""
        try:
            # 🔥 ORCA架构：为batch中的每个request分别构造beam_search.Request
            beam_requests = []
            for req in requests:
                beam_req = self._create_beam_request(req)
                beam_requests.append(beam_req)
            
            print(f"🔍 [ORCA-DECODE] 处理batch: {len(beam_requests)} 个requests")
            
            # 直接调用beam_search的decode函数
            from model.flashinfer.beam_search import decode
            
            processed_requests, speech_pagetable, llm_prefill_pagetable, llm_decode_pagetable = decode(
                requests=beam_requests,
                model=self.model.model,  # 使用内部的模型
                tokenizer=self.tokenizer,
                num_beams=self.config.beam_size,
                length_penalty=1.0,
                speech_pagetable=self.model.speech_pagetable,
                llm_prefill_pagetable=self.model.llm_prefill_pagetable,
                llm_decode_pagetable=self.model.llm_decode_pagetable
            )
            
            # 🔥 关键修复：更新pagetable状态并验证连续性
            self.model.speech_pagetable = speech_pagetable
            self.model.llm_prefill_pagetable = llm_prefill_pagetable
            self.model.llm_decode_pagetable = llm_decode_pagetable
            
            # 验证pagetable状态
            self._verify_pagetable_consistency("Decode", speech_pagetable, llm_prefill_pagetable, llm_decode_pagetable)
            
            # 转换结果并更新每个request的cache引用
            results = []
            for i, (orig_req, processed_req) in enumerate(zip(requests, processed_requests)):
                result = self._convert_beam_result_to_inference_result(orig_req, processed_req, is_prefill=False)
                
                # 🔥 ORCA关键：立即更新原始request的cache引用
                # Decode阶段：根据processed_req的状态决定cache格式
                if hasattr(processed_req, 'decode_finished') and processed_req.decode_finished:
                    # 如果decode完成，转换为单个cache
                    orig_req.speech_cache = [processed_req.speech_cache]
                    orig_req.past_key_values = [processed_req.llm_cache]  
                    print(f"🔍 [ORCA-CACHE] Request {orig_req.request_id} decode完成，cache转换为单个格式")
                else:
                    # 如果decode未完成，保持beam cache列表格式
                    orig_req.speech_cache = [processed_req.speech_cache]
                    if isinstance(processed_req.llm_cache, list):
                        orig_req.past_key_values = processed_req.llm_cache  # 保持beam列表
                        print(f"🔍 [ORCA-CACHE] Request {orig_req.request_id} decode继续，保持beam cache列表 ({len(processed_req.llm_cache)}个beam)")
                    else:
                        orig_req.past_key_values = [processed_req.llm_cache]
                        print(f"🔍 [ORCA-CACHE] Request {orig_req.request_id} decode继续，cache转换为列表格式")
                
                # 🔥 关键修复：保存beam_state到原始request
                if hasattr(processed_req, 'beam_state'):
                    orig_req.beam_state = processed_req.beam_state
                    print(f"🔍 [ORCA-CACHE] 保存beam_state到request {orig_req.request_id}")
                
                results.append(result)
            
            print(f"🔍 [ORCA-DECODE] Batch完成: {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"Decode batch处理失败: {e}")
            # 返回错误结果
            return [
                {
                    'request_id': req.request_id,
                    'success': False,
                    'error': str(e),
                    'generated_text': '',
                    'generated_tokens': [],
                    'decode_finished': False
                }
                for req in requests
            ]
    
    def _create_beam_request(self, request: InferenceRequest):
        """将InferenceRequest转换为beam_search的Request格式"""
        from model.flashinfer.beam_search import Request
        from model.flashinfer.engine import SpeechCache, LLMCache
        from agents.infinisst import S2TAgentStates
        
        # 🔥 关键修复：创建S2TAgentStates对象，让model的_prepare_speech和_prepare_inputs方法处理
        states = S2TAgentStates(
            speech_cache=request.speech_cache,
            past_key_values=request.past_key_values,
            target_ids=getattr(request, 'target_ids', []),
            segment_idx=getattr(request, 'segment_idx', 0),
            translations_list=getattr(request, 'translations_list', [])
        )
        
        # 转换为list格式（S2TAgentStates期望的格式）
        states.source = request.speech_batch
        states.source_finished = getattr(request, 'is_final', False)
        states.source_sample_rate = 16000
        # 🔥 直接调用model的prepare方法，就像infinisst_faster.policy()那样
        speech_batch = self.model._prepare_speech(states)
        input_ids = self.model._prepare_inputs(states)
        
        print(f"🔧 [PREPARE-DATA] 调用model._prepare_speech和_prepare_inputs完成:")
        print(f"   - speech_batch shape: {speech_batch.shape}")
        print(f"   - input_ids shape: {input_ids.shape}")
        
        # 确保数据维度正确
        if speech_batch.dim() == 2:
            speech_batch = speech_batch[0]  # [1, seq_len] -> [seq_len]
        if input_ids.dim() == 2:
            input_ids = input_ids[0]  # [1, seq_len] -> [seq_len]
        
        # 🔥 关键修复：正确处理cache结构
        # 根据infinisst_faster.py，states.speech_cache和states.past_key_values是列表
        # 在ORCA架构中，每个request对应一个cache条目
        cache_index = 0  # 每个request使用第一个cache（在ORCA中每个request都是独立的）
        
        if states.speech_cache is None:
            speech_cache_for_request = None
        else:
            # 如果是列表，取指定索引；如果不是列表，直接使用
            if isinstance(states.speech_cache, list):
                speech_cache_for_request = states.speech_cache[cache_index] if len(states.speech_cache) > cache_index else None
            else:
                speech_cache_for_request = states.speech_cache
        
        if states.past_key_values is None:
            past_key_values_for_request = None
        else:
            # 🔥 关键修复：Decode阶段需要特殊处理
            if request.stage == RequestStage.DECODE:
                # Decode阶段：past_key_values应该是beam cache列表
                if isinstance(states.past_key_values, list) and len(states.past_key_values) > cache_index:
                    # 🔥 修复：检查第一个元素是否是LLMCache对象来判断是否为beam cache列表
                    first_element = states.past_key_values[0]
                    if hasattr(first_element, 'paged_kv_indices'):
                        # 第一个元素是LLMCache对象，说明这就是beam cache列表
                        past_key_values_for_request = states.past_key_values
                        print(f"🔍 [DECODE-CACHE] 识别为beam cache列表，长度: {len(states.past_key_values)}")
                    else:
                        # 第一个元素不是LLMCache，需要进一步解析
                        past_key_values_cache = states.past_key_values[cache_index]
                        if isinstance(past_key_values_cache, list):
                            # 检查嵌套列表的第一个元素
                            if len(past_key_values_cache) > 0 and hasattr(past_key_values_cache[0], 'paged_kv_indices'):
                                # 这是beam cache列表，直接使用
                                past_key_values_for_request = past_key_values_cache
                                print(f"🔍 [DECODE-CACHE] 使用嵌套beam cache列表，长度: {len(past_key_values_cache)}")
                            else:
                                # 这是外层包装列表，需要进一步解析
                                if len(past_key_values_cache) > 0 and isinstance(past_key_values_cache[0], list):
                                    # 双层包装：[[beam_cache_1, beam_cache_2, ...]]
                                    past_key_values_for_request = past_key_values_cache[0]
                                    print(f"🔍 [DECODE-CACHE] 解析双层包装，beam cache列表长度: {len(past_key_values_for_request)}")
                                else:
                                    # 单个cache被包装：[single_cache]
                                    past_key_values_for_request = past_key_values_cache
                                    print(f"⚠️ [DECODE-CACHE] 检测到单cache包装，长度: {len(past_key_values_cache)}")
                        else:
                            # 单个cache，需要包装成列表（这种情况不应该发生在正确的prefill之后）
                            past_key_values_for_request = [past_key_values_cache]
                            print(f"⚠️ [DECODE-CACHE] 单个cache包装为列表")
                else:
                    past_key_values_for_request = None
                    print(f"⚠️ [DECODE-CACHE] 无法获取cache")
            else:
                # Prefill阶段：单个cache
                if isinstance(states.past_key_values, list):
                    past_key_values_for_request = states.past_key_values[cache_index] if len(states.past_key_values) > cache_index else None
                else:
                    past_key_values_for_request = states.past_key_values
        
        print(f"🔍 [BEAM-CACHE] Cache状态:")
        print(f"   - speech_cache类型: {type(states.speech_cache)}, 长度: {len(states.speech_cache) if isinstance(states.speech_cache, list) else 'N/A'}")
        print(f"   - past_key_values类型: {type(states.past_key_values)}, 长度: {len(states.past_key_values) if isinstance(states.past_key_values, list) else 'N/A'}")
        print(f"   - 使用cache索引: {cache_index}")
        print(f"   - speech_cache_for_request: {speech_cache_for_request is not None}")
        print(f"   - past_key_values_for_request类型: {type(past_key_values_for_request)}")
        if isinstance(past_key_values_for_request, list):
            print(f"   - past_key_values_for_request长度: {len(past_key_values_for_request)}")
        else:
            print(f"   - past_key_values_for_request: {past_key_values_for_request is not None}")
        
        # 🔥 关键修复：按照infinisst_faster.py的Request构造方式
        beam_req = Request(
            input_ids.view(-1),  # 按照原始代码：input_ids.view(-1)
            speech_batch.view(-1),  # 按照原始代码：speech_batch.view(-1)
            self.model.latency_multiplier * self.model.blocksize,  # blocksize参数
            request.max_new_tokens,  # max_new_tokens
            
            # speech相关参数
            self.model_args.max_cache_size,  # speech_max_steps
            speech_cache_for_request,  # speech_cache
            
            # LLM相关参数  
            self.model_args.max_llm_cache_size,  # llm_max_steps
            getattr(self.model, 'system_prompt_size', 0),  # llm_max_steps_start
            past_key_values_for_request  # llm_cache
        )
        
        # 设置状态 - 根据request.stage判断是否已经prefill
        beam_req.prefill_finished = (request.stage == RequestStage.DECODE)
        beam_req.decode_finished = False
        
        # 🔥 关键修复：正确设置beam_state
        if request.stage == RequestStage.DECODE and hasattr(request, 'beam_state') and request.beam_state is not None:
            # Decode阶段：恢复保存的beam_state
            beam_req.beam_state = request.beam_state
            print(f"🔍 [BEAM-STATE] 恢复decode阶段的beam_state for {request.request_id}")
        else:
            # Prefill阶段：设置为None，将由beam_search.prefill()创建
            beam_req.beam_state = None
            print(f"🔍 [BEAM-STATE] Prefill阶段，beam_state将被创建 for {request.request_id}")
        
        print(f"🔍 [BEAM-REQUEST] Created beam request for {request.request_id}")
        print(f"   - Speech shape: {speech_batch.shape}")
        print(f"   - Input IDs shape: {input_ids.shape}")
        print(f"   - Max new tokens: {beam_req.max_new_tokens}")
        print(f"   - Blocksize: {self.model.latency_multiplier * self.model.blocksize}")
        
        return beam_req
    
    def _convert_beam_result_to_inference_result(self, orig_request: InferenceRequest, 
                                               processed_request, is_prefill: bool) -> Dict[str, Any]:
        """将beam_search的结果转换为InferenceResult格式"""
        
        result = {
            'request_id': orig_request.request_id,
            'success': True,
            'generated_text': '',
            'generated_tokens': [],
            'finished': False,
            'speech_cache': processed_request.speech_cache,
            'past_key_values': processed_request.llm_cache
        }
        
        if is_prefill:
            # Prefill阶段完成
            result['prefill_finished'] = processed_request.prefill_finished
            result['decode_finished'] = False
            
            # Prefill通常不生成文本，只是准备beam状态
            if hasattr(processed_request, 'beam_state') and processed_request.beam_state:
                beam_state = processed_request.beam_state
                if hasattr(beam_state, 'generated_ids') and beam_state.generated_ids is not None:
                    # 获取初始的beam candidates
                    first_tokens = beam_state.generated_ids[:, 0].tolist()  # 第一个token
                    result['generated_tokens'] = first_tokens
                    
                    # 尝试解码第一个token
                    if len(first_tokens) > 0:
                        try:
                            decoded_text = self.tokenizer.decode([first_tokens[0]], skip_special_tokens=True)
                            result['generated_text'] = decoded_text
                            print(f"🔍 [PREFILL-RESULT] Generated first token: {first_tokens[0]} -> '{decoded_text}'")
                        except Exception as e:
                            print(f"⚠️ [PREFILL-RESULT] Failed to decode token {first_tokens[0]}: {e}")
            
            print(f"🔍 [PREFILL-RESULT] Request {orig_request.request_id} prefill完成")
            
        else:
            # Decode阶段 - 生成了新的token
            result['prefill_finished'] = True
            result['decode_finished'] = processed_request.decode_finished
            
            # 🔥 修复：检查beam_state是否为None
            if hasattr(processed_request, 'beam_state') and processed_request.beam_state is not None:
                beam_state = processed_request.beam_state
                if hasattr(beam_state, 'generated_ids') and beam_state.generated_ids is not None:
                    # 获取当前最佳beam的所有token
                    if len(beam_state.generated_ids) > 0:
                        best_sequence = beam_state.generated_ids[0].tolist()  # 取第一个beam
                        result['generated_tokens'] = best_sequence
                        
                        # 解码完整序列
                        try:
                            decoded_text = self.tokenizer.decode(best_sequence, skip_special_tokens=True)
                            
                            # 🔥 关键修复：后处理生成的文本，过滤掉prompt格式token
                            filtered_text = self._filter_prompt_tokens(decoded_text)
                            result['generated_text'] = filtered_text
                            
                            print(f"🔍 [DECODE-RESULT] Generated sequence: {best_sequence} -> '{decoded_text}'")
                            print(f"🔍 [DECODE-RESULT] Filtered translation: '{filtered_text}'")
                        except Exception as e:
                            print(f"⚠️ [DECODE-RESULT] Failed to decode sequence {best_sequence}: {e}")
                            result['generated_text'] = ""
                    
                    # 检查是否完成
                    result['finished'] = processed_request.decode_finished
                else:
                    print(f"⚠️ [DECODE-RESULT] beam_state.generated_ids is None or missing")
                    result['finished'] = True  # 如果beam_state有问题，标记为完成避免无限循环
            else:
                print(f"⚠️ [DECODE-RESULT] beam_state is None or missing")
                result['finished'] = True  # 如果beam_state为None，标记为完成
            
            # 检查是否有最终结果
            if hasattr(processed_request, 'results') and processed_request.results:
                # 如果已经有最终结果
                final_result = processed_request.results
                if isinstance(final_result, dict) and 'sequence' in final_result:
                    sequence = final_result['sequence']
                    result['generated_tokens'] = sequence
                    
                    try:
                        decoded_text = self.tokenizer.decode(sequence, skip_special_tokens=True)
                        result['generated_text'] = decoded_text
                        result['finished'] = True
                        print(f"🔍 [DECODE-FINAL] Final result: {sequence} -> '{decoded_text}'")
                    except Exception as e:
                        print(f"⚠️ [DECODE-FINAL] Failed to decode final sequence {sequence}: {e}")
                        
            print(f"🔍 [DECODE-RESULT] Request {orig_request.request_id} decode step完成, finished={result['finished']}")
        
        return result

    def _filter_prompt_tokens(self, text: str) -> str:
        """
        过滤掉prompt格式token，只保留真正的翻译内容
        
        主要过滤的格式token包括：
        - <speech>, <|user|>, <|assistant|>, <|startofprev|>, <|endofprev|> 等
        - 换行符和多余的空格
        """
        if not text:
            return ""
        
        # 需要过滤的格式token模式
        format_tokens = [
            '<speech>',
            '<|user|>',
            '<|assistant|>', 
            '<|startofprev|>',
            '<|endofprev|>',
            '<|start_header_id|>',
            '<|end_header_id|>',
            '<|eot_id|>',
            '<sp_patch>',
            '<|',
            '|>',
            'Translate the following speech',
            'from English to Chinese',
            'from English to Italian',
            'from English to German', 
            'from English to Spanish'
        ]
        
        # 移除格式token
        filtered_text = text
        for token in format_tokens:
            filtered_text = filtered_text.replace(token, '')
        
        # 清理多余的空白字符
        filtered_text = filtered_text.strip()
        
        # 移除连续的换行符和空格
        import re
        filtered_text = re.sub(r'\s+', ' ', filtered_text)
        filtered_text = filtered_text.strip()
        
        # 🔥 特殊处理：如果结果只包含格式字符（如'<'），返回空字符串
        if filtered_text in ['<', '><', '|', '>', ''] or filtered_text.isspace():
            filtered_text = ""
        
        # 🔥 检查是否还在生成prompt格式
        if any(pattern in filtered_text.lower() for pattern in ['translate', 'speech', 'english', 'chinese']):
            # 如果还包含这些关键词，说明还在生成prompt，返回空字符串
            filtered_text = ""
        
        return filtered_text

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
    
    def _cleanup_finished_sessions(self, requests: List[InferenceRequest], results: List[Dict[str, Any]]):
        """清理已结束的session的KV cache页面"""
        try:
            for i, request in enumerate(requests):
                if i < len(results):
                    result = results[i]
                    
                    # 检查session是否结束（翻译完成或出错）
                    session_finished = (
                        not result.get('success', False) or  # 出错了
                        result.get('finished', False) or     # 明确标记完成
                        getattr(request, 'is_final', False)   # 是最后一个请求
                    )
                    
                    if session_finished:
                        logger.info(f"🧹 Session结束，开始清理KV cache页面: {request.request_id}")
                        self._cleanup_session_kv_cache(request)
                        
        except Exception as e:
            logger.error(f"清理session时出错: {e}")
    
    def _cleanup_session_kv_cache(self, request: InferenceRequest):
        """清理单个session的KV cache页面"""
        try:
            # 这里需要访问具体的KV cache数据结构
            # 假设request中包含了KV cache的引用
            
            session_id = getattr(request, 'session_id', request.request_id)
            
            # 🔥 关键：释放speech cache页面
            if hasattr(request, 'speech_cache') and request.speech_cache:
                self._release_speech_cache_pages(request.speech_cache, session_id)
            
            # 🔥 关键：释放LLM KV cache页面
            if hasattr(request, 'past_key_values') and request.past_key_values:
                self._release_llm_cache_pages(request.past_key_values, session_id)
            
            logger.info(f"✅ Session {session_id} KV cache页面清理完成")
            
        except Exception as e:
            logger.error(f"清理session {request.request_id} KV cache时出错: {e}")
    
    def _release_speech_cache_pages(self, speech_cache, session_id: str):
        """释放speech cache占用的页面"""
        try:
            # 这里需要根据实际的speech cache结构来实现
            # 假设speech_cache包含页面索引信息
            
            if hasattr(speech_cache, 'paged_kv_indices') and speech_cache.paged_kv_indices:
                pages_to_release = len(speech_cache.paged_kv_indices)
                
                # 调用页面释放函数（需要从flashinfer引擎获取pagetable）
                if hasattr(self.model, 'speech_pagetable'):
                    pagetable = self.model.speech_pagetable
                    self._release_pages_to_pool(pagetable, speech_cache.paged_kv_indices, session_id, 'speech')
                    
                    # 清空cache中的页面引用
                    speech_cache.paged_kv_indices = []
                    speech_cache.paged_kv_last_page_len = 16  # PAGE_SIZE
                    
                    logger.info(f"🔄 释放了 {pages_to_release} 个speech cache页面到页面池")
                    
        except Exception as e:
            logger.error(f"释放speech cache页面时出错: {e}")
    
    def _release_llm_cache_pages(self, past_key_values, session_id: str):
        """释放LLM KV cache占用的页面"""
        try:
            # 这里需要根据实际的past_key_values结构来实现
            
            if hasattr(past_key_values, 'paged_kv_indices') and past_key_values.paged_kv_indices:
                pages_to_release = len(past_key_values.paged_kv_indices)
                
                # 分别处理prefill和decode cache
                if hasattr(self.model, 'llm_prefill_pagetable'):
                    self._release_pages_to_pool(self.model.llm_prefill_pagetable, 
                                              past_key_values.paged_kv_indices, 
                                              session_id, 'llm_prefill')
                
                if hasattr(self.model, 'llm_decode_pagetable'):
                    self._release_pages_to_pool(self.model.llm_decode_pagetable, 
                                              past_key_values.paged_kv_indices, 
                                              session_id, 'llm_decode')
                
                # 清空cache中的页面引用
                past_key_values.paged_kv_indices = []
                past_key_values.paged_kv_last_page_len = 16  # PAGE_SIZE
                
                logger.info(f"🔄 释放了 {pages_to_release} 个LLM cache页面到页面池")
                
        except Exception as e:
            logger.error(f"释放LLM cache页面时出错: {e}")
    
    def _release_pages_to_pool(self, pagetable, page_indices: list, session_id: str, cache_type: str):
        """将页面释放回页面池"""
        try:
            if not page_indices:
                return
            
            import torch
            
            # 减少页面引用计数
            page_indices_tensor = torch.tensor(page_indices, dtype=torch.long)
            pagetable.page_cnt[page_indices_tensor] -= 1
            
            # 找出引用计数为0的页面（可以被释放）
            free_mask = pagetable.page_cnt[page_indices_tensor] == 0
            free_pages = page_indices_tensor[free_mask]
            
            if len(free_pages) > 0:
                # 将页面放回可用队列
                free_pages_list = free_pages.tolist()
                pagetable.paged_queue.extend(free_pages_list)
                
                logger.info(f"🔄 [{cache_type}] Session {session_id} 释放了 {len(free_pages_list)} 个页面回页面池")
                logger.info(f"🔄 [{cache_type}] 页面池现在有 {len(pagetable.paged_queue)} 个可用页面")
                
                # 🔍 详细记录页面使用情况
                total_pages = len(pagetable.page_cnt)
                used_pages = torch.sum(pagetable.page_cnt > 0).item()
                logger.info(f"📊 [{cache_type}] 页面使用统计: {used_pages}/{total_pages} 页被使用")
            else:
                logger.warning(f"⚠️ [{cache_type}] Session {session_id} 的 {len(page_indices)} 个页面仍被其他session引用")
                
        except Exception as e:
            logger.error(f"释放页面到池时出错: {e}")
    
    def force_cleanup_all_sessions(self):
        """强制清理所有session的KV cache（紧急情况使用）"""
        try:
            logger.warning("🚨 强制清理所有session的KV cache页面")
            
            # 重置所有页面池到初始状态
            if hasattr(self.model, 'speech_pagetable'):
                self._reset_pagetable(self.model.speech_pagetable, 'speech')
            
            if hasattr(self.model, 'llm_prefill_pagetable'):
                self._reset_pagetable(self.model.llm_prefill_pagetable, 'llm_prefill')
            
            if hasattr(self.model, 'llm_decode_pagetable'):
                self._reset_pagetable(self.model.llm_decode_pagetable, 'llm_decode')
            
            logger.info("✅ 强制清理完成，所有页面已重置")
            
        except Exception as e:
            logger.error(f"强制清理时出错: {e}")
    
    def _reset_pagetable(self, pagetable, cache_type: str):
        """重置页面表到初始状态"""
        try:
            total_pages = len(pagetable.page_cnt)
            
            # 重置页面引用计数
            pagetable.page_cnt.zero_()
            
            # 重建可用页面队列
            pagetable.paged_queue = list(range(total_pages))
            
            logger.info(f"🔄 [{cache_type}] 页面表已重置: {total_pages} 个页面全部可用")
            
        except Exception as e:
            logger.error(f"重置页面表时出错: {e}")

class MultiGPUInferenceEngine:
    """
    多GPU推理引擎管理器
    管理多个GPU上的推理引擎实例
    """
    
    def __init__(self, gpu_language_map: Dict[int, str], model_args_map: Dict[int, Any] = None):
        """
        初始化多GPU推理引擎
        
        Args:
            gpu_language_map: GPU到语言对的映射
            model_args_map: GPU到模型参数的映射（可选）
        """
        self.gpu_language_map = gpu_language_map
        self.model_args_map = model_args_map or {}
        
        # 创建引擎实例
        self.engines: Dict[int, InferenceEngine] = {}
        for gpu_id, language_pair in gpu_language_map.items():
            model_args = self.model_args_map.get(gpu_id, {})
            
            engine = InferenceEngine(
                model_args=model_args,
                gpu_id=gpu_id,
                language_id=language_pair
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
        with torch.cuda.device("cuda:"+str(gpu_id)):
            res = engine.process_batch(requests)
        return res
    
    def get_all_stats(self) -> Dict[int, Dict[str, Any]]:
        """获取所有引擎的统计信息"""
        return {gpu_id: engine.get_stats() for gpu_id, engine in self.engines.items()}

# 在推理引擎类中添加验证方法
def _verify_pagetable_consistency(engine, stage_name: str, speech_pagetable, llm_prefill_pagetable, llm_decode_pagetable):
    """验证pagetable状态的连续性"""
    try:
        print(f"🔍 [PAGETABLE-VERIFY] {stage_name} 阶段后 pagetable 状态:")
        
        # 检查speech pagetable
        if hasattr(speech_pagetable, 'paged_queue'):
            available_speech_pages = len(speech_pagetable.paged_queue)
            total_speech_pages = len(speech_pagetable.page_cnt)
            used_speech_pages = torch.sum(speech_pagetable.page_cnt > 0).item()
            print(f"   - Speech: {used_speech_pages}/{total_speech_pages} 页被使用, {available_speech_pages} 页可用")
        
        # 检查LLM prefill pagetable
        if hasattr(llm_prefill_pagetable, 'paged_queue'):
            available_prefill_pages = len(llm_prefill_pagetable.paged_queue)
            total_prefill_pages = len(llm_prefill_pagetable.page_cnt)
            used_prefill_pages = torch.sum(llm_prefill_pagetable.page_cnt > 0).item()
            print(f"   - LLM Prefill: {used_prefill_pages}/{total_prefill_pages} 页被使用, {available_prefill_pages} 页可用")
        
        # 检查LLM decode pagetable
        if hasattr(llm_decode_pagetable, 'paged_queue'):
            available_decode_pages = len(llm_decode_pagetable.paged_queue)
            total_decode_pages = len(llm_decode_pagetable.page_cnt)
            used_decode_pages = torch.sum(llm_decode_pagetable.page_cnt > 0).item()
            print(f"   - LLM Decode: {used_decode_pages}/{total_decode_pages} 页被使用, {available_decode_pages} 页可用")
        
        print(f"✅ [PAGETABLE-VERIFY] {stage_name} pagetable状态验证完成")
        
    except Exception as e:
        print(f"⚠️ [PAGETABLE-VERIFY] {stage_name} pagetable验证失败: {e}")

# 将验证方法添加到InferenceEngine类中
InferenceEngine._verify_pagetable_consistency = lambda self, stage_name, speech_pt, llm_prefill_pt, llm_decode_pt: _verify_pagetable_consistency(self, stage_name, speech_pt, llm_prefill_pt, llm_decode_pt) 