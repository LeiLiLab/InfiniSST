#!/usr/bin/env python3
"""
InfiniSST API服务 - 整合版
整合scheduler、engine和前端，实现基于ORCA思路的多请求并发推理服务
"""

import asyncio
import logging
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
from threading import Thread

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import numpy as np
import torch

# 导入调度器和相关组件
from scheduler import LLMScheduler, RequestStage, UserSession, InferenceRequest

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InfiniSSTAPIWithScheduler:
    """
    整合版InfiniSST API服务
    支持scheduler调度、多GPU、并发推理
    """
    
    def __init__(self, 
                 gpu_language_map: Dict[int, str] = None,
                 max_batch_size: int = 32,
                 batch_timeout: float = 0.1):
        """
        初始化API服务
        
        Args:
            gpu_language_map: GPU到语言对的映射，例如 {0: "en-zh", 1: "en-de"}
            max_batch_size: 最大批处理大小
            batch_timeout: 批处理超时时间
        """
        # 默认GPU语言映射
        if gpu_language_map is None:
            gpu_language_map = {0: "English -> Chinese"}
        
        self.gpu_language_map = gpu_language_map
        
        # 创建调度器
        class Args:
            def __init__(self, max_batch_size, batch_timeout):
                self.max_batch_size = max_batch_size
                self.batch_timeout = batch_timeout
                self.session_timeout = 3600
            
        self.scheduler = LLMScheduler(gpu_language_map, Args(max_batch_size, batch_timeout))
        
        # 初始化Flask应用
        self.app = Flask(__name__)
        CORS(self.app)
        
        # 状态管理
        self.is_running = False
        self.models_loaded = {}  # {gpu_id: model_instance}
        
        # 注册路由
        self._register_routes()
        
        logger.info(f"InfiniSST API初始化完成，支持语言对: {list(gpu_language_map.values())}")
    
    def _register_routes(self):
        """注册API路由"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查"""
            return jsonify({
                "status": "healthy",
                "scheduler_running": self.scheduler.is_running,
                "supported_languages": self.scheduler.get_supported_languages(),
                "gpu_mapping": self.gpu_language_map
            })
        
        @self.app.route('/load_model', methods=['POST'])
        def load_model():
            """加载模型（模拟实现）"""
            try:
                data = request.get_json()
                gpu_id = data.get('gpu_id', 0)
                language_pair = data.get('language_pair', 'English -> Chinese')
                
                # 这里应该实际加载模型，现在先模拟
                self.models_loaded[gpu_id] = {
                    'language_pair': language_pair,
                    'loaded_at': time.time(),
                    'status': 'loaded'
                }
                
                logger.info(f"模型已加载到GPU {gpu_id}，语言对: {language_pair}")
                
                return jsonify({
                    "success": True,
                    "message": f"模型已加载到GPU {gpu_id}",
                    "gpu_id": gpu_id,
                    "language_pair": language_pair
                })
                
            except Exception as e:
                logger.error(f"加载模型失败: {e}")
                return jsonify({"success": False, "error": str(e)}), 500
        
        @self.app.route('/translate', methods=['POST'])
        def translate():
            """处理翻译请求"""
            try:
                data = request.get_json()
                
                # 提取请求参数
                user_id = data.get('user_id', f'user_{int(time.time())}')
                language_pair = data.get('language_pair', 'English -> Chinese')
                audio_data = data.get('audio_data', [])
                is_final = data.get('is_final', False)
                max_new_tokens = data.get('max_new_tokens', 20)
                
                # 验证输入
                if not audio_data:
                    return jsonify({"success": False, "error": "缺少音频数据"}), 400
                
                if language_pair not in self.scheduler.get_supported_languages():
                    return jsonify({
                        "success": False, 
                        "error": f"不支持的语言对: {language_pair}"
                    }), 400
                
                # 转换音频数据
                if isinstance(audio_data, list):
                    speech_data = torch.tensor(audio_data, dtype=torch.float32)
                else:
                    speech_data = torch.tensor(audio_data, dtype=torch.float32)
                
                # 提交到调度器
                request_id = self.scheduler.submit_request(
                    user_id=user_id,
                    language_id=language_pair,
                    speech_data=speech_data,
                    stage=RequestStage.PREFILL,
                    is_final=is_final,
                    max_new_tokens=max_new_tokens
                )
                
                logger.info(f"翻译请求已提交: {request_id}")
                
                return jsonify({
                    "success": True,
                    "request_id": request_id,
                    "user_id": user_id,
                    "language_pair": language_pair,
                    "status": "processing"
                })
                
            except Exception as e:
                logger.error(f"翻译请求处理失败: {e}")
                return jsonify({"success": False, "error": str(e)}), 500
        
        @self.app.route('/session/<user_id>/<language_id>', methods=['GET'])
        def get_session_info(user_id, language_id):
            """获取用户会话信息"""
            try:
                session_info = self.scheduler.get_session_info(user_id, language_id)
                if session_info:
                    return jsonify({
                        "success": True,
                        "session_info": session_info
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": "会话不存在"
                    }), 404
                    
            except Exception as e:
                logger.error(f"获取会话信息失败: {e}")
                return jsonify({"success": False, "error": str(e)}), 500
        
        @self.app.route('/session/<user_id>/<language_id>/reset', methods=['POST'])
        def reset_session(user_id, language_id):
            """重置用户会话"""
            try:
                success = self.scheduler.reset_session(user_id, language_id)
                if success:
                    return jsonify({
                        "success": True,
                        "message": "会话已重置"
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": "会话不存在"
                    }), 404
                    
            except Exception as e:
                logger.error(f"重置会话失败: {e}")
                return jsonify({"success": False, "error": str(e)}), 500
        
        @self.app.route('/stats', methods=['GET'])
        def get_stats():
            """获取系统统计信息"""
            try:
                queue_stats = self.scheduler.get_queue_stats()
                return jsonify({
                    "success": True,
                    "scheduler_stats": queue_stats,
                    "models_loaded": self.models_loaded,
                    "uptime": time.time() - getattr(self, 'start_time', time.time())
                })
                
            except Exception as e:
                logger.error(f"获取统计信息失败: {e}")
                return jsonify({"success": False, "error": str(e)}), 500
        
        @self.app.route('/stream/<request_id>', methods=['GET'])
        def stream_results(request_id):
            """流式返回翻译结果"""
            def generate():
                # 这里应该实现流式结果返回逻辑
                # 现在先返回模拟数据
                for i in range(10):
                    yield f"data: {json.dumps({'chunk': f'翻译片段 {i}', 'final': i == 9})}\n\n"
                    time.sleep(0.1)
            
            return Response(generate(), mimetype='text/plain')
    
    def start_scheduler(self):
        """启动调度器"""
        if not self.scheduler.is_running:
            self.scheduler.start()
            logger.info("调度器已启动")
    
    def stop_scheduler(self):
        """停止调度器"""
        if self.scheduler.is_running:
            self.scheduler.stop()
            logger.info("调度器已停止")
    
    def run(self, host='0.0.0.0', port=8000, debug=False):
        """运行API服务"""
        self.start_time = time.time()
        
        # 启动调度器
        self.start_scheduler()
        
        try:
            logger.info(f"InfiniSST API服务启动，监听 {host}:{port}")
            self.app.run(host=host, port=port, debug=debug, threaded=True)
        except KeyboardInterrupt:
            logger.info("收到停止信号")
        finally:
            self.stop_scheduler()
            logger.info("API服务已停止")

def create_api_server():
    """创建API服务实例"""
    # 配置GPU语言映射
    gpu_language_map = {
        0: "English -> Chinese",
        # 可以添加更多GPU和语言对
        # 1: "English -> German",
    }
    
    return InfiniSSTAPIWithScheduler(
        gpu_language_map=gpu_language_map,
        max_batch_size=32,
        batch_timeout=0.1
    )

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='InfiniSST API服务器 - 整合版')
    parser.add_argument('--host', default='0.0.0.0', help='服务器地址')
    parser.add_argument('--port', type=int, default=8000, help='服务器端口')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    # 创建并运行API服务
    api_server = create_api_server()
    api_server.run(host=args.host, port=args.port, debug=args.debug) 