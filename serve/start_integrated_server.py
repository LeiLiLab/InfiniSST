#!/usr/bin/env python3
"""
InfiniSST 整合服务启动脚本
将scheduler、inference engine和API服务串联起来，实现完整的多请求并发推理系统
"""

import argparse
import logging
import sys
import os
import signal
import time
import threading
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入相关模块
from serve.api_with_scheduler import InfiniSSTAPIWithScheduler
from serve.inference_engine import MultiGPUInferenceEngine, EngineConfig
from serve.scheduler import LLMScheduler

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('infinisst_integrated.log')
    ]
)
logger = logging.getLogger(__name__)

class InfiniSSTIntegratedServer:
    """
    InfiniSST 整合服务器
    管理scheduler、inference engine和API服务的生命周期
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化整合服务器
        
        Args:
            config: 服务器配置
        """
        self.config = config
        self.is_running = False
        
        # 组件实例
        self.api_server = None
        self.inference_engine = None
        self.scheduler = None
        
        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("InfiniSST整合服务器初始化完成")
    
    def _signal_handler(self, signum, frame):
        """处理停止信号"""
        logger.info(f"接收到信号 {signum}，开始停止服务...")
        self.stop()
        sys.exit(0)
    
    def _create_model_args(self, gpu_id: int) -> Any:
        """创建模型参数（mock实现）"""
        # 这里应该根据实际需求创建模型参数
        # 现在先返回一个mock参数对象
        class MockModelArgs:
            def __init__(self):
                self.model_name = "mock_model"
                self.gpu_id = gpu_id
                self.batch_size = 32
                self.max_new_tokens = 20
                self.beam_size = 4
                # 添加更多需要的参数...
        
        return MockModelArgs()
    
    def start(self):
        """启动整合服务器"""
        if self.is_running:
            logger.warning("服务器已在运行")
            return
        
        try:
            logger.info("开始启动InfiniSST整合服务器...")
            
            # 1. 创建推理引擎
            self._create_inference_engine()
            
            # 2. 创建和配置API服务器
            self._create_api_server()
            
            # 3. 连接组件
            self._connect_components()
            
            # 4. 启动所有组件
            self._start_components()
            
            self.is_running = True
            logger.info("🚀 InfiniSST整合服务器启动成功！")
            
            # 启动API服务器（阻塞）
            self._run_api_server()
            
        except Exception as e:
            logger.error(f"启动服务器失败: {e}")
            self.stop()
            raise
    
    def _create_inference_engine(self):
        """创建推理引擎"""
        logger.info("创建推理引擎...")
        
        gpu_language_map = self.config.get('gpu_language_map', {0: "English -> Chinese"})
        
        # 创建模型参数映射
        model_args_map = {}
        for gpu_id in gpu_language_map.keys():
            model_args_map[gpu_id] = self._create_model_args(gpu_id)
        
        # 创建多GPU推理引擎
        self.inference_engine = MultiGPUInferenceEngine(
            gpu_language_map=gpu_language_map,
            model_args_map=model_args_map
        )
        
        # 加载模型（如果启用）
        if self.config.get('load_models', False):
            logger.info("加载模型...")
            success = self.inference_engine.load_all_models()
            if not success:
                logger.warning("部分模型加载失败，将使用模拟推理")
        else:
            logger.info("跳过模型加载，将使用模拟推理")
        
        logger.info("推理引擎创建完成")
    
    def _create_api_server(self):
        """创建API服务器"""
        logger.info("创建API服务器...")
        
        gpu_language_map = self.config.get('gpu_language_map', {0: "English -> Chinese"})
        
        self.api_server = InfiniSSTAPIWithScheduler(
            gpu_language_map=gpu_language_map,
            max_batch_size=self.config.get('max_batch_size', 32),
            batch_timeout=self.config.get('batch_timeout', 0.1)
        )
        
        self.scheduler = self.api_server.scheduler
        logger.info("API服务器创建完成")
    
    def _connect_components(self):
        """连接各个组件"""
        logger.info("连接系统组件...")
        
        # 将推理引擎连接到调度器
        if self.scheduler and self.inference_engine:
            self.scheduler.set_inference_engine(self.inference_engine)
            logger.info("推理引擎已连接到调度器")
        
        logger.info("组件连接完成")
    
    def _start_components(self):
        """启动所有组件"""
        logger.info("启动系统组件...")
        
        # 启动推理引擎
        if self.inference_engine:
            self.inference_engine.start_all()
            logger.info("推理引擎已启动")
        
        # 启动调度器
        if self.scheduler:
            self.scheduler.start()
            logger.info("调度器已启动")
        
        logger.info("所有组件启动完成")
    
    def _run_api_server(self):
        """运行API服务器"""
        if self.api_server:
            host = self.config.get('host', '0.0.0.0')
            port = self.config.get('port', 8000)
            debug = self.config.get('debug', False)
            
            logger.info(f"启动API服务器，监听 {host}:{port}")
            self.api_server.run(host=host, port=port, debug=debug)
    
    def stop(self):
        """停止整合服务器"""
        if not self.is_running:
            return
        
        logger.info("开始停止InfiniSST整合服务器...")
        
        # 停止推理引擎
        if self.inference_engine:
            self.inference_engine.stop_all()
            logger.info("推理引擎已停止")
        
        # 停止调度器
        if self.scheduler:
            self.scheduler.stop()
            logger.info("调度器已停止")
        
        self.is_running = False
        logger.info("InfiniSST整合服务器已停止")
    
    def get_status(self) -> Dict[str, Any]:
        """获取服务器状态"""
        status = {
            'is_running': self.is_running,
            'config': self.config,
            'scheduler_stats': None,
            'inference_engine_stats': None
        }
        
        if self.scheduler:
            status['scheduler_stats'] = self.scheduler.get_queue_stats()
        
        if self.inference_engine:
            status['inference_engine_stats'] = self.inference_engine.get_all_stats()
        
        return status

def create_default_config() -> Dict[str, Any]:
    """创建默认配置"""
    return {
        'gpu_language_map': {
            0: "English -> Chinese",
            # 可以添加更多GPU和语言对
            # 1: "English -> German",
        },
        'max_batch_size': 32,
        'batch_timeout': 0.1,
        'load_models': False,  # 是否加载实际模型
        'host': '0.0.0.0',
        'port': 8000,
        'debug': False
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='InfiniSST整合服务器')
    
    # 服务器配置
    parser.add_argument('--host', default='0.0.0.0', help='服务器地址')
    parser.add_argument('--port', type=int, default=8000, help='服务器端口')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    # 推理配置
    parser.add_argument('--max-batch-size', type=int, default=32, help='最大批处理大小')
    parser.add_argument('--batch-timeout', type=float, default=0.1, help='批处理超时时间')
    parser.add_argument('--load-models', action='store_true', help='加载实际模型（默认使用模拟）')
    
    # GPU配置
    parser.add_argument('--gpus', type=str, default='0', help='使用的GPU ID列表，逗号分隔')
    parser.add_argument('--languages', type=str, default='English -> Chinese', 
                       help='语言对列表，逗号分隔，与GPU对应')
    
    args = parser.parse_args()
    
    # 解析GPU和语言映射
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    languages = [x.strip() for x in args.languages.split(',')]
    
    if len(gpu_ids) != len(languages):
        logger.error("GPU数量和语言对数量不匹配")
        sys.exit(1)
    
    gpu_language_map = dict(zip(gpu_ids, languages))
    
    # 创建配置
    config = create_default_config()
    config.update({
        'gpu_language_map': gpu_language_map,
        'max_batch_size': args.max_batch_size,
        'batch_timeout': args.batch_timeout,
        'load_models': args.load_models,
        'host': args.host,
        'port': args.port,
        'debug': args.debug
    })
    
    # 创建并启动服务器
    server = InfiniSSTIntegratedServer(config)
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("接收到键盘中断信号")
    except Exception as e:
        logger.error(f"服务器运行出错: {e}")
        sys.exit(1)
    finally:
        server.stop()

if __name__ == '__main__':
    main() 