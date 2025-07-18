#!/usr/bin/env python3
"""
InfiniSST 整合系统测试脚本
测试从前端发请求 → scheduler调度 → 模型生成 → 返回结果的完整链路
"""

import asyncio
import json
import time
import requests
import logging
import numpy as np
from typing import List, Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InfiniSSTSystemTester:
    """InfiniSST系统测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        初始化测试器
        
        Args:
            base_url: API服务器的基础URL
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def test_health_check(self) -> bool:
        """测试健康检查接口"""
        logger.info("🔍 测试健康检查接口...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"✅ 健康检查成功: {data}")
            
            # 检查关键字段
            assert data.get('status') == 'healthy'
            assert 'supported_languages' in data
            assert 'scheduler_running' in data
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 健康检查失败: {e}")
            return False
    
    def test_load_model(self) -> bool:
        """测试模型加载接口"""
        logger.info("🔍 测试模型加载接口...")
        
        try:
            payload = {
                "gpu_id": 0,
                "language_pair": "English -> Chinese"
            }
            
            response = self.session.post(f"{self.base_url}/load_model", json=payload)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"✅ 模型加载成功: {data}")
            
            assert data.get('success') is True
            assert data.get('gpu_id') == 0
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            return False
    
    def test_single_translation(self) -> bool:
        """测试单个翻译请求"""
        logger.info("🔍 测试单个翻译请求...")
        
        try:
            # 生成模拟音频数据
            audio_data = self._generate_mock_audio()
            
            payload = {
                "user_id": "test_user_1",
                "language_pair": "English -> Chinese",
                "audio_data": audio_data,
                "is_final": True,
                "max_new_tokens": 20
            }
            
            response = self.session.post(f"{self.base_url}/translate", json=payload)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"✅ 翻译请求成功: {data}")
            
            assert data.get('success') is True
            assert 'request_id' in data
            assert data.get('user_id') == "test_user_1"
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 翻译请求失败: {e}")
            return False
    
    def test_concurrent_translations(self, num_requests: int = 5) -> bool:
        """测试并发翻译请求"""
        logger.info(f"🔍 测试 {num_requests} 个并发翻译请求...")
        
        try:
            import threading
            import concurrent.futures
            
            results = []
            
            def send_translation_request(user_id: str):
                audio_data = self._generate_mock_audio()
                payload = {
                    "user_id": user_id,
                    "language_pair": "English -> Chinese",
                    "audio_data": audio_data,
                    "is_final": True,
                    "max_new_tokens": 20
                }
                
                response = self.session.post(f"{self.base_url}/translate", json=payload)
                response.raise_for_status()
                return response.json()
            
            # 使用线程池发送并发请求
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
                futures = []
                for i in range(num_requests):
                    future = executor.submit(send_translation_request, f"test_user_{i}")
                    futures.append(future)
                
                # 收集结果
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
                    logger.info(f"收到响应: {result.get('request_id', 'unknown')}")
            
            logger.info(f"✅ {len(results)} 个并发请求全部成功")
            
            # 验证所有请求都成功
            for result in results:
                assert result.get('success') is True
                assert 'request_id' in result
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 并发翻译测试失败: {e}")
            return False
    
    def test_session_management(self) -> bool:
        """测试会话管理功能"""
        logger.info("🔍 测试会话管理功能...")
        
        try:
            user_id = "test_session_user"
            language_id = "English -> Chinese"
            
            # 1. 发送翻译请求创建会话
            audio_data = self._generate_mock_audio()
            payload = {
                "user_id": user_id,
                "language_pair": language_id,
                "audio_data": audio_data,
                "is_final": False,
                "max_new_tokens": 20
            }
            
            response = self.session.post(f"{self.base_url}/translate", json=payload)
            response.raise_for_status()
            
            # 2. 获取会话信息
            response = self.session.get(f"{self.base_url}/session/{user_id}/{language_id}")
            response.raise_for_status()
            
            session_data = response.json()
            logger.info(f"会话信息: {session_data}")
            
            assert session_data.get('success') is True
            assert 'session_info' in session_data
            
            # 3. 重置会话
            response = self.session.post(f"{self.base_url}/session/{user_id}/{language_id}/reset")
            response.raise_for_status()
            
            reset_data = response.json()
            logger.info(f"重置结果: {reset_data}")
            
            assert reset_data.get('success') is True
            
            logger.info("✅ 会话管理测试成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ 会话管理测试失败: {e}")
            return False
    
    def test_system_stats(self) -> bool:
        """测试系统统计信息"""
        logger.info("🔍 测试系统统计信息...")
        
        try:
            response = self.session.get(f"{self.base_url}/stats")
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"系统统计: {json.dumps(data, indent=2, ensure_ascii=False)}")
            
            assert data.get('success') is True
            assert 'scheduler_stats' in data
            
            logger.info("✅ 系统统计测试成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ 系统统计测试失败: {e}")
            return False
    
    def test_streaming_translation(self) -> bool:
        """测试流式翻译（模拟）"""
        logger.info("🔍 测试流式翻译...")
        
        try:
            # 发送多个非最终的音频片段
            user_id = "stream_test_user"
            language_pair = "English -> Chinese"
            
            # 发送3个音频片段
            for i in range(3):
                audio_data = self._generate_mock_audio(length=1000)  # 更短的音频片段
                payload = {
                    "user_id": user_id,
                    "language_pair": language_pair,
                    "audio_data": audio_data,
                    "is_final": i == 2,  # 最后一个片段标记为最终
                    "max_new_tokens": 10
                }
                
                response = self.session.post(f"{self.base_url}/translate", json=payload)
                response.raise_for_status()
                
                result = response.json()
                logger.info(f"片段 {i+1} 响应: {result.get('request_id', 'unknown')}")
                
                assert result.get('success') is True
                
                # 短暂等待
                time.sleep(0.1)
            
            logger.info("✅ 流式翻译测试成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ 流式翻译测试失败: {e}")
            return False
    
    def _generate_mock_audio(self, length: int = 16000) -> List[float]:
        """生成模拟音频数据"""
        # 生成简单的正弦波作为模拟音频
        sample_rate = 16000
        duration = length / sample_rate
        t = np.linspace(0, duration, length, False)
        frequency = 440  # A4音符
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        return audio.tolist()
    
    def run_all_tests(self) -> bool:
        """运行所有测试"""
        logger.info("🚀 开始运行InfiniSST系统完整测试...")
        
        tests = [
            ("健康检查", self.test_health_check),
            ("模型加载", self.test_load_model),
            ("单个翻译", self.test_single_translation),
            ("并发翻译", self.test_concurrent_translations),
            ("会话管理", self.test_session_management),
            ("系统统计", self.test_system_stats),
            ("流式翻译", self.test_streaming_translation),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"执行测试: {test_name}")
            logger.info(f"{'='*50}")
            
            try:
                if test_func():
                    passed += 1
                    logger.info(f"✅ {test_name} - 通过")
                else:
                    logger.error(f"❌ {test_name} - 失败")
            except Exception as e:
                logger.error(f"❌ {test_name} - 异常: {e}")
            
            time.sleep(1)  # 测试间隔
        
        logger.info(f"\n{'='*60}")
        logger.info(f"测试完成: {passed}/{total} 个测试通过")
        logger.info(f"{'='*60}")
        
        if passed == total:
            logger.info("🎉 所有测试通过！系统工作正常。")
            return True
        else:
            logger.warning(f"⚠️ {total - passed} 个测试失败，请检查系统配置。")
            return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='InfiniSST系统测试器')
    parser.add_argument('--url', default='http://localhost:8000', help='API服务器URL')
    parser.add_argument('--test', choices=[
        'health', 'model', 'single', 'concurrent', 'session', 'stats', 'stream', 'all'
    ], default='all', help='要执行的测试')
    
    args = parser.parse_args()
    
    tester = InfiniSSTSystemTester(args.url)
    
    # 等待服务器启动
    logger.info("等待服务器启动...")
    for i in range(10):
        try:
            response = requests.get(f"{args.url}/health", timeout=5)
            if response.status_code == 200:
                break
        except:
            pass
        time.sleep(2)
        logger.info(f"重试连接... ({i+1}/10)")
    else:
        logger.error("无法连接到服务器，请确保服务器已启动")
        return 1
    
    # 执行测试
    if args.test == 'all':
        success = tester.run_all_tests()
    elif args.test == 'health':
        success = tester.test_health_check()
    elif args.test == 'model':
        success = tester.test_load_model()
    elif args.test == 'single':
        success = tester.test_single_translation()
    elif args.test == 'concurrent':
        success = tester.test_concurrent_translations()
    elif args.test == 'session':
        success = tester.test_session_management()
    elif args.test == 'stats':
        success = tester.test_system_stats()
    elif args.test == 'stream':
        success = tester.test_streaming_translation()
    
    return 0 if success else 1

if __name__ == '__main__':
    exit_code = main()
    exit(exit_code) 