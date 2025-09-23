#!/usr/bin/env python3
"""
调度器系统测试脚本
用于验证调度器系统是否正常工作，能否处理多个并发请求
"""

import requests
import time
import asyncio
import websockets
import json
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class SchedulerSystemTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        
    def test_health_check(self):
        """测试健康检查，验证调度器状态"""
        print("🔍 测试健康检查...")
        
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            
            data = response.json()
            print("📊 系统状态:")
            print(f"   - 总状态: {data.get('status')}")
            print(f"   - 调度器可用: {data.get('scheduler_available')}")
            print(f"   - 调度器启用: {data.get('scheduler_enabled')}")
            print(f"   - 活跃会话: {data.get('active_sessions')}")
            
            if 'scheduler' in data:
                scheduler_info = data['scheduler']
                print(f"   - 调度器运行: {scheduler_info.get('running')}")
                print(f"   - 支持语言: {scheduler_info.get('supported_languages')}")
                
            if 'session_breakdown' in data:
                breakdown = data['session_breakdown']
                print(f"   - 传统会话: {breakdown.get('traditional_sessions')}")
                print(f"   - 调度器会话: {breakdown.get('scheduler_sessions')}")
                
            return data.get('status') == 'healthy' and data.get('scheduler_enabled', False)
            
        except Exception as e:
            print(f"❌ 健康检查失败: {e}")
            return False
    
    def create_session(self, client_id=None):
        """创建新的翻译会话"""
        print(f"🚀 创建新会话 (client_id: {client_id})...")
        
        try:
            params = {
                "agent_type": "InfiniSST",
                "language_pair": "English -> Chinese",
                "latency_multiplier": 2
            }
            
            if client_id:
                params["client_id"] = client_id
                
            response = requests.post(f"{self.base_url}/init", params=params)
            response.raise_for_status()
            
            data = response.json()
            session_id = data.get('session_id')
            scheduler_based = data.get('scheduler_based', False)
            
            print(f"✅ 会话创建成功:")
            print(f"   - Session ID: {session_id}")
            print(f"   - 基于调度器: {scheduler_based}")
            print(f"   - 排队状态: {data.get('queued', False)}")
            
            return session_id, scheduler_based
            
        except Exception as e:
            print(f"❌ 创建会话失败: {e}")
            return None, False
    
    def test_websocket_connection(self, session_id, test_duration=10):
        """测试WebSocket连接和音频处理"""
        print(f"🔌 测试WebSocket连接 {session_id}...")
        
        async def websocket_test():
            uri = f"{self.ws_url}/wss/{session_id}"
            
            try:
                async with websockets.connect(uri) as websocket:
                    print(f"✅ WebSocket 连接成功")
                    
                    # 等待READY消息
                    ready_message = await websocket.recv()
                    print(f"📩 收到消息: {ready_message}")
                    
                    # 发送几个音频数据块
                    for i in range(5):
                        # 生成模拟音频数据 (1秒的音频，16kHz)
                        audio_data = np.random.uniform(-0.1, 0.1, 16000).astype(np.float32)
                        
                        # 发送音频数据
                        await websocket.send(audio_data.tobytes())
                        print(f"📤 发送音频块 {i+1}/5")
                        
                        # 等待响应（如果有的话）
                        try:
                            response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                            print(f"📨 收到翻译: {response}")
                        except asyncio.TimeoutError:
                            print(f"⏰ 块 {i+1} 暂无响应")
                        
                        await asyncio.sleep(1)
                    
                    # 发送EOF
                    await websocket.send("EOF")
                    print("📤 发送EOF信号")
                    
                    # 等待最终响应
                    try:
                        final_response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        print(f"📨 最终响应: {final_response}")
                    except asyncio.TimeoutError:
                        print("⏰ 未收到最终响应")
                        
                    return True
                    
            except Exception as e:
                print(f"❌ WebSocket测试失败: {e}")
                return False
        
        # 运行异步测试
        try:
            return asyncio.run(websocket_test())
        except Exception as e:
            print(f"❌ WebSocket连接失败: {e}")
            return False
    
    def test_concurrent_sessions(self, num_sessions=3):
        """测试并发会话处理"""
        print(f"🔀 测试 {num_sessions} 个并发会话...")
        
        def create_and_test_session(session_idx):
            client_id = f"test_client_{session_idx}"
            session_id, scheduler_based = self.create_session(client_id)
            
            if not session_id:
                return f"Session {session_idx}: 创建失败"
            
            if not scheduler_based:
                return f"Session {session_idx}: 未使用调度器系统"
            
            # 简单的WebSocket测试
            success = self.test_websocket_connection(session_id, test_duration=5)
            
            return f"Session {session_idx}: {'成功' if success else '失败'}"
        
        # 并发执行测试
        with ThreadPoolExecutor(max_workers=num_sessions) as executor:
            futures = [executor.submit(create_and_test_session, i) for i in range(num_sessions)]
            
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"📊 {result}")
        
        return results
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🧪 开始调度器系统测试...")
        print("=" * 50)
        
        # 测试1: 健康检查
        health_ok = self.test_health_check()
        if not health_ok:
            print("❌ 健康检查失败，停止测试")
            return False
        
        print("\n" + "=" * 50)
        
        # 测试2: 单个会话
        session_id, scheduler_based = self.create_session("test_single")
        if not scheduler_based:
            print("⚠️ 未使用调度器系统，可能回退到传统模式")
        
        if session_id:
            self.test_websocket_connection(session_id)
        
        print("\n" + "=" * 50)
        
        # 测试3: 并发会话（这是关键测试）
        concurrent_results = self.test_concurrent_sessions(3)
        
        print("\n" + "=" * 50)
        print("📋 测试总结:")
        print(f"   - 健康检查: {'✅' if health_ok else '❌'}")
        print(f"   - 单个会话: {'✅' if session_id else '❌'}")
        print(f"   - 调度器启用: {'✅' if scheduler_based else '❌'}")
        print(f"   - 并发测试: {len([r for r in concurrent_results if '成功' in r])}/{len(concurrent_results)} 成功")
        
        return True

if __name__ == "__main__":
    tester = SchedulerSystemTester()
    tester.run_all_tests() 