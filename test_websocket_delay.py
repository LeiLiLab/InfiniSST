#!/usr/bin/env python3
"""
测试WebSocket连接时间和延迟计算的修复
"""

import asyncio
import aiohttp
import numpy as np
import time
import json

async def test_evaluation_session():
    """测试evaluation模式的session创建和WebSocket连接"""
    
    # 1. 创建evaluation session
    print("🔧 创建evaluation session...")
    
    async with aiohttp.ClientSession() as session:
        # 初始化session
        init_params = {
            "agent_type": "InfiniSST",
            "language_pair": "English -> Chinese",
            "latency_multiplier": 2,
            "client_id": "test_user_001",
            "evaluation_mode": "true"
        }
        
        async with session.post("http://localhost:8000/init", params=init_params) as response:
            if response.status != 200:
                print(f"❌ 初始化失败: {response.status}")
                return
            
            data = await response.json()
            session_id = data["session_id"]
            print(f"✅ Session创建成功: {session_id}")
            print(f"   - Scheduler based: {data.get('scheduler_based', False)}")
            print(f"   - Evaluation mode: {data.get('evaluation_mode', False)}")
        
        # 2. 建立WebSocket连接
        print(f"\n🔗 建立WebSocket连接...")
        ws_url = f"ws://localhost:8000/wss/{session_id}"
        
        try:
            async with session.ws_connect(ws_url) as ws:
                print(f"✅ WebSocket连接成功")
                
                # 等待READY消息
                msg = await ws.receive()
                if msg.type == aiohttp.WSMsgType.TEXT:
                    print(f"📨 收到消息: {msg.data}")
                
                # 3. 发送测试音频数据
                print(f"\n🎵 发送测试音频数据...")
                
                # 创建简单的测试音频数据 (1秒，16kHz)
                sample_rate = 16000
                duration = 1.0  # 1秒
                samples = int(sample_rate * duration)
                
                # 生成简单的正弦波测试音频
                audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples)).astype(np.float32)
                
                # 分块发送
                chunk_size = 4096
                chunks_sent = 0
                
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i+chunk_size]
                    await ws.send_bytes(chunk.tobytes())
                    chunks_sent += 1
                    print(f"📤 发送chunk {chunks_sent}: {len(chunk)} samples")
                    
                    # 等待一小段时间模拟实时音频
                    await asyncio.sleep(0.1)
                    
                    # 尝试接收翻译结果
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=0.1)
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            print(f"📥 收到翻译: {msg.data}")
                    except asyncio.TimeoutError:
                        pass
                
                # 4. 发送EOF信号
                print(f"\n📋 发送EOF信号...")
                await ws.send_str("EOF")
                
                # 等待处理完成
                timeout = 10
                start_wait = time.time()
                
                while time.time() - start_wait < timeout:
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            text = msg.data
                            print(f"📥 收到消息: {text}")
                            
                            if text.startswith("PROCESSING_COMPLETE"):
                                print(f"✅ 处理完成")
                                break
                        elif msg.type == aiohttp.WSMsgType.CLOSE:
                            print(f"🔌 WebSocket连接关闭")
                            break
                    except asyncio.TimeoutError:
                        continue
                
                print(f"\n✅ 测试完成")
                
        except Exception as e:
            print(f"❌ WebSocket连接失败: {e}")

async def main():
    """主函数"""
    print("🧪 开始WebSocket延迟计算测试...")
    await test_evaluation_session()

if __name__ == "__main__":
    asyncio.run(main()) 