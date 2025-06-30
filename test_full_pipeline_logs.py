#!/usr/bin/env python3
"""
全链路日志测试脚本
用于验证重复prefill问题的修复效果
"""

import asyncio
import websockets
import json
import numpy as np
import time
import sys
from urllib.parse import quote  # 添加URL编码

# 添加必要的依赖检查和导入
try:
    import aiohttp
except ImportError:
    print("❌ 需要安装aiohttp: pip install aiohttp")
    sys.exit(1)

async def check_server_status():
    """检查服务器状态"""
    
    # 尝试不同的端口
    ports_to_try = [5000, 8000, 8080, 3000]
    
    for port in ports_to_try:
        try:
            print(f"🔍 尝试连接到 localhost:{port}...")
            async with aiohttp.ClientSession() as session:
                # 先尝试简单的健康检查
                try:
                    async with session.get(f'http://localhost:{port}/health', timeout=3) as response:
                        if response.status == 200:
                            print(f"✅ 服务器运行在端口 {port}")
                            # 更新全局端口变量
                            global SERVER_PORT
                            SERVER_PORT = port
                            return True
                        else:
                            print(f"⚠️ 端口 {port} 响应状态: HTTP {response.status}")
                except aiohttp.ClientConnectorError:
                    print(f"❌ 端口 {port} 连接失败")
                    continue
                except asyncio.TimeoutError:
                    print(f"⏰ 端口 {port} 连接超时")
                    continue
                
        except Exception as e:
            print(f"❌ 端口 {port} 检查失败: {e}")
            continue
    
    print("❌ 尝试所有端口都失败，服务器可能未启动")
    return False

# 全局变量存储服务器端口
SERVER_PORT = 5000

async def create_session(session_base_name: str) -> str:
    """通过API创建session"""
    async with aiohttp.ClientSession() as session:
        # 构建查询参数
        params = {
            "agent_type": "infinisst_faster",  # 或者 "infinisst"
            "language_pair": "English -> Chinese",
            "latency_multiplier": 2,
            "client_id": session_base_name
        }
        
        try:
            async with session.post(f'http://localhost:{SERVER_PORT}/init', params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    session_id = result.get('session_id')
                    print(f"✅ 创建session成功: {session_id}")
                    print(f"   - Agent类型: {params['agent_type']}")
                    print(f"   - 语言对: {params['language_pair']}")
                    print(f"   - 调度器模式: {result.get('scheduler_based', 'unknown')}")
                    print(f"   - 排队状态: {result.get('queued', 'unknown')}")
                    return session_id
                else:
                    print(f"❌ 创建session失败: HTTP {response.status}")
                    response_text = await response.text()
                    print(f"   - 响应: {response_text}")
                    return None
        except Exception as e:
            print(f"❌ 创建session时出错: {e}")
            return None

async def test_single_session_behavior():
    """测试单个session的正常行为，观察是否还有重复prefill"""
    
    print("🚀 开始测试单个session的行为...")
    print("=" * 60)
    
    # 先创建session
    session_id = await create_session("test_session_123")
    if not session_id:
        print("❌ 无法创建session，跳过测试")
        return
    
    # 🔥 修复：对session ID进行URL编码
    encoded_session_id = quote(session_id, safe='')
    uri = f"ws://localhost:{SERVER_PORT}/wss/{encoded_session_id}"
    
    print(f"🔗 连接信息:")
    print(f"   - 原始session ID: {session_id}")
    print(f"   - 编码后session ID: {encoded_session_id}")
    print(f"   - WebSocket URI: {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"✅ WebSocket连接成功!")
            
            # 等待初始化消息
            try:
                init_message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                print(f"📨 收到初始化消息: {init_message}")
            except asyncio.TimeoutError:
                print("⏰ 10秒内没有收到初始化消息，继续测试...")
            
            # 生成测试音频数据
            sample_rate = 16000
            duration = 0.5  # 0.5秒的音频
            audio_samples = int(sample_rate * duration)
            
            print(f"🎵 准备发送音频数据:")
            print(f"   - 采样率: {sample_rate} Hz")
            print(f"   - 时长: {duration} 秒")
            print(f"   - 样本数: {audio_samples}")
            
            # 发送多个音频片段，观察行为
            for chunk_num in range(3):  # 减少到3个片段，更容易观察
                print(f"\n📤 发送第 {chunk_num + 1} 个音频片段...")
                
                # 生成正弦波音频数据
                frequency = 440 + chunk_num * 100  # 不同频率
                t = np.linspace(0, duration, audio_samples, False)
                audio_data = (np.sin(2 * np.pi * frequency * t) * 0.1).astype(np.float32)
                
                print(f"   - 频率: {frequency} Hz")
                print(f"   - 数据范围: {audio_data.min():.6f} ~ {audio_data.max():.6f}")
                print(f"   - 发送时间: {time.strftime('%H:%M:%S.%f')[:-3]}")
                
                # 发送音频数据
                await websocket.send(audio_data.tobytes())
                
                # 等待一段时间观察处理
                await asyncio.sleep(2)
                
                # 尝试接收响应
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    print(f"📥 收到响应: {response}")
                except asyncio.TimeoutError:
                    print("⏰ 5秒内无响应")
                
                print(f"   - 片段 {chunk_num + 1} 处理完成")
            
            print(f"\n🏁 测试完成，等待最终响应...")
            
            # 等待可能的延迟响应
            try:
                while True:
                    response = await asyncio.wait_for(websocket.recv(), timeout=8.0)
                    print(f"📥 最终响应: {response}")
            except asyncio.TimeoutError:
                print("⏰ 8秒内无更多响应")
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

async def test_multiple_quick_requests():
    """测试快速连续请求，这种情况最容易触发重复prefill"""
    
    print("\n🔥 开始测试快速连续请求...")
    print("=" * 60)
    
    # 先创建session
    session_id = await create_session("test_rapid_123")
    if not session_id:
        print("❌ 无法创建session，跳过测试")
        return
    
    # 🔥 修复：对session ID进行URL编码
    encoded_session_id = quote(session_id, safe='')
    uri = f"ws://localhost:{SERVER_PORT}/wss/{encoded_session_id}"
    
    print(f"🔗 连接信息:")
    print(f"   - 原始session ID: {session_id}")
    print(f"   - 编码后session ID: {encoded_session_id}")
    print(f"   - WebSocket URI: {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"✅ WebSocket连接成功!")
            
            # 等待初始化
            try:
                init_message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"📨 收到初始化消息: {init_message}")
            except asyncio.TimeoutError:
                print("⏰ 没有收到初始化消息，继续测试...")
            
            # 快速发送多个音频片段
            sample_rate = 16000
            duration = 0.2  # 稍长一点的片段
            audio_samples = int(sample_rate * duration)
            
            print(f"🎵 准备快速发送 {audio_samples} 样本的音频片段...")
            
            for i in range(3):
                print(f"\n📤 快速发送第 {i + 1} 个片段 (无等待)...")
                
                # 生成简单的音频数据
                audio_data = (np.random.normal(0, 0.05, audio_samples)).astype(np.float32)
                
                print(f"   - 片段 {i + 1}: {audio_samples} 样本")
                print(f"   - 发送时间: {time.strftime('%H:%M:%S.%f')[:-3]}")
                
                # 立即发送，不等待
                await websocket.send(audio_data.tobytes())
                
                # 稍微等待一下，避免过快
                await asyncio.sleep(0.5)
            
            print(f"\n⏰ 快速发送完成，等待处理结果...")
            
            # 等待处理结果
            try:
                responses = []
                while len(responses) < 5:  # 最多等待5个响应
                    response = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                    responses.append(response)
                    print(f"📥 响应 {len(responses)}: {response}")
            except asyncio.TimeoutError:
                print(f"⏰ 超时，共收到 {len(responses)} 个响应")
                
    except Exception as e:
        print(f"❌ 快速请求测试失败: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """主函数"""
    print("🧪 全链路日志测试脚本")
    print("🎯 目标：验证重复prefill问题的修复效果")
    print("📋 测试计划：")
    print("   1. 检查服务器状态")
    print("   2. 测试单个session的正常行为")
    print("   3. 测试快速连续请求")
    print("   4. 观察日志输出是否正常")
    print()
    
    # 步骤1：检查服务器状态
    print("步骤 1: 检查服务器状态")
    if not await check_server_status():
        print("❌ 服务器不可用，请先启动服务器")
        print("   启动命令: cd /path/to/InfiniSST && python serve/api.py")
        return
    
    print(f"✅ 服务器在端口 {SERVER_PORT} 正常运行")
    print()
    
    # 步骤2：测试单个session的正常行为
    print("步骤 2: 测试单个session的正常行为")
    await test_single_session_behavior()
    
    # 稍等片刻
    print("\n⏸️ 等待 3 秒后继续下一个测试...")
    await asyncio.sleep(3)
    
    # 步骤3：测试快速连续请求
    print("\n步骤 3: 测试快速连续请求")
    await test_multiple_quick_requests()
    
    print("\n🏁 全链路日志测试完成！")
    print("🔍 请查看服务器日志是否存在：")
    print("   - 重复的 'already prefilled' 错误")
    print("   - 队列中的重复PREFILL请求")
    print("   - 不合理的内存使用模式")
    print()
    print("📋 正常情况下应该看到：")
    print("   - 第一个音频片段触发PREFILL")
    print("   - 后续音频片段触发DECODE")
    print("   - 没有重复prefill警告")
    print("   - 翻译结果正常返回")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试脚本执行失败: {e}")
        import traceback
        traceback.print_exc() 