#!/usr/bin/env python3
"""
远程服务器增量处理修复测试脚本
"""
import requests
import json
import base64
import websocket
import time
import threading
from urllib.parse import quote_plus

def test_incremental_processing():
    print("🧪 测试增量处理修复效果")
    
    # 测试数据
    session_id = "test_fix_incremental"
    
    # 模拟多个音频片段
    audio_chunks = [
        b"Hello world chunk 1",
        b"Hello world chunk 2", 
        b"Hello world chunk 3",
        b"Hello world chunk 4"
    ]
    
    print("📝 步骤1: 创建session")
    init_url = f"http://localhost:8000/init?session_id={session_id}&agent_type=scheduler&language_pair=en-zh&init_audio_data={base64.b64encode(audio_chunks[0]).decode()}"
    
    try:
        response = requests.get(init_url, timeout=10)
        print(f"   初始化响应: {response.status_code}")
        if response.status_code != 200:
            print(f"   错误: {response.text}")
            return False
        
        print("📝 步骤2: 建立WebSocket连接")
        ws_url = f"ws://localhost:8000/ws/{quote_plus(session_id)}"
        
        # 收集所有翻译结果
        results = []
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if data.get('type') == 'translation':
                    result = data.get('translation', '')
                    results.append(result)
                    print(f"   📤 收到翻译结果 #{len(results)}: '{result}'")
                elif data.get('type') == 'error':
                    print(f"   ❌ 收到错误: {data.get('message', 'Unknown error')}")
            except Exception as e:
                print(f"   ⚠️ 解析消息失败: {e}")
        
        def on_error(ws, error):
            print(f"   ❌ WebSocket错误: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print(f"   🔌 WebSocket连接关闭")
        
        def on_open(ws):
            print(f"   ✅ WebSocket连接建立")
            
            # 发送后续音频片段
            def send_chunks():
                for i, chunk in enumerate(audio_chunks[1:], 2):
                    time.sleep(2)  # 间隔2秒发送
                    audio_data = base64.b64encode(chunk).decode()
                    print(f"   🎵 发送音频片段 #{i}")
                    ws.send(audio_data)
                
                # 等待5秒然后关闭连接
                time.sleep(5)
                ws.close()
            
            threading.Thread(target=send_chunks, daemon=True).start()
        
        # 创建WebSocket连接
        ws = websocket.WebSocketApp(ws_url,
                                  on_message=on_message,
                                  on_error=on_error,
                                  on_close=on_close,
                                  on_open=on_open)
        
        print("🔍 步骤3: 发送音频数据并监听翻译结果")
        ws.run_forever(ping_interval=30, ping_timeout=10)
        
        print("📊 测试结果分析:")
        print(f"   - 总共收到 {len(results)} 个翻译结果")
        print(f"   - 发送了 {len(audio_chunks)} 个音频片段")
        
        if len(results) >= 2:
            print("   - 翻译结果:")
            for i, result in enumerate(results, 1):
                print(f"     #{i}: '{result}'")
            
            # 检查是否有增量更新
            if len(set(results)) > 1:
                print("   ✅ 成功：发现不同的翻译结果，增量处理正常工作")
                return True
            else:
                print("   ❌ 失败：所有翻译结果都相同，增量处理仍有问题")
                return False
        else:
            print("   ❌ 失败：收到的翻译结果太少")
            return False
            
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_incremental_processing()
    if success:
        print("🎉 增量处理修复测试通过！")
    else:
        print("😞 增量处理修复测试失败") 