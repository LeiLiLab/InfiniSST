#!/usr/bin/env python3
"""
快速检查scheduler状态
"""
import requests
import json

def quick_check():
    try:
        # 检查diagnose接口
        response = requests.get("http://localhost:8000/diagnose", timeout=2)
        if response.status_code == 200:
            data = response.json()
            
            print("🔍 当前Scheduler状态:")
            
            # 队列信息
            queue_stats = data.get('queue_stats', {})
            detailed_info = queue_stats.get('detailed_queue_info', {})
            
            total_prefill = 0
            total_decode = 0
            
            for gpu_id, info in detailed_info.items():
                prefill = info.get('prefill_queue_size', 0)
                decode = info.get('decode_queue_size', 0)
                total_prefill += prefill
                total_decode += decode
                language = info.get('language', 'unknown')
                
                print(f"  GPU {gpu_id} ({language}): {prefill}P + {decode}D = {prefill+decode} total")
                
                # 估算页面使用 
                estimated_pages = prefill * 20 + decode * 5
                print(f"    估算页面使用: ~{estimated_pages} 页")
            
            print(f"\n📊 总计: {total_prefill} PREFILL + {total_decode} DECODE = {total_prefill + total_decode} requests")
            
            # Session信息
            active_sessions = queue_stats.get('active_sessions', 0)
            inactive_count = queue_stats.get('inactive_session_count', 0)
            print(f"👥 Sessions: {active_sessions} 活跃, {inactive_count} 不活跃")
            
            # 卡住的session
            stuck_sessions = data.get('stuck_sessions', [])
            if stuck_sessions:
                print(f"🚨 卡住的Sessions: {len(stuck_sessions)}")
                for session in stuck_sessions[:3]:  # 只显示前3个
                    print(f"  - {session.get('user_id', 'unknown')} 不活跃 {session.get('inactive_seconds', 0):.1f}s")
            
            # 瓶颈分析
            if total_prefill + total_decode > 32:
                print("🚨 瓶颈: 队列积压严重")
            elif total_decode > 20:
                print("⚠️ 瓶颈: DECODE队列积压")
            elif total_prefill + total_decode > 0:
                print("✅ 队列正常运行")
            else:
                print("💤 系统空闲")
                
        else:
            print(f"❌ API错误: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 连接失败: {e}")
        print("提示: 确保服务器在 http://localhost:8000 运行")

if __name__ == "__main__":
    quick_check() 