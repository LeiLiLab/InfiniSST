#!/usr/bin/env python3
"""
修复版快速检查scheduler状态
支持ngrok URL并正确解析数据
"""
import requests
import json
import sys

def quick_check(server_url="https://infinisst.ngrok.app"):
    """快速检查scheduler状态"""
    print(f"🔍 检查Scheduler状态 - {server_url}")
    print("=" * 60)
    
    try:
        # 1. 健康检查
        print("1️⃣ 健康检查...")
        health_response = requests.get(f"{server_url}/health", timeout=10)
        if health_response.status_code == 200:
            health = health_response.json()
            print(f"   ✅ 服务器状态: {health.get('status', 'unknown')}")
            print(f"   📊 Scheduler可用: {health.get('scheduler_available', False)}")
            print(f"   👥 活跃Sessions: {health.get('scheduler_sessions', 0)}")
        else:
            print(f"   ❌ 健康检查失败: {health_response.status_code}")
            return
        
        # 2. 详细诊断
        print("\n2️⃣ 队列诊断...")
        diagnose_response = requests.get(f"{server_url}/diagnose", timeout=10)
        if diagnose_response.status_code == 200:
            data = diagnose_response.json()
            
            # 解析队列统计
            queue_stats = data.get('queue_stats', {})
            detailed_info = queue_stats.get('detailed_queue_info', {})
            
            print(f"   📊 总请求数: {queue_stats.get('total_requests', 0)}")
            print(f"   ✅ 已完成请求: {queue_stats.get('completed_requests', 0)}")
            print(f"   👥 活跃Sessions: {queue_stats.get('active_sessions', 0)}")
            
            print("\n🖥️ 各GPU状态:")
            total_prefill = 0
            total_decode = 0
            
            if detailed_info:
                for gpu_id, info in detailed_info.items():
                    prefill = info.get('prefill_queue_size', 0)
                    decode = info.get('decode_queue_size', 0)
                    total_prefill += prefill
                    total_decode += decode
                    language = info.get('language', 'unknown')
                    
                    # 状态emoji
                    total_queue = prefill + decode
                    if total_queue > 32:
                        status = "🚨"  # 严重积压
                    elif total_queue > 16:
                        status = "⚠️"   # 积压
                    elif total_queue > 0:
                        status = "🟢"  # 正常运行
                    else:
                        status = "💤"  # 空闲
                    
                    print(f"   GPU {gpu_id} ({language}) {status}")
                    print(f"      队列: {prefill}P + {decode}D = {total_queue} 总计")
                    
                    # 估算页面使用
                    estimated_pages = prefill * 20 + decode * 5
                    utilization = min(100, (estimated_pages / 576) * 100)
                    print(f"      估算页面使用: ~{estimated_pages} 页 ({utilization:.1f}%)")
                    
                    # 瓶颈分析
                    if total_queue > 32:
                        print(f"      🚨 瓶颈: 队列严重积压")
                    elif decode > 20:
                        print(f"      ⚠️ 瓶颈: DECODE队列积压 ({decode}个)")
                    elif prefill > 8:
                        print(f"      ⚠️ 瓶颈: PREFILL队列积压 ({prefill}个)")
                    
                print(f"\n📊 总计: {total_prefill} PREFILL + {total_decode} DECODE = {total_prefill + total_decode} requests")
            else:
                print("   ⚠️ 无GPU队列信息")
            
            # 3. 卡住的Sessions
            diagnosis = data.get('scheduler_diagnosis', {})
            stuck_sessions = diagnosis.get('stuck_sessions', [])
            if stuck_sessions:
                print(f"\n🚨 卡住的Sessions ({len(stuck_sessions)}个):")
                for session in stuck_sessions[:5]:  # 显示前5个
                    print(f"   - {session.get('user_id', 'unknown')} ({session.get('language_id', 'unknown')})")
                    print(f"     不活跃: {session.get('inactive_seconds', 0):.1f}s")
                    print(f"     未处理音频: {session.get('unprocessed_samples', 0)} 样本")
            else:
                print(f"\n✅ 无卡住Sessions")
            
            # 4. 系统健康评估
            print(f"\n💡 系统健康评估:")
            total_requests = total_prefill + total_decode
            
            if total_requests == 0:
                print("   💤 系统空闲")
            elif total_requests > 50:
                print("   🚨 系统负载极高，建议暂停新用户")
            elif total_requests > 20:
                print("   ⚠️ 系统负载较高，需要关注")
            elif stuck_sessions:
                print("   ⚠️ 有卡住的sessions，需要处理")
            else:
                print("   ✅ 系统运行正常")
            
        else:
            print(f"   ❌ 诊断接口失败: {diagnose_response.status_code}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 连接错误: {e}")
        print("💡 检查服务器URL和网络连接")
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='快速检查InfiniSST Scheduler状态')
    parser.add_argument('--server', default='https://infinisst.ngrok.app', 
                       help='服务器URL (默认: https://infinisst.ngrok.app)')
    
    args = parser.parse_args()
    quick_check(args.server)

if __name__ == "__main__":
    main() 