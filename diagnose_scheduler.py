#!/usr/bin/env python3
"""
Scheduler队列诊断工具
实时分析InfiniSST scheduler的状态，检测瓶颈和问题
"""

import asyncio
import json
import time
from typing import Dict, Any, List
import requests
import argparse

class SchedulerDiagnostics:
    """Scheduler诊断工具"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.previous_stats = None
        
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """从API获取scheduler统计信息"""
        try:
            response = requests.get(f"{self.server_url}/diagnose", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ API返回错误: {response.status_code}")
                return {}
        except Exception as e:
            print(f"❌ 无法连接到服务器: {e}")
            return {}
    
    def analyze_queue_status(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """分析队列状态"""
        analysis = {
            'total_prefill_requests': 0,
            'total_decode_requests': 0,
            'total_requests': 0,
            'gpu_analysis': {},
            'bottlenecks': [],
            'page_estimates': {}
        }
        
        queue_info = stats.get('queue_stats', {}).get('detailed_queue_info', {})
        
        for gpu_id, gpu_info in queue_info.items():
            gpu_id = str(gpu_id)  # 确保是字符串
            prefill_count = gpu_info.get('prefill_queue_size', 0)
            decode_count = gpu_info.get('decode_queue_size', 0)
            total_count = prefill_count + decode_count
            
            analysis['total_prefill_requests'] += prefill_count
            analysis['total_decode_requests'] += decode_count
            analysis['total_requests'] += total_count
            
            # 每个GPU的分析
            gpu_analysis = {
                'language': gpu_info.get('language', 'unknown'),
                'prefill_queue': prefill_count,
                'decode_queue': decode_count,
                'total_queue': total_count,
                'queue_ratio': f"{prefill_count}P:{decode_count}D",
                'status': 'normal'
            }
            
            # 估算页面使用量
            # 基于经验值：每个prefill request ~20页，每个decode request ~5页
            estimated_prefill_pages = prefill_count * 20
            estimated_decode_pages = decode_count * 5
            total_estimated_pages = estimated_prefill_pages + estimated_decode_pages
            
            analysis['page_estimates'][gpu_id] = {
                'prefill_pages': estimated_prefill_pages,
                'decode_pages': estimated_decode_pages,
                'total_pages': total_estimated_pages,
                'page_utilization_percent': min(100, (total_estimated_pages / 576) * 100)  # 假设576个总页面
            }
            
            # 检测瓶颈
            if total_count > 32:
                analysis['bottlenecks'].append(f"GPU {gpu_id} 队列严重积压: {total_count}个requests")
                gpu_analysis['status'] = 'overloaded'
            elif total_count > 16:
                analysis['bottlenecks'].append(f"GPU {gpu_id} 队列积压: {total_count}个requests")
                gpu_analysis['status'] = 'busy'
            elif decode_count > 20:
                analysis['bottlenecks'].append(f"GPU {gpu_id} DECODE队列积压: {decode_count}个requests")
                gpu_analysis['status'] = 'decode_backlog'
            elif prefill_count > 8:
                analysis['bottlenecks'].append(f"GPU {gpu_id} PREFILL队列积压: {prefill_count}个requests")
                gpu_analysis['status'] = 'prefill_backlog'
            
            # 页面使用率检查
            if total_estimated_pages > 500:
                analysis['bottlenecks'].append(f"GPU {gpu_id} 页面使用率过高: ~{total_estimated_pages}页")
                gpu_analysis['status'] = 'memory_pressure'
            
            analysis['gpu_analysis'][gpu_id] = gpu_analysis
        
        return analysis
    
    def analyze_session_status(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """分析session状态"""
        analysis = {
            'active_sessions': 0,
            'stuck_sessions': [],
            'session_distribution': {},
            'memory_usage': {}
        }
        
        # 获取卡住的session信息
        if 'stuck_sessions' in stats:
            analysis['stuck_sessions'] = stats['stuck_sessions']
        
        # 分析session分布
        queue_stats = stats.get('queue_stats', {})
        if 'inactive_sessions' in queue_stats:
            inactive_sessions = queue_stats['inactive_sessions']
            analysis['active_sessions'] = queue_stats.get('active_sessions', 0)
            
            # 按语言统计session
            for session in inactive_sessions:
                lang = session.get('language_id', 'unknown')
                if lang not in analysis['session_distribution']:
                    analysis['session_distribution'][lang] = 0
                analysis['session_distribution'][lang] += 1
        
        return analysis
    
    def print_diagnosis_report(self, stats: Dict[str, Any]):
        """打印诊断报告"""
        print("=" * 80)
        print("🔍 InfiniSST Scheduler 诊断报告")
        print("=" * 80)
        print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 队列状态分析
        queue_analysis = self.analyze_queue_status(stats)
        print("📊 队列状态:")
        print(f"   总请求数: {queue_analysis['total_requests']}")
        print(f"   PREFILL队列: {queue_analysis['total_prefill_requests']} 个请求")
        print(f"   DECODE队列: {queue_analysis['total_decode_requests']} 个请求")
        print()
        
        # 各GPU详细状态
        print("🖥️  各GPU状态:")
        for gpu_id, gpu_info in queue_analysis['gpu_analysis'].items():
            status_emoji = {
                'normal': '✅',
                'busy': '⚠️',
                'overloaded': '🚨',
                'decode_backlog': '🟡',
                'prefill_backlog': '🟠',
                'memory_pressure': '🔴'
            }.get(gpu_info['status'], '❓')
            
            print(f"   GPU {gpu_id} ({gpu_info['language']}) {status_emoji}")
            print(f"      队列: {gpu_info['queue_ratio']} (总计: {gpu_info['total_queue']})")
            
            # 页面使用估算
            page_info = queue_analysis['page_estimates'][gpu_id]
            print(f"      估算页面使用: {page_info['total_pages']} 页 ({page_info['page_utilization_percent']:.1f}%)")
            print(f"         - PREFILL: {page_info['prefill_pages']} 页")
            print(f"         - DECODE: {page_info['decode_pages']} 页")
            print()
        
        # Session状态分析
        session_analysis = self.analyze_session_status(stats)
        print("👥 Session状态:")
        print(f"   活跃Sessions: {session_analysis['active_sessions']}")
        print(f"   卡住Sessions: {len(session_analysis['stuck_sessions'])}")
        
        if session_analysis['stuck_sessions']:
            print("   卡住的Sessions详情:")
            for session in session_analysis['stuck_sessions']:
                print(f"      - {session.get('user_id', 'unknown')} ({session.get('language_id', 'unknown')})")
                print(f"        不活跃时间: {session.get('inactive_seconds', 0):.1f}s")
                print(f"        未处理音频: {session.get('unprocessed_samples', 0)} 样本")
                print(f"        队列状态: P{session.get('prefill_queue_size', 0)} + D{session.get('decode_queue_size', 0)}")
        print()
        
        # 瓶颈分析
        print("🚨 瓶颈分析:")
        if queue_analysis['bottlenecks']:
            for bottleneck in queue_analysis['bottlenecks']:
                print(f"   - {bottleneck}")
        else:
            print("   ✅ 未检测到明显瓶颈")
        print()
        
        # 推荐操作
        print("💡 推荐操作:")
        self.generate_recommendations(queue_analysis, session_analysis)
        print()
    
    def generate_recommendations(self, queue_analysis: Dict, session_analysis: Dict):
        """生成推荐操作"""
        recommendations = []
        
        # 基于队列状态的推荐
        if queue_analysis['total_requests'] > 50:
            recommendations.append("系统负载过高，建议暂停新用户接入")
        
        if queue_analysis['total_decode_requests'] > 30:
            recommendations.append("DECODE队列积压严重，可能存在decode无限循环问题")
        
        # 基于页面使用的推荐
        for gpu_id, page_info in queue_analysis['page_estimates'].items():
            if page_info['page_utilization_percent'] > 80:
                recommendations.append(f"GPU {gpu_id} 页面使用率过高，建议清理长时间不活跃的sessions")
        
        # 基于session状态的推荐
        if len(session_analysis['stuck_sessions']) > 0:
            recommendations.append("发现卡住的sessions，建议检查前端WebSocket连接或重置这些sessions")
        
        # 基于瓶颈的推荐
        if len(queue_analysis['bottlenecks']) > 3:
            recommendations.append("检测到多个瓶颈，建议重启服务")
        
        if not recommendations:
            recommendations.append("系统运行正常，继续监控")
        
        for rec in recommendations:
            print(f"   - {rec}")
    
    def monitor_continuously(self, interval: int = 10):
        """持续监控模式"""
        print(f"🔄 开始持续监控，每{interval}秒更新一次...")
        print("按 Ctrl+C 停止监控\n")
        
        try:
            while True:
                stats = self.get_scheduler_stats()
                if stats:
                    self.print_diagnosis_report(stats)
                    print(f"⏰ 下次更新: {interval}秒后...")
                    print("=" * 80 + "\n")
                else:
                    print("❌ 无法获取统计信息，检查服务器状态")
                
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n👋 监控已停止")
    
    def single_check(self):
        """单次检查模式"""
        stats = self.get_scheduler_stats()
        if stats:
            self.print_diagnosis_report(stats)
        else:
            print("❌ 无法获取统计信息，检查服务器状态")

def main():
    parser = argparse.ArgumentParser(description='InfiniSST Scheduler 诊断工具')
    parser.add_argument('--server', default='http://localhost:8000', help='服务器地址')
    parser.add_argument('--monitor', action='store_true', help='持续监控模式')
    parser.add_argument('--interval', type=int, default=10, help='监控间隔(秒)')
    
    args = parser.parse_args()
    
    diagnostics = SchedulerDiagnostics(args.server)
    
    if args.monitor:
        diagnostics.monitor_continuously(args.interval)
    else:
        diagnostics.single_check()

if __name__ == "__main__":
    main() 