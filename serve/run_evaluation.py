#!/usr/bin/env python3
"""
InfiniSST Evaluation Runner
运行并发用户评估测试的示例脚本
"""

import asyncio
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

from evaluation_framework import EvaluationFramework, TestConfig

async def run_quick_test():
    """运行快速测试：8个用户，30秒"""
    config = TestConfig(
        num_users=8,
        language_split=0.5,  # 50% Chinese, 50% Italian
        arrival_rate=1.0,    # 1 user per second
        test_duration=30,    # 30 seconds
        server_url="http://localhost:8000",
        output_dir="evaluation_results/quick_test",
        use_dynamic_schedule=False,
        max_batch_size=16,
        batch_timeout=0.1
    )
    
    framework = EvaluationFramework(config)
    results = await framework.run_evaluation()
    
    print(f"\n🎉 Quick Test Results:")
    print(f"   - Completed users: {results.completed_users}")
    print(f"   - Failed users: {results.failed_users}")
    print(f"   - Average streamLAAL: {results.avg_stream_laal:.3f}s")
    print(f"   - Test duration: {results.total_duration:.1f}s")
    
    return results

async def run_moderate_test():
    """运行中等规模测试：16个用户，2分钟"""
    config = TestConfig(
        num_users=16,
        language_split=0.5,
        arrival_rate=2.0,    # 2 users per second
        test_duration=120,   # 2 minutes
        server_url="http://localhost:8000",
        output_dir="evaluation_results/moderate_test",
        use_dynamic_schedule=False,
        max_batch_size=32,
        batch_timeout=0.1
    )
    
    framework = EvaluationFramework(config)
    results = await framework.run_evaluation()
    
    print(f"\n🎉 Moderate Test Results:")
    print(f"   - Completed users: {results.completed_users}")
    print(f"   - Failed users: {results.failed_users}")
    print(f"   - Average streamLAAL: {results.avg_stream_laal:.3f}s")
    print(f"   - Test duration: {results.total_duration:.1f}s")
    
    return results

async def run_large_scale_test():
    """运行大规模测试：32个用户，5分钟"""
    config = TestConfig(
        num_users=32,
        language_split=0.5,
        arrival_rate=2.0,    # 2 users per second
        test_duration=300,   # 5 minutes
        server_url="http://localhost:8000",
        output_dir="evaluation_results/large_scale_test",
        use_dynamic_schedule=False,
        max_batch_size=32,
        batch_timeout=0.1
    )
    
    framework = EvaluationFramework(config)
    results = await framework.run_evaluation()
    
    print(f"\n🎉 Large Scale Test Results:")
    print(f"   - Completed users: {results.completed_users}")
    print(f"   - Failed users: {results.failed_users}")
    print(f"   - Average streamLAAL: {results.avg_stream_laal:.3f}s")
    print(f"   - Test duration: {results.total_duration:.1f}s")
    
    return results

async def run_dynamic_schedule_comparison():
    """运行动态调度对比测试"""
    print("🚀 Running Dynamic Scheduling Comparison Test...")
    
    # 测试静态调度
    print("\n1️⃣ Testing Static Scheduling...")
    static_config = TestConfig(
        num_users=16,
        language_split=0.5,
        arrival_rate=2.0,
        test_duration=120,
        server_url="http://localhost:8000",
        output_dir="evaluation_results/static_schedule",
        use_dynamic_schedule=False,
        max_batch_size=32,
        batch_timeout=0.1
    )
    
    static_framework = EvaluationFramework(static_config)
    static_results = await static_framework.run_evaluation()
    
    # 测试动态调度
    print("\n2️⃣ Testing Dynamic Scheduling...")
    dynamic_config = TestConfig(
        num_users=16,
        language_split=0.5,
        arrival_rate=2.0,
        test_duration=120,
        server_url="http://localhost:8000",
        output_dir="evaluation_results/dynamic_schedule",
        use_dynamic_schedule=True,
        max_batch_size=32,
        batch_timeout=0.05  # 更短的batch timeout
    )
    
    dynamic_framework = EvaluationFramework(dynamic_config)
    dynamic_results = await dynamic_framework.run_evaluation()
    
    # 比较结果
    print(f"\n📊 Comparison Results:")
    print(f"   Static Scheduling:")
    print(f"     - Average streamLAAL: {static_results.avg_stream_laal:.3f}s")
    print(f"     - Completed users: {static_results.completed_users}")
    print(f"     - Failed users: {static_results.failed_users}")
    print(f"   Dynamic Scheduling:")
    print(f"     - Average streamLAAL: {dynamic_results.avg_stream_laal:.3f}s")
    print(f"     - Completed users: {dynamic_results.completed_users}")
    print(f"     - Failed users: {dynamic_results.failed_users}")
    
    improvement = (static_results.avg_stream_laal - dynamic_results.avg_stream_laal) / static_results.avg_stream_laal * 100
    print(f"   Improvement: {improvement:+.1f}% (negative means dynamic is slower)")
    
    return static_results, dynamic_results

async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="InfiniSST Evaluation Runner")
    parser.add_argument("--test", choices=["quick", "moderate", "large", "dynamic_comparison"], 
                       default="quick", help="Type of test to run")
    parser.add_argument("--server-url", default="http://localhost:8000", 
                       help="Server URL")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting InfiniSST Evaluation Test: {args.test}")
    print(f"📡 Server URL: {args.server_url}")
    
    try:
        if args.test == "quick":
            results = await run_quick_test()
        elif args.test == "moderate":
            results = await run_moderate_test()
        elif args.test == "large":
            results = await run_large_scale_test()
        elif args.test == "dynamic_comparison":
            results = await run_dynamic_schedule_comparison()
        
        print(f"\n✅ Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 