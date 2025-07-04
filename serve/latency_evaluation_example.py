#!/usr/bin/env python3
"""
Latency Evaluation Example
展示如何使用不同latency配置进行评估测试的示例
"""

import asyncio
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

from evaluation_framework import EvaluationFramework, TestConfig

async def example_uniform_latency_test():
    """示例1: 均匀分布的latency测试"""
    print("🔥 示例1: 均匀分布latency测试 (1x-4x)")
    
    config = TestConfig(
        num_users=12,
        language_split=0.5,
        arrival_rate=1.5,
        test_duration=90,
        server_url="http://localhost:8000",
        output_dir="evaluation_results/uniform_latency",
        latency_range=[1, 2, 3, 4],        # 🔥 测试所有latency
        latency_distribution=None          # 🔥 均匀分布
    )
    
    framework = EvaluationFramework(config)
    results = await framework.run_evaluation()
    
    print(f"\n📊 均匀latency测试结果:")
    print(f"   - 完成用户: {results.completed_users}")
    print(f"   - 平均streamLAAL: {results.avg_stream_laal:.3f}s")
    
    # 显示按latency分组的结果
    for latency, stats in sorted(results.latency_results.items()):
        print(f"   - Latency {latency}x: {stats['count']} 用户, avg = {stats['avg_stream_laal']:.3f}s")
    
    return results

async def example_weighted_latency_test():
    """示例2: 加权分布的latency测试"""
    print("\n🔥 示例2: 加权分布latency测试 (偏向2x和3x)")
    
    config = TestConfig(
        num_users=16,
        language_split=0.5,
        arrival_rate=2.0,
        test_duration=120,
        server_url="http://localhost:8000",
        output_dir="evaluation_results/weighted_latency",
        latency_range=[1, 2, 3, 4],        # 🔥 测试所有latency
        latency_distribution=[0.1, 0.4, 0.4, 0.1]  # 🔥 偏向中等latency
    )
    
    framework = EvaluationFramework(config)
    results = await framework.run_evaluation()
    
    print(f"\n📊 加权latency测试结果:")
    print(f"   - 完成用户: {results.completed_users}")
    print(f"   - 平均streamLAAL: {results.avg_stream_laal:.3f}s")
    
    # 显示按latency分组的结果
    for latency, stats in sorted(results.latency_results.items()):
        print(f"   - Latency {latency}x: {stats['count']} 用户, avg = {stats['avg_stream_laal']:.3f}s")
    
    return results

async def example_single_latency_test():
    """示例3: 单一latency测试"""
    print("\n🔥 示例3: 单一latency测试 (仅3x)")
    
    config = TestConfig(
        num_users=8,
        language_split=0.5,
        arrival_rate=1.0,
        test_duration=60,
        server_url="http://localhost:8000",
        output_dir="evaluation_results/single_latency_3x",
        latency_range=[3],                 # 🔥 仅测试3x latency
        latency_distribution=None
    )
    
    framework = EvaluationFramework(config)
    results = await framework.run_evaluation()
    
    print(f"\n📊 单一latency (3x) 测试结果:")
    print(f"   - 完成用户: {results.completed_users}")
    print(f"   - 平均streamLAAL: {results.avg_stream_laal:.3f}s")
    print(f"   - 标准差: {results.std_stream_laal:.3f}s")
    
    return results

async def example_extreme_latency_test():
    """示例4: 极端latency对比测试"""
    print("\n🔥 示例4: 极端latency对比测试 (1x vs 4x)")
    
    # 测试最低latency (1x)
    low_config = TestConfig(
        num_users=6,
        language_split=0.5,
        arrival_rate=1.0,
        test_duration=60,
        server_url="http://localhost:8000",
        output_dir="evaluation_results/extreme_low_latency",
        latency_range=[1],                 # 🔥 仅1x latency
        latency_distribution=None
    )
    
    low_framework = EvaluationFramework(low_config)
    low_results = await low_framework.run_evaluation()
    
    # 测试最高latency (4x)
    high_config = TestConfig(
        num_users=6,
        language_split=0.5,
        arrival_rate=1.0,
        test_duration=60,
        server_url="http://localhost:8000",
        output_dir="evaluation_results/extreme_high_latency",
        latency_range=[4],                 # 🔥 仅4x latency
        latency_distribution=None
    )
    
    high_framework = EvaluationFramework(high_config)
    high_results = await high_framework.run_evaluation()
    
    # 对比结果
    print(f"\n📊 极端latency对比结果:")
    print(f"   1x Latency: {low_results.avg_stream_laal:.3f}s ({low_results.completed_users} 用户)")
    print(f"   4x Latency: {high_results.avg_stream_laal:.3f}s ({high_results.completed_users} 用户)")
    
    if low_results.avg_stream_laal > 0 and high_results.avg_stream_laal > 0:
        ratio = high_results.avg_stream_laal / low_results.avg_stream_laal
        print(f"   4x/1x 比例: {ratio:.2f}")
        
        if ratio > 1.5:
            print(f"   📈 高latency显著增加了streamLAAL")
        elif ratio < 0.8:
            print(f"   📉 高latency实际降低了streamLAAL (可能因为减少了等待)")
        else:
            print(f"   ⚖️  latency对streamLAAL影响较小")
    
    return low_results, high_results

async def example_mixed_language_latency_test():
    """示例5: 混合语言+latency测试"""
    print("\n🔥 示例5: 混合语言+latency组合测试")
    
    config = TestConfig(
        num_users=20,
        language_split=0.5,               # 50% Chinese, 50% Italian
        arrival_rate=2.0,
        test_duration=150,
        server_url="http://localhost:8000",
        output_dir="evaluation_results/mixed_lang_latency",
        latency_range=[1, 2, 3, 4],       # 🔥 所有latency
        latency_distribution=[0.25, 0.25, 0.25, 0.25]  # 🔥 完全均匀分布
    )
    
    framework = EvaluationFramework(config)
    results = await framework.run_evaluation()
    
    print(f"\n📊 混合语言+latency测试结果:")
    print(f"   - 完成用户: {results.completed_users}")
    print(f"   - 平均streamLAAL: {results.avg_stream_laal:.3f}s")
    
    # 按语言显示结果
    if results.chinese_results:
        print(f"   - 中文翻译: {results.chinese_results['count']} 用户, avg = {results.chinese_results['avg_stream_laal']:.3f}s")
    if results.italian_results:
        print(f"   - 意大利语翻译: {results.italian_results['count']} 用户, avg = {results.italian_results['avg_stream_laal']:.3f}s")
    
    # 按latency显示结果
    print(f"   - 按Latency分组:")
    for latency, stats in sorted(results.latency_results.items()):
        print(f"     Latency {latency}x: {stats['count']} 用户, avg = {stats['avg_stream_laal']:.3f}s")
    
    return results

async def main():
    """运行所有latency评估示例"""
    print("🚀 开始Latency评估示例测试")
    print("=" * 80)
    
    try:
        # 运行所有示例
        await example_uniform_latency_test()
        await asyncio.sleep(2)  # 短暂休息
        
        await example_weighted_latency_test()
        await asyncio.sleep(2)
        
        await example_single_latency_test()
        await asyncio.sleep(2)
        
        await example_extreme_latency_test()
        await asyncio.sleep(2)
        
        await example_mixed_language_latency_test()
        
        print("\n" + "=" * 80)
        print("🎉 所有Latency评估示例完成!")
        print("\n📁 结果保存在以下目录:")
        print("   - evaluation_results/uniform_latency/")
        print("   - evaluation_results/weighted_latency/")
        print("   - evaluation_results/single_latency_3x/")
        print("   - evaluation_results/extreme_low_latency/")
        print("   - evaluation_results/extreme_high_latency/")
        print("   - evaluation_results/mixed_lang_latency/")
        
        print("\n💡 使用技巧:")
        print("   1. 调整 latency_range 来测试不同的latency组合")
        print("   2. 使用 latency_distribution 来控制latency分布权重")
        print("   3. 结合语言和latency分析可以发现更深层的性能模式")
        print("   4. 对比极端值可以评估latency策略的效果")
        
    except Exception as e:
        print(f"\n❌ 示例测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 