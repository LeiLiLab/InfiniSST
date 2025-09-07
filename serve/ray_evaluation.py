#!/usr/bin/env python3
"""
Ray-based InfiniSST Evaluation Framework
基于Ray的InfiniSST评估框架
"""

import asyncio
import aiohttp
import numpy as np
import time
import json
import logging
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import websockets
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RayTestConfig:
    """Ray测试配置"""
    server_url: str = "http://localhost:8000"
    websocket_url: str = "ws://localhost:8000"
    
    # Test parameters
    num_users: int = 8
    test_duration: int = 120  # seconds
    arrival_rate: float = 1.0  # users per second
    
    # Language distribution
    language_pairs: List[str] = field(default_factory=lambda: [
        "English -> Chinese",
        "English -> Italian"
    ])
    language_weights: List[float] = field(default_factory=lambda: [0.5, 0.5])
    
    # Audio configuration
    audio_duration: float = 10.0  # seconds
    chunk_size: int = 1600  # samples per chunk (0.1s at 16kHz)
    sample_rate: int = 16000
    
    # Ray-specific configuration
    test_batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])
    test_batch_timeouts: List[float] = field(default_factory=lambda: [50.0, 100.0, 200.0])
    
    # Output
    output_dir: str = "ray_evaluation_results"

@dataclass
class UserSimulation:
    """用户模拟"""
    user_id: str
    language_pair: str
    arrival_time: float
    session_id: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    total_latency: float = 0.0
    response_times: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    chunks_sent: int = 0
    chunks_received: int = 0
    status: str = "pending"  # pending, running, completed, failed

@dataclass
class BatchTestResult:
    """批处理测试结果"""
    batch_size: int
    batch_timeout: float
    total_requests: int
    avg_latency: float
    throughput: float
    success_rate: float
    error_count: int

class RayEvaluationFramework:
    """Ray评估框架"""
    
    def __init__(self, config: RayTestConfig):
        self.config = config
        self.results: List[UserSimulation] = []
        self.batch_results: List[BatchTestResult] = []
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Ray评估框架初始化完成，输出目录: {self.config.output_dir}")
    
    async def run_basic_functionality_test(self) -> bool:
        """基本功能测试"""
        logger.info("🧪 开始基本功能测试...")
        
        try:
            async with aiohttp.ClientSession() as session:
                # 测试健康检查
                async with session.get(f"{self.config.server_url}/health") as response:
                    if response.status != 200:
                        logger.error("健康检查失败")
                        return False
                    
                    health_data = await response.json()
                    logger.info(f"健康检查通过: {health_data.get('status', 'unknown')}")
                
                # 测试Ray统计信息
                async with session.get(f"{self.config.server_url}/ray/stats") as response:
                    if response.status == 200:
                        ray_stats = await response.json()
                        logger.info(f"Ray统计信息: {ray_stats.get('success', False)}")
                    else:
                        logger.warning("Ray统计信息获取失败")
                
                # 测试会话创建
                init_data = {
                    "agent_type": "InfiniSST",
                    "language_pair": "English -> Chinese",
                    "client_id": "test_user_001"
                }
                
                async with session.post(f"{self.config.server_url}/init", json=init_data) as response:
                    if response.status != 200:
                        logger.error("会话创建失败")
                        return False
                    
                    session_data = await response.json()
                    session_id = session_data.get("session_id")
                    if not session_id:
                        logger.error("未获取到有效的会话ID")
                        return False
                    
                    logger.info(f"会话创建成功: {session_id}")
                
                # 测试WebSocket连接
                websocket_url = f"{self.config.websocket_url}/wss/{session_id}"
                try:
                    async with websockets.connect(websocket_url) as websocket:
                        # 等待READY消息
                        ready_msg = await websocket.recv()
                        if "READY" not in ready_msg:
                            logger.error(f"未收到READY消息: {ready_msg}")
                            return False
                        
                        logger.info("WebSocket连接测试成功")
                        
                        # 发送测试音频数据
                        test_audio = np.random.randn(1600).astype(np.float32)
                        await websocket.send(test_audio.tobytes())
                        
                        # 等待响应或超时
                        try:
                            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                            logger.info(f"收到响应: {response[:100]}...")
                        except asyncio.TimeoutError:
                            logger.warning("未在10秒内收到响应，但连接正常")
                        
                        # 发送EOF
                        await websocket.send("EOF")
                        final_response = await websocket.recv()
                        logger.info(f"最终响应: {final_response}")
                        
                except Exception as e:
                    logger.error(f"WebSocket测试失败: {e}")
                    return False
                
                # 清理会话
                await session.post(f"{self.config.server_url}/delete_session", json={"session_id": session_id})
                logger.info("测试会话已清理")
            
            logger.info("✅ 基本功能测试通过")
            return True
            
        except Exception as e:
            logger.error(f"❌ 基本功能测试失败: {e}")
            return False
    
    async def run_batch_performance_test(self) -> List[BatchTestResult]:
        """批处理性能测试"""
        logger.info("🚀 开始批处理性能测试...")
        
        batch_results = []
        
        for batch_size in self.config.test_batch_sizes:
            for batch_timeout in self.config.test_batch_timeouts:
                logger.info(f"测试批处理配置: size={batch_size}, timeout={batch_timeout}ms")
                
                # 配置Ray系统
                async with aiohttp.ClientSession() as session:
                    config_data = {
                        "max_batch_size": batch_size,
                        "batch_timeout_ms": batch_timeout,
                        "enable_dynamic_batching": True
                    }
                    
                    async with session.post(f"{self.config.server_url}/ray/configure", json=config_data) as response:
                        if response.status != 200:
                            logger.warning(f"批处理配置失败: {await response.text()}")
                            continue
                
                # 运行负载测试
                result = await self._run_load_test_for_batch_config(batch_size, batch_timeout)
                if result:
                    batch_results.append(result)
                    logger.info(f"批处理测试完成: {result}")
                
                # 等待系统稳定
                await asyncio.sleep(5)
        
        self.batch_results = batch_results
        return batch_results
    
    async def _run_load_test_for_batch_config(self, batch_size: int, batch_timeout: float) -> Optional[BatchTestResult]:
        """为特定批处理配置运行负载测试"""
        
        # 创建并发用户
        concurrent_users = min(batch_size * 2, 16)  # 不超过16个并发用户
        
        users = []
        for i in range(concurrent_users):
            user = UserSimulation(
                user_id=f"batch_test_user_{i}",
                language_pair=np.random.choice(self.config.language_pairs, p=self.config.language_weights),
                arrival_time=time.time() + np.random.exponential(1.0 / 2.0)  # 高频率到达
            )
            users.append(user)
        
        # 并发运行用户模拟
        start_time = time.time()
        tasks = [self._simulate_user(user) for user in users]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # 统计结果
            successful_users = []
            error_count = 0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    users[i].status = "failed"
                    users[i].errors.append(str(result))
                    error_count += 1
                else:
                    if users[i].status == "completed":
                        successful_users.append(users[i])
            
            if not successful_users:
                logger.warning(f"批处理测试无成功用户: batch_size={batch_size}")
                return None
            
            # 计算指标
            total_requests = sum(user.chunks_sent for user in users)
            avg_latency = np.mean([np.mean(user.response_times) for user in successful_users if user.response_times])
            throughput = total_requests / (end_time - start_time)
            success_rate = len(successful_users) / len(users)
            
            return BatchTestResult(
                batch_size=batch_size,
                batch_timeout=batch_timeout,
                total_requests=total_requests,
                avg_latency=avg_latency,
                throughput=throughput,
                success_rate=success_rate,
                error_count=error_count
            )
            
        except Exception as e:
            logger.error(f"负载测试异常: {e}")
            return None
    
    async def run_concurrent_users_test(self) -> List[UserSimulation]:
        """并发用户测试"""
        logger.info(f"👥 开始并发用户测试: {self.config.num_users} 用户")
        
        # 生成用户模拟
        users = []
        for i in range(self.config.num_users):
            user = UserSimulation(
                user_id=f"concurrent_user_{i:03d}",
                language_pair=np.random.choice(self.config.language_pairs, p=self.config.language_weights),
                arrival_time=time.time() + np.random.exponential(1.0 / self.config.arrival_rate)
            )
            users.append(user)
        
        # 按到达时间排序
        users.sort(key=lambda u: u.arrival_time)
        
        # 并发运行用户模拟
        tasks = []
        for user in users:
            # 延迟启动用户
            delay = max(0, user.arrival_time - time.time())
            task = asyncio.create_task(self._delayed_user_start(user, delay))
            tasks.append(task)
        
        # 等待所有用户完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                users[i].status = "failed"
                users[i].errors.append(str(result))
                logger.error(f"用户 {users[i].user_id} 失败: {result}")
        
        self.results = users
        return users
    
    async def _delayed_user_start(self, user: UserSimulation, delay: float) -> UserSimulation:
        """延迟启动用户"""
        if delay > 0:
            await asyncio.sleep(delay)
        
        return await self._simulate_user(user)
    
    async def _simulate_user(self, user: UserSimulation) -> UserSimulation:
        """模拟单个用户"""
        user.start_time = time.time()
        user.status = "running"
        
        try:
            # 创建会话
            session_id = await self._create_session(user)
            if not session_id:
                user.status = "failed"
                user.errors.append("会话创建失败")
                return user
            
            user.session_id = session_id
            
            # WebSocket通信
            await self._websocket_communication(user)
            
            # 清理会话
            await self._cleanup_session(user)
            
            user.end_time = time.time()
            user.status = "completed"
            
        except Exception as e:
            user.status = "failed"
            user.errors.append(str(e))
            user.end_time = time.time()
            logger.error(f"用户模拟失败 {user.user_id}: {e}")
        
        return user
    
    async def _create_session(self, user: UserSimulation) -> Optional[str]:
        """创建会话"""
        async with aiohttp.ClientSession() as session:
            init_data = {
                "agent_type": "InfiniSST",
                "language_pair": user.language_pair,
                "client_id": user.user_id,
                "evaluation_mode": "true"
            }
            
            try:
                async with session.post(f"{self.config.server_url}/init", json=init_data) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("session_id")
                    else:
                        logger.error(f"会话创建失败 {user.user_id}: {response.status}")
                        return None
            except Exception as e:
                logger.error(f"会话创建异常 {user.user_id}: {e}")
                return None
    
    async def _websocket_communication(self, user: UserSimulation):
        """WebSocket通信"""
        websocket_url = f"{self.config.websocket_url}/wss/{user.session_id}"
        
        async with websockets.connect(websocket_url) as websocket:
            # 等待READY消息
            ready_msg = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            if "READY" not in ready_msg:
                raise Exception(f"未收到READY消息: {ready_msg}")
            
            # 生成并发送音频数据
            total_samples = int(self.config.audio_duration * self.config.sample_rate)
            
            for chunk_start in range(0, total_samples, self.config.chunk_size):
                chunk_end = min(chunk_start + self.config.chunk_size, total_samples)
                chunk = np.random.randn(chunk_end - chunk_start).astype(np.float32)
                
                send_time = time.time()
                await websocket.send(chunk.tobytes())
                user.chunks_sent += 1
                
                # 尝试接收响应（非阻塞）
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    receive_time = time.time()
                    user.response_times.append(receive_time - send_time)
                    user.chunks_received += 1
                except asyncio.TimeoutError:
                    pass  # 没有响应，继续发送下一块
                
                # 模拟实时音频间隔
                await asyncio.sleep(self.config.chunk_size / self.config.sample_rate)
            
            # 发送EOF并等待最终响应
            await websocket.send("EOF")
            try:
                final_response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                user.chunks_received += 1
            except asyncio.TimeoutError:
                logger.warning(f"用户 {user.user_id} 未收到最终响应")
    
    async def _cleanup_session(self, user: UserSimulation):
        """清理会话"""
        if not user.session_id:
            return
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(f"{self.config.server_url}/delete_session", 
                                      json={"session_id": user.session_id}) as response:
                    if response.status != 200:
                        logger.warning(f"会话清理失败 {user.user_id}: {response.status}")
            except Exception as e:
                logger.warning(f"会话清理异常 {user.user_id}: {e}")
    
    def generate_report(self):
        """生成评估报告"""
        logger.info("📊 生成评估报告...")
        
        timestamp = int(time.time())
        report_dir = Path(self.config.output_dir) / f"ray_evaluation_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成用户测试报告
        if self.results:
            self._generate_user_test_report(report_dir)
        
        # 生成批处理测试报告
        if self.batch_results:
            self._generate_batch_test_report(report_dir)
        
        # 生成总结报告
        self._generate_summary_report(report_dir)
        
        logger.info(f"评估报告已生成: {report_dir}")
    
    def _generate_user_test_report(self, report_dir: Path):
        """生成用户测试报告"""
        successful_users = [u for u in self.results if u.status == "completed"]
        failed_users = [u for u in self.results if u.status == "failed"]
        
        # 统计数据
        stats = {
            "total_users": len(self.results),
            "successful_users": len(successful_users),
            "failed_users": len(failed_users),
            "success_rate": len(successful_users) / len(self.results) if self.results else 0,
        }
        
        if successful_users:
            response_times = []
            for user in successful_users:
                response_times.extend(user.response_times)
            
            if response_times:
                stats.update({
                    "avg_response_time": np.mean(response_times),
                    "median_response_time": np.median(response_times),
                    "p95_response_time": np.percentile(response_times, 95),
                    "p99_response_time": np.percentile(response_times, 99),
                })
        
        # 保存统计数据
        with open(report_dir / "user_test_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        # 生成图表
        if successful_users and any(u.response_times for u in successful_users):
            plt.figure(figsize=(12, 8))
            
            # 响应时间分布
            plt.subplot(2, 2, 1)
            all_response_times = []
            for user in successful_users:
                all_response_times.extend(user.response_times)
            
            plt.hist(all_response_times, bins=50, alpha=0.7)
            plt.xlabel("Response Time (s)")
            plt.ylabel("Frequency")
            plt.title("Response Time Distribution")
            
            # 用户成功率
            plt.subplot(2, 2, 2)
            labels = ["Successful", "Failed"]
            sizes = [len(successful_users), len(failed_users)]
            colors = ["green", "red"]
            plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%")
            plt.title("User Success Rate")
            
            # 语言对分布
            plt.subplot(2, 2, 3)
            language_counts = {}
            for user in self.results:
                lang = user.language_pair
                language_counts[lang] = language_counts.get(lang, 0) + 1
            
            plt.bar(language_counts.keys(), language_counts.values())
            plt.xlabel("Language Pair")
            plt.ylabel("Number of Users")
            plt.title("Language Pair Distribution")
            plt.xticks(rotation=45)
            
            # 每个用户的响应时间
            plt.subplot(2, 2, 4)
            user_avg_times = [np.mean(u.response_times) if u.response_times else 0 for u in successful_users]
            plt.scatter(range(len(user_avg_times)), user_avg_times, alpha=0.6)
            plt.xlabel("User Index")
            plt.ylabel("Average Response Time (s)")
            plt.title("Response Time per User")
            
            plt.tight_layout()
            plt.savefig(report_dir / "user_test_charts.png", dpi=300, bbox_inches="tight")
            plt.close()
    
    def _generate_batch_test_report(self, report_dir: Path):
        """生成批处理测试报告"""
        if not self.batch_results:
            return
        
        # 转换为DataFrame以便分析
        df_data = []
        for result in self.batch_results:
            df_data.append({
                "batch_size": result.batch_size,
                "batch_timeout": result.batch_timeout,
                "total_requests": result.total_requests,
                "avg_latency": result.avg_latency,
                "throughput": result.throughput,
                "success_rate": result.success_rate,
                "error_count": result.error_count
            })
        
        df = pd.DataFrame(df_data)
        
        # 保存数据
        df.to_csv(report_dir / "batch_test_results.csv", index=False)
        
        # 生成图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 吞吐量 vs 批处理大小
        pivot_throughput = df.pivot(index="batch_size", columns="batch_timeout", values="throughput")
        sns.heatmap(pivot_throughput, annot=True, fmt=".2f", ax=axes[0, 0])
        axes[0, 0].set_title("Throughput (requests/s)")
        axes[0, 0].set_ylabel("Batch Size")
        
        # 延迟 vs 批处理大小
        pivot_latency = df.pivot(index="batch_size", columns="batch_timeout", values="avg_latency")
        sns.heatmap(pivot_latency, annot=True, fmt=".3f", ax=axes[0, 1])
        axes[0, 1].set_title("Average Latency (s)")
        axes[0, 1].set_ylabel("Batch Size")
        
        # 成功率 vs 批处理大小
        pivot_success = df.pivot(index="batch_size", columns="batch_timeout", values="success_rate")
        sns.heatmap(pivot_success, annot=True, fmt=".2f", ax=axes[1, 0])
        axes[1, 0].set_title("Success Rate")
        axes[1, 0].set_ylabel("Batch Size")
        
        # 错误数量 vs 批处理大小
        pivot_errors = df.pivot(index="batch_size", columns="batch_timeout", values="error_count")
        sns.heatmap(pivot_errors, annot=True, fmt="d", ax=axes[1, 1])
        axes[1, 1].set_title("Error Count")
        axes[1, 1].set_ylabel("Batch Size")
        
        plt.tight_layout()
        plt.savefig(report_dir / "batch_test_heatmaps.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # 找到最优配置
        best_config = df.loc[df["throughput"].idxmax()]
        best_config_info = {
            "best_batch_size": int(best_config["batch_size"]),
            "best_batch_timeout": float(best_config["batch_timeout"]),
            "best_throughput": float(best_config["throughput"]),
            "best_avg_latency": float(best_config["avg_latency"]),
            "best_success_rate": float(best_config["success_rate"])
        }
        
        with open(report_dir / "best_batch_config.json", "w") as f:
            json.dump(best_config_info, f, indent=2)
    
    def _generate_summary_report(self, report_dir: Path):
        """生成总结报告"""
        summary = {
            "test_config": {
                "num_users": self.config.num_users,
                "test_duration": self.config.test_duration,
                "language_pairs": self.config.language_pairs,
                "batch_sizes_tested": self.config.test_batch_sizes,
                "batch_timeouts_tested": self.config.test_batch_timeouts
            },
            "test_timestamp": time.time(),
            "results_summary": {}
        }
        
        # 用户测试总结
        if self.results:
            successful_users = [u for u in self.results if u.status == "completed"]
            summary["results_summary"]["user_test"] = {
                "total_users": len(self.results),
                "successful_users": len(successful_users),
                "success_rate": len(successful_users) / len(self.results)
            }
        
        # 批处理测试总结
        if self.batch_results:
            best_throughput = max(r.throughput for r in self.batch_results)
            best_config = next(r for r in self.batch_results if r.throughput == best_throughput)
            
            summary["results_summary"]["batch_test"] = {
                "configurations_tested": len(self.batch_results),
                "best_throughput": best_throughput,
                "best_batch_size": best_config.batch_size,
                "best_batch_timeout": best_config.batch_timeout
            }
        
        with open(report_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # 生成Markdown报告
        self._generate_markdown_report(report_dir, summary)
    
    def _generate_markdown_report(self, report_dir: Path, summary: Dict[str, Any]):
        """生成Markdown格式的报告"""
        markdown_content = f"""# Ray-based InfiniSST Evaluation Report

## Test Configuration
- **Number of Users**: {summary['test_config']['num_users']}
- **Test Duration**: {summary['test_config']['test_duration']} seconds
- **Language Pairs**: {', '.join(summary['test_config']['language_pairs'])}
- **Batch Sizes Tested**: {summary['test_config']['batch_sizes_tested']}
- **Batch Timeouts Tested**: {summary['test_config']['batch_timeouts_tested']} ms

## Results Summary

### User Test Results
"""
        
        if "user_test" in summary["results_summary"]:
            user_results = summary["results_summary"]["user_test"]
            markdown_content += f"""
- **Total Users**: {user_results['total_users']}
- **Successful Users**: {user_results['successful_users']}
- **Success Rate**: {user_results['success_rate']:.2%}
"""
        
        if "batch_test" in summary["results_summary"]:
            batch_results = summary["results_summary"]["batch_test"]
            markdown_content += f"""
### Batch Test Results
- **Configurations Tested**: {batch_results['configurations_tested']}
- **Best Throughput**: {batch_results['best_throughput']:.2f} requests/s
- **Optimal Batch Size**: {batch_results['best_batch_size']}
- **Optimal Batch Timeout**: {batch_results['best_batch_timeout']} ms
"""
        
        markdown_content += f"""
## Files Generated
- `user_test_stats.json` - Detailed user test statistics
- `user_test_charts.png` - User test visualization
- `batch_test_results.csv` - Batch test raw data
- `batch_test_heatmaps.png` - Batch test performance heatmaps
- `best_batch_config.json` - Optimal batch configuration
- `summary.json` - Complete test summary

## Test Timestamp
{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(summary['test_timestamp']))}
"""
        
        with open(report_dir / "README.md", "w") as f:
            f.write(markdown_content)

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Ray InfiniSST Evaluation Framework")
    parser.add_argument("--server-url", default="http://localhost:8000", help="API server URL")
    parser.add_argument("--num-users", type=int, default=8, help="Number of concurrent users")
    parser.add_argument("--test-duration", type=int, default=120, help="Test duration in seconds")
    parser.add_argument("--output-dir", default="ray_evaluation_results", help="Output directory")
    parser.add_argument("--basic-test-only", action="store_true", help="Run only basic functionality test")
    parser.add_argument("--batch-test-only", action="store_true", help="Run only batch performance test")
    parser.add_argument("--skip-basic-test", action="store_true", help="Skip basic functionality test")
    
    args = parser.parse_args()
    
    # 创建配置
    config = RayTestConfig(
        server_url=args.server_url,
        websocket_url=args.server_url.replace("http://", "ws://").replace("https://", "wss://"),
        num_users=args.num_users,
        test_duration=args.test_duration,
        output_dir=args.output_dir
    )
    
    # 创建评估框架
    framework = RayEvaluationFramework(config)
    
    logger.info(f"🚀 开始Ray InfiniSST评估测试")
    logger.info(f"服务器URL: {config.server_url}")
    logger.info(f"并发用户数: {config.num_users}")
    logger.info(f"测试时长: {config.test_duration}秒")
    
    try:
        # 基本功能测试
        if not args.skip_basic_test and not args.batch_test_only:
            basic_test_passed = await framework.run_basic_functionality_test()
            if not basic_test_passed:
                logger.error("基本功能测试失败，终止评估")
                return
        
        if args.basic_test_only:
            logger.info("仅运行基本功能测试，完成")
            return
        
        # 批处理性能测试
        if not args.basic_test_only:
            await framework.run_batch_performance_test()
        
        # 并发用户测试
        if not args.batch_test_only and not args.basic_test_only:
            await framework.run_concurrent_users_test()
        
        # 生成报告
        framework.generate_report()
        
        logger.info("✅ Ray InfiniSST评估测试完成")
        
    except KeyboardInterrupt:
        logger.info("用户中断测试")
    except Exception as e:
        logger.error(f"评估测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 