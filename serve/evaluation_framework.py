#!/usr/bin/env python3
"""
InfiniSST Evaluation Framework
模拟并发用户，执行streamLAAL测试，支持动态调度
"""

import asyncio
import json
import logging
import os
import random
import statistics
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import argparse
import numpy as np
import urllib.parse

import aiohttp
import aiofiles
import soundfile as sf

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestConfig:
    """测试配置"""
    num_users: int = 16
    language_split: float = 0.5  # 50% Chinese, 50% Italian
    server_url: str = "http://localhost:8000"
    test_videos: List[str] = field(default_factory=lambda: [
        "2022.acl-long.110.wav",
        "2022.acl-long.117.wav", 
        "2022.acl-long.268.wav",
        "2022.acl-long.367.wav",
        "2022.acl-long.590.wav"
    ])
    arrival_rate: float = 2.0  # Poisson rate parameter (users per second)
    test_duration: int = 300  # 5 minutes
    output_dir: str = "evaluation_results"
    use_dynamic_schedule: bool = False
    max_batch_size: int = 32
    batch_timeout: float = 0.1
    # 评估专用配置
    session_timeout_extension: bool = True  # 是否延长session超时时间
    ping_interval: int = 60  # ping间隔（秒）
    
@dataclass 
class UserSimulation:
    """单个用户模拟"""
    user_id: str
    language_pair: str
    video_file: str
    arrival_time: float
    session_id: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    stream_laal: Optional[float] = None
    total_characters: int = 0
    total_segments: int = 0
    delays: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed

@dataclass
class EvaluationResults:
    """评估结果"""
    config: TestConfig
    users: List[UserSimulation]
    start_time: float
    end_time: float
    total_duration: float
    
    # 汇总统计
    completed_users: int = 0
    failed_users: int = 0
    avg_stream_laal: float = 0.0
    median_stream_laal: float = 0.0
    std_stream_laal: float = 0.0
    min_stream_laal: float = 0.0
    max_stream_laal: float = 0.0
    
    # 按语言的统计
    chinese_results: Dict[str, float] = field(default_factory=dict)
    italian_results: Dict[str, float] = field(default_factory=dict)
    
    # 系统性能指标
    server_stats: Dict[str, Any] = field(default_factory=dict)

class SimulatedUser:
    """模拟用户类"""
    
    def __init__(self, simulation: UserSimulation, config: TestConfig):
        self.simulation = simulation
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def run(self) -> UserSimulation:
        """运行用户模拟"""
        try:
            self.simulation.status = "running"
            self.simulation.start_time = time.time()
            
            # 创建HTTP会话
            async with aiohttp.ClientSession() as session:
                self.session = session
                
                # 步骤1：加载模型
                await self._load_model()
                                        
                # 步骤2：获取测试视频
                video_path = await self._get_test_video()
                
                # 步骤3：处理音频并收集延迟数据
                await self._process_audio(video_path)
                
                # 步骤4：计算最终指标
                await self._calculate_metrics()
                
                self.simulation.status = "completed"
                self.simulation.end_time = time.time()
                
        except Exception as e:
            self.simulation.status = "failed"
            self.simulation.errors.append(str(e))
            self.simulation.end_time = time.time()
            logger.error(f"User {self.simulation.user_id} failed: {e}")
            
        return self.simulation
    
    async def _load_model(self):
        """加载翻译模型"""
        logger.info(f"🤖 User {self.simulation.user_id}: Loading model for {self.simulation.language_pair}")
        
        payload = {
            "agent_type": "InfiniSST",
            "language_pair": self.simulation.language_pair,
            "latency_multiplier": 2,
            "client_id": self.simulation.user_id,
            "evaluation_mode": "true"  # 🔥 使用字符串而不是布尔值
        }
        
        async with self.session.post(f"{self.config.server_url}/init", params=payload) as response:
            if response.status != 200:
                raise Exception(f"Failed to load model: {response.status}")
            
            data = await response.json()
            self.simulation.session_id = data["session_id"]
            
            # 如果在队列中，等待模型加载
            if data.get("queued", False):
                await self._wait_for_model_ready()
                
        logger.info(f"✅ User {self.simulation.user_id}: Model loaded, session {self.simulation.session_id}")
    
    async def _wait_for_model_ready(self):
        """等待模型准备就绪"""
        max_wait = 180  # 3分钟
        start_wait = time.time()
        
        while time.time() - start_wait < max_wait:
            async with self.session.get(f"{self.config.server_url}/queue_status/{self.simulation.session_id}") as response:
                if response.status == 200:
                    data = await response.json()
                    if data["status"] == "active":
                        return
                    elif data["status"] == "not_found":
                        raise Exception("Session not found in queue")
                        
            await asyncio.sleep(2)
            
        raise Exception(f"Model loading timeout after {max_wait}s")
    
    async def _get_test_video(self) -> str:
        """获取测试视频文件"""
        # 尝试从服务器获取测试视频
        # try:
        #     async with self.session.get(f"{self.config.server_url}/test_video/{self.simulation.video_file}") as response:
        #         if response.status == 200:
        #             # 保存到临时文件
        #             temp_path = f"/tmp/{self.simulation.user_id}_{self.simulation.video_file}"
                    
        #             async with aiofiles.open(temp_path, 'wb') as f:
        #                 async for chunk in response.content.iter_chunked(8192):
        #                     await f.write(chunk)
                    
        #             logger.info(f"📹 User {self.simulation.user_id}: Downloaded test video to {temp_path}")
        #             return temp_path
        # except Exception as e:
        #     logger.warning(f"Failed to download test video from server: {e}")
        
        # 回退：尝试本地文件
        local_paths = [
            f"static/test_video/{self.simulation.video_file}",
            f"~/Downloads/{self.simulation.video_file}",
            f"/tmp/{self.simulation.video_file}"
        ]
        
        for path in local_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                logger.info(f"📹 User {self.simulation.user_id}: Using local video {expanded_path}")
                return expanded_path
                
        raise Exception(f"Test video {self.simulation.video_file} not found")
    
    async def _process_audio(self, video_path: str):
        """处理音频并收集延迟数据 - 参照index.html的音频处理方式"""
        # 提取音频
        audio_data, original_sample_rate = sf.read(video_path)
        
        # 转换为单声道
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        logger.info(f"🎵 User {self.simulation.user_id}: Processing {len(audio_data)/original_sample_rate:.1f}s of audio")
        logger.info(f"   - Original sample rate: {original_sample_rate}Hz")
        
        # 创建ping任务以保持session活跃
        ping_task = asyncio.create_task(self._ping_session_periodically())
        
        try:
            # 建立WebSocket连接并流式传输音频
            ws_url = f"ws://localhost:8000/wss/{self.simulation.session_id}"
            
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    async with self.session.ws_connect(ws_url) as ws:
                        logger.info(f"🔗 User {self.simulation.user_id}: WebSocket connected (attempt {retry_count + 1})")
                        
                        # 🎯 参照index.html的音频处理参数
                        segment_size = 4096
                        target_sample_rate = 16000
                        base_chunk_size = 960 * 16  # 15360 samples
                        current_latency_multiplier = 2
                        
                        # 计算重采样比例
                        resample_ratio = target_sample_rate / original_sample_rate
                        logger.info(f"   - Target sample rate: {target_sample_rate}Hz")
                        logger.info(f"   - Resample ratio: {resample_ratio:.3f}")
                        logger.info(f"   - Segment size: {segment_size} samples")
                        logger.info(f"   - Base chunk size: {base_chunk_size} samples")
                        
                        # 重采样音频数据（参照index.html的重采样逻辑）
                        if original_sample_rate != target_sample_rate:
                            resampled_length = int(len(audio_data) * resample_ratio)
                            resampled_audio = np.zeros(resampled_length, dtype=np.float32)
                            
                            for i in range(resampled_length):
                                original_index = int(i / resample_ratio)
                                if original_index < len(audio_data):
                                    resampled_audio[i] = audio_data[original_index]
                            
                            audio_data = resampled_audio
                            sample_rate = target_sample_rate
                            logger.info(f"   - Resampled to {len(audio_data)} samples @ {sample_rate}Hz")
                        else:
                            sample_rate = original_sample_rate
                        
                        # 🎯 参照index.html的缓冲区和分块逻辑
                        resampled_buffer = np.array([], dtype=np.float32)
                        target_chunk_size = base_chunk_size * current_latency_multiplier
                        
                        logger.info(f"   - Target chunk size: {target_chunk_size} samples ({target_chunk_size/sample_rate:.3f}s)")
                        logger.info(f"   - Segment size: {segment_size} samples ({segment_size/sample_rate:.3f}s)")
                        
                        # 🎯 模拟实时音频流处理 (参照index.html的音频处理流程)
                        audio_position = 0
                        start_time = time.time()
                        chunks_sent = 0
                        
                        while audio_position < len(audio_data):
                            # 检查WebSocket连接状态
                            if ws.closed:
                                raise aiohttp.ClientConnectionError("WebSocket connection closed unexpectedly")
                            
                            # 计算当前应该处理到的位置（基于实时播放）
                            elapsed_time = time.time() - start_time
                            expected_position = int(elapsed_time * sample_rate)
                            
                            # 如果我们处理得太快，等待一下以模拟实时播放
                            if audio_position >= expected_position:
                                sleep_time = (audio_position - expected_position) / sample_rate
                                if sleep_time > 0:
                                    await asyncio.sleep(min(sleep_time, 0.1))  # 最多等待100ms
                            
                            # 获取下一个segment（参照index.html的segment_size）
                            segment_end = min(audio_position + segment_size, len(audio_data))
                            current_segment = audio_data[audio_position:segment_end]
                            
                            # 添加到重采样缓冲区（参照index.html的缓冲区管理）
                            resampled_buffer = np.concatenate([resampled_buffer, current_segment])
                            
                            # 记录输入时间戳（当segment进入缓冲区时）
                            input_timestamp = time.time()
                            
                            # 当缓冲区足够大时，发送chunk（参照index.html的chunk发送逻辑）
                            while len(resampled_buffer) >= target_chunk_size:
                                chunk = resampled_buffer[:target_chunk_size]
                                resampled_buffer = resampled_buffer[target_chunk_size:]
                                
                                # 发送音频chunk（参照index.html发送Float32Array）
                                try:
                                    if ws.closed:
                                        raise aiohttp.ClientConnectionError("WebSocket closed before sending chunk")
                                        
                                    await ws.send_bytes(chunk.astype(np.float32).tobytes())
                                    chunks_sent += 1
                                    logger.debug(f"📤 User {self.simulation.user_id}: Sent chunk {chunks_sent} ({len(chunk)} samples)")
                                    
                                except Exception as chunk_error:
                                    logger.error(f"❌ User {self.simulation.user_id}: Error sending chunk {chunks_sent}: {chunk_error}")
                                    raise
                                
                                # 接收翻译结果（非阻塞）
                                try:
                                    msg = await asyncio.wait_for(ws.receive(), timeout=0.1)
                                    if msg.type == aiohttp.WSMsgType.TEXT:
                                        text = msg.data
                                        output_timestamp = time.time()
                                        
                                        # 计算字符级延迟
                                        if text and not text.startswith("ERROR:") and not text.startswith("READY:"):
                                            # 🎯 更精确的延迟计算：每个字符使用相同的延迟
                                            chunk_delay = output_timestamp - input_timestamp
                                            for char in text:
                                                self.simulation.delays.append(chunk_delay)
                                                self.simulation.total_characters += 1
                                            
                                            logger.debug(f"📤 User {self.simulation.user_id}: Received '{text}' (delay: {chunk_delay:.3f}s)")
                                    elif msg.type == aiohttp.WSMsgType.ERROR:
                                        logger.warning(f"⚠️ User {self.simulation.user_id}: WebSocket error: {msg.data}")
                                    elif msg.type == aiohttp.WSMsgType.CLOSE:
                                        logger.warning(f"⚠️ User {self.simulation.user_id}: WebSocket closed by server")
                                        raise aiohttp.ClientConnectionError("WebSocket closed by server")
                                    
                                except asyncio.TimeoutError:
                                    # 没有立即收到回复，继续处理
                                    pass
                                except Exception as receive_error:
                                    logger.warning(f"⚠️ User {self.simulation.user_id}: Error receiving message: {receive_error}")
                                    # 继续处理，不中断音频流
                            
                            audio_position = segment_end
                            
                            # 短暂休眠以避免过度占用CPU（参照index.html的处理间隔）
                            await asyncio.sleep(0.01)
                        
                        # 发送剩余的缓冲区数据
                        if len(resampled_buffer) > 0:
                            try:
                                if not ws.closed:
                                    await ws.send_bytes(resampled_buffer.astype(np.float32).tobytes())
                                    logger.debug(f"📤 User {self.simulation.user_id}: Sent final chunk {len(resampled_buffer)} samples")
                            except Exception as e:
                                logger.error(f"❌ User {self.simulation.user_id}: Error sending final chunk: {e}")
                        
                        # 发送结束信号
                        try:
                            if not ws.closed:
                                await ws.send_str("EOF")
                        except Exception as e:
                            logger.warning(f"⚠️ User {self.simulation.user_id}: Error sending EOF: {e}")
                        
                        # 等待剩余的翻译结果
                        final_wait_start = time.time()
                        while time.time() - final_wait_start < 5.0:
                            try:
                                if ws.closed:
                                    break
                                    
                                msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
                                if msg.type == aiohttp.WSMsgType.TEXT:
                                    text = msg.data
                                    if text.startswith("PROCESSING_COMPLETE"):
                                        logger.info(f"✅ User {self.simulation.user_id}: Audio processing completed")
                                        break
                                    elif text and not text.startswith("ERROR:") and not text.startswith("READY:"):
                                        # 处理最终的翻译结果
                                        output_timestamp = time.time()
                                        final_delay = output_timestamp - input_timestamp
                                        for char in text:
                                            self.simulation.delays.append(final_delay)
                                            self.simulation.total_characters += 1
                                        
                                        logger.debug(f"📤 User {self.simulation.user_id}: Final result '{text}' (delay: {final_delay:.3f}s)")
                                elif msg.type == aiohttp.WSMsgType.CLOSE:
                                    break
                                    
                            except asyncio.TimeoutError:
                                continue
                            except Exception as e:
                                logger.warning(f"⚠️ User {self.simulation.user_id}: Error in final wait: {e}")
                                break
                        
                        logger.info(f"🏁 User {self.simulation.user_id}: Audio processing completed. Chunks sent: {chunks_sent}, Total characters: {self.simulation.total_characters}")
                        
                        # 成功完成，退出重试循环
                        break
                        
                except (aiohttp.ClientConnectionError, aiohttp.ClientResponseError, ConnectionResetError) as e:
                    retry_count += 1
                    logger.warning(f"🔄 User {self.simulation.user_id}: Connection error (attempt {retry_count}/{max_retries}): {e}")
                    
                    if retry_count < max_retries:
                        # 等待一段时间后重试
                        wait_time = min(2 ** retry_count, 10)  # 指数退避，最多10秒
                        logger.info(f"⏳ User {self.simulation.user_id}: Waiting {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"❌ User {self.simulation.user_id}: Max retries exceeded, giving up")
                        raise
                        
                except Exception as e:
                    logger.error(f"❌ User {self.simulation.user_id}: Unexpected error: {e}")
                    raise
        
        finally:
            # 停止ping任务
            ping_task.cancel()
            try:
                await ping_task
            except asyncio.CancelledError:
                pass

    
    async def _calculate_metrics(self):
        """计算用户指标"""
        # 首先尝试从服务器获取精确的延迟数据
        await self._fetch_server_delays()
        
        # 使用本地收集的延迟数据作为备用
        if self.simulation.delays:
            self.simulation.stream_laal = statistics.mean(self.simulation.delays)
            self.simulation.total_segments = len(self.simulation.delays)
            logger.info(f"📊 User {self.simulation.user_id}: streamLAAL = {self.simulation.stream_laal:.3f}s "
                       f"({self.simulation.total_characters} chars, {self.simulation.total_segments} segments)")
        else:
            self.simulation.stream_laal = 0.0
            logger.warning(f"⚠️ User {self.simulation.user_id}: No delays recorded")
    
    async def _fetch_server_delays(self):
        """从服务器获取精确的延迟数据"""
        try:
            if not self.simulation.session_id:
                return
            
            # 直接使用session_id作为URL参数，让FastAPI自动处理
            async with self.session.get(f"{self.config.server_url}/session_delays/{self.simulation.session_id}", params={"include_details": True}) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("success", False):
                        delay_stats = data.get("delays", {})
                        
                        # 更新统计数据
                        if "stream_laal" in delay_stats:
                            self.simulation.stream_laal = delay_stats["stream_laal"]
                            self.simulation.total_characters = delay_stats.get("total_characters", 0)
                            self.simulation.total_segments = delay_stats.get("segments", 0)
                            
                            # 如果有详细的字符延迟数据，使用它
                            if "character_delays" in delay_stats:
                                self.simulation.delays = [cd["delay"] for cd in delay_stats["character_delays"]]
                            
                            logger.info(f"🎯 User {self.simulation.user_id}: Server delays - streamLAAL = {self.simulation.stream_laal:.3f}s "
                                       f"({self.simulation.total_characters} chars, {self.simulation.total_segments} segments)")
                        else:
                            logger.warning(f"⚠️ User {self.simulation.user_id}: No server delay data available")
                    else:
                        logger.warning(f"⚠️ User {self.simulation.user_id}: Failed to get server delays: {data.get('error', 'Unknown error')}")
                else:
                    logger.debug(f"⚠️ User {self.simulation.user_id}: Server delays API returned status {response.status}")
                    
        except Exception as e:
            logger.debug(f"⚠️ User {self.simulation.user_id}: Failed to fetch server delays: {e}")
            # 使用本地数据作为备用
    
    async def _ping_session_periodically(self):
        """定期发送ping以保持session活跃"""
        ping_interval = self.config.ping_interval  # 使用配置的ping间隔
        
        while True:
            try:
                await asyncio.sleep(ping_interval)
                
                if self.simulation.session_id:
                    try:
                        # 直接使用session_id作为参数，让FastAPI自动处理
                        async with self.session.post(f"{self.config.server_url}/ping", params={"session_id": self.simulation.session_id}) as response:
                            if response.status == 200:
                                logger.debug(f"💓 User {self.simulation.user_id}: Session ping successful")
                            else:
                                logger.warning(f"⚠️ User {self.simulation.user_id}: Session ping failed with status {response.status}")
                    except Exception as e:
                        logger.warning(f"⚠️ User {self.simulation.user_id}: Session ping error: {e}")
                        
            except asyncio.CancelledError:
                logger.debug(f"🔇 User {self.simulation.user_id}: Ping task cancelled")
                break
            except Exception as e:
                logger.error(f"❌ User {self.simulation.user_id}: Ping task error: {e}")
                # 继续运行，不要因为ping失败而停止

class EvaluationFramework:
    """评估框架主类"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results = None
        
    async def run_evaluation(self) -> EvaluationResults:
        """运行完整评估"""
        logger.info(f"🚀 Starting evaluation with {self.config.num_users} users")
        logger.info(f"   - Language split: {self.config.language_split*100:.0f}% Chinese, {(1-self.config.language_split)*100:.0f}% Italian")
        logger.info(f"   - Arrival rate: {self.config.arrival_rate} users/second")
        logger.info(f"   - Dynamic scheduling: {self.config.use_dynamic_schedule}")
        
        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 生成用户模拟
        users = self._generate_user_simulations()
        
        # 初始化结果
        self.results = EvaluationResults(
            config=self.config,
            users=users,
            start_time=time.time(),
            end_time=0.0,
            total_duration=0.0
        )
        
        # 启动服务器配置（如果需要动态调度）
        if self.config.use_dynamic_schedule:
            await self._configure_server()
        
        # 模拟泊松到达并执行用户
        await self._simulate_poisson_arrivals(users)
        
        # 计算汇总统计
        self._calculate_summary_statistics()
        
        # 导出结果
        await self._export_results()
        
        self.results.end_time = time.time()
        self.results.total_duration = self.results.end_time - self.results.start_time
        
        logger.info(f"✅ Evaluation completed in {self.results.total_duration:.1f}s")
        return self.results
    
    def _generate_user_simulations(self) -> List[UserSimulation]:
        """生成用户模拟配置"""
        users = []
        
        for i in range(self.config.num_users):
            # 生成泊松到达时间
            if i == 0:
                arrival_time = 0.0
            else:
                # 泊松过程：指数分布的间隔时间
                interval = random.expovariate(self.config.arrival_rate)
                arrival_time = users[-1].arrival_time + interval
            
            # 分配语言对
            if random.random() < self.config.language_split:
                language_pair = "English -> Chinese"
            else:
                language_pair = "English -> Italian"
            
            # 随机选择测试视频
            video_file = random.choice(self.config.test_videos)
            
            user = UserSimulation(
                user_id=f"eval_user_{i:03d}_{uuid.uuid4().hex[:8]}",
                language_pair=language_pair,
                video_file=video_file,
                arrival_time=arrival_time
            )
            
            users.append(user)
        
        return users
    
    async def _configure_server(self):
        """配置服务器的动态调度参数"""
        # TODO: 实现服务器配置API调用
        logger.info(f"🔧 Configuring server for dynamic scheduling")
        logger.info(f"   - max_batch_size: {self.config.max_batch_size}")
        logger.info(f"   - batch_timeout: {self.config.batch_timeout}s")
        
        # 这里可以添加对服务器配置API的调用
    
    async def _simulate_poisson_arrivals(self, users: List[UserSimulation]):
        """模拟泊松到达过程"""
        logger.info(f"⏰ Simulating Poisson arrivals over {self.config.test_duration}s")
        
        # 启动所有用户任务
        tasks = []
        
        for user in users:
            # 如果到达时间超过测试持续时间，跳过
            if user.arrival_time > self.config.test_duration:
                user.status = "skipped"
                continue
                
            # 创建延迟任务
            task = asyncio.create_task(self._delayed_user_start(user))
            tasks.append(task)
        
        # 等待所有任务完成或超时
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.test_duration + 300  # 额外5分钟缓冲
            )
        except asyncio.TimeoutError:
            logger.warning("⏰ Some users did not complete within timeout")
            
            # 取消未完成的任务
            for task in tasks:
                if not task.done():
                    task.cancel()
    
    async def _delayed_user_start(self, user: UserSimulation) -> UserSimulation:
        """延迟启动用户"""
        # 等待到达时间
        await asyncio.sleep(user.arrival_time)
        
        logger.info(f"👤 User {user.user_id} arriving at t={user.arrival_time:.1f}s ({user.language_pair})")
        
        # 启动用户模拟
        simulated_user = SimulatedUser(user, self.config)
        return await simulated_user.run()
    
    def _calculate_summary_statistics(self):
        """计算汇总统计"""
        assert self.results is not None, "Results must be initialized before calculating statistics"
        
        completed_users = [u for u in self.results.users if u.status == "completed" and u.stream_laal is not None]
        failed_users = [u for u in self.results.users if u.status == "failed"]
        
        self.results.completed_users = len(completed_users)
        self.results.failed_users = len(failed_users)
        
        if completed_users:
            # 过滤掉None值，确保只有有效的streamLAAL数据
            laals = [u.stream_laal for u in completed_users if u.stream_laal is not None]
            
            if laals:  # 确保有有效数据
                self.results.avg_stream_laal = statistics.mean(laals)
                self.results.median_stream_laal = statistics.median(laals)
                self.results.std_stream_laal = statistics.stdev(laals) if len(laals) > 1 else 0.0
                self.results.min_stream_laal = min(laals)
                self.results.max_stream_laal = max(laals)
                
                # 按语言分组统计
                chinese_laals = [u.stream_laal for u in completed_users if "Chinese" in u.language_pair and u.stream_laal is not None]
                italian_laals = [u.stream_laal for u in completed_users if "Italian" in u.language_pair and u.stream_laal is not None]
                
                if chinese_laals:
                    self.results.chinese_results = {
                        "count": len(chinese_laals),
                        "avg_stream_laal": statistics.mean(chinese_laals),
                        "std_stream_laal": statistics.stdev(chinese_laals) if len(chinese_laals) > 1 else 0.0
                    }
                
                if italian_laals:
                    self.results.italian_results = {
                        "count": len(italian_laals),
                        "avg_stream_laal": statistics.mean(italian_laals),
                        "std_stream_laal": statistics.stdev(italian_laals) if len(italian_laals) > 1 else 0.0
                    }
        
        logger.info(f"📊 Summary: {self.results.completed_users} completed, {self.results.failed_users} failed")
        logger.info(f"📊 Overall streamLAAL: {self.results.avg_stream_laal:.3f}s ± {self.results.std_stream_laal:.3f}s")
    
    async def _export_results(self):
        """导出结果到文件"""
        assert self.results is not None, "Results must be initialized before exporting"
        
        timestamp = int(time.time())
        
        # 导出详细结果
        detailed_results_path = os.path.join(
            self.config.output_dir, 
            f"evaluation_detailed_{timestamp}.json"
        )
        
        # 准备导出数据
        export_data = {
            "config": {
                "num_users": self.config.num_users,
                "language_split": self.config.language_split,
                "arrival_rate": self.config.arrival_rate,
                "test_duration": self.config.test_duration,
                "use_dynamic_schedule": self.config.use_dynamic_schedule,
                "max_batch_size": self.config.max_batch_size,
                "batch_timeout": self.config.batch_timeout
            },
            "summary": {
                "completed_users": self.results.completed_users,
                "failed_users": self.results.failed_users,
                "avg_stream_laal": self.results.avg_stream_laal,
                "median_stream_laal": self.results.median_stream_laal,
                "std_stream_laal": self.results.std_stream_laal,
                "min_stream_laal": self.results.min_stream_laal,
                "max_stream_laal": self.results.max_stream_laal,
                "chinese_results": self.results.chinese_results,
                "italian_results": self.results.italian_results
            },
            "users": []
        }
        
        # 添加用户详细数据
        for user in self.results.users:
            user_data = {
                "user_id": user.user_id,
                "language_pair": user.language_pair,
                "video_file": user.video_file,
                "arrival_time": user.arrival_time,
                "start_time": user.start_time,
                "end_time": user.end_time,
                "stream_laal": user.stream_laal,
                "total_characters": user.total_characters,
                "total_segments": user.total_segments,
                "status": user.status,
                "errors": user.errors
            }
            export_data["users"].append(user_data)
        
        # 写入详细结果文件
        async with aiofiles.open(detailed_results_path, 'w') as f:
            await f.write(json.dumps(export_data, indent=2, ensure_ascii=False))
        
        logger.info(f"📝 Detailed results exported to {detailed_results_path}")
        
        # 导出simuleval格式的instance.log文件
        await self._export_simuleval_logs()
        
        # 生成汇总报告
        await self._generate_summary_report(timestamp)
    
    async def _export_simuleval_logs(self):
        """导出simuleval兼容的instance.log文件"""
        for user in self.results.users:
            if user.status == "completed":
                log_path = os.path.join(
                    self.config.output_dir,
                    f"instance_{user.user_id}.log"
                )
                
                # 首先尝试从服务器导出精确的延迟数据
                server_exported = await self._export_server_delays(user, log_path)
                
                if not server_exported and user.delays:
                    # 备用方案：使用本地收集的延迟数据
                    await self._export_local_delays(user, log_path)
                    
                logger.debug(f"📝 Simuleval log exported for {user.user_id}")
    
    async def _export_server_delays(self, user: UserSimulation, log_path: str) -> bool:
        """从服务器导出精确的延迟数据"""
        try:
            if not user.session_id:
                return False
                
            # 调用服务器的导出API
            async with aiohttp.ClientSession() as temp_session:
                # 直接使用session_id作为参数，让FastAPI自动处理
                async with temp_session.post(
                    f"{self.config.server_url}/export_session_delays", 
                    params={
                        "session_id": user.session_id,
                        "filepath": log_path
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("success", False):
                            logger.info(f"🎯 User {user.user_id}: Server delays exported to {log_path}")
                            return True
                        else:
                            logger.warning(f"⚠️ User {user.user_id}: Server export failed: {data.get('error', 'Unknown error')}")
                    else:
                        logger.debug(f"⚠️ User {user.user_id}: Server export API returned status {response.status}")
                        
        except Exception as e:
            logger.debug(f"⚠️ User {user.user_id}: Failed to export server delays: {e}")
            
        return False
    
    async def _export_local_delays(self, user: UserSimulation, log_path: str):
        """使用本地收集的延迟数据导出（备用方案）"""
        log_entries = []
        
        # 🔥 改进：尝试从服务器获取真实的翻译文本
        real_segments = []
        try:
            if user.session_id:
                async with aiohttp.ClientSession() as temp_session:
                    # 直接使用session_id作为URL参数，让FastAPI自动处理
                    async with temp_session.get(
                        f"{self.config.server_url}/session_delays/{user.session_id}",
                        params={"include_details": True}
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get("success", False):
                                delay_stats = data.get("delays", {})
                                if "segment_logs" in delay_stats:
                                    real_segments = delay_stats["segment_logs"]
                                    logger.info(f"🎯 User {user.user_id}: Retrieved {len(real_segments)} real segments from server")
        except Exception as e:
            logger.debug(f"⚠️ User {user.user_id}: Failed to get real segments: {e}")
        
        # 改进的本地延迟导出 - 优先使用真实数据
        if real_segments:
            # 使用从服务器获取的真实数据
            for segment in real_segments:
                entry = {
                    "segment_id": segment["segment_id"],
                    "src": segment["src"],  # 真实的源文本（虽然可能是占位符）
                    "tgt": segment["tgt"],  # 真实的翻译文本
                    "tokens": segment["tokens"],  # 真实的token
                    "delays": segment["delays"],  # 真实的延迟
                    "input_start_time": segment.get("input_start_time", 0),
                    "output_time": segment.get("output_time", 0),
                    "average_delay": segment.get("average_delay", 0),
                    "user_id": user.user_id,
                    "language_pair": user.language_pair,
                    "video_file": user.video_file,
                    "estimated": False  # 标记这是真实数据
                }
                log_entries.append(entry)
        else:
            raise ValueError("No real segments found")
        
        # 写入文件
        async with aiofiles.open(log_path, 'w') as f:
            for entry in log_entries:
                await f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        data_type = "real" if real_segments else "estimated"
        logger.info(f"📝 User {user.user_id}: Local delays exported to {log_path} ({len(log_entries)} entries, {data_type} data)")
    
    async def _generate_summary_report(self, timestamp: int):
        """生成汇总报告"""
        report_path = os.path.join(
            self.config.output_dir,
            f"evaluation_summary_{timestamp}.txt"
        )
        
        report_lines = [
            "=" * 80,
            "InfiniSST Evaluation Framework - Summary Report",
            "=" * 80,
            "",
            f"Test Configuration:",
            f"  - Number of users: {self.config.num_users}",
            f"  - Language split: {self.config.language_split*100:.0f}% Chinese, {(1-self.config.language_split)*100:.0f}% Italian",
            f"  - Arrival rate: {self.config.arrival_rate} users/second",
            f"  - Test duration: {self.config.test_duration}s",
            f"  - Dynamic scheduling: {self.config.use_dynamic_schedule}",
            "",
            f"Results Summary:",
            f"  - Completed users: {self.results.completed_users}",
            f"  - Failed users: {self.results.failed_users}",
            f"  - Success rate: {self.results.completed_users/(self.results.completed_users+self.results.failed_users)*100:.1f}%",
            "",
            f"StreamLAAL Metrics:",
            f"  - Average: {self.results.avg_stream_laal:.3f}s",
            f"  - Median: {self.results.median_stream_laal:.3f}s", 
            f"  - Std Dev: {self.results.std_stream_laal:.3f}s",
            f"  - Min: {self.results.min_stream_laal:.3f}s",
            f"  - Max: {self.results.max_stream_laal:.3f}s",
            ""
        ]
        
        # 按语言分组的结果
        if self.results.chinese_results:
            report_lines.extend([
                f"Chinese Translation Results:",
                f"  - Count: {self.results.chinese_results['count']}",
                f"  - Average streamLAAL: {self.results.chinese_results['avg_stream_laal']:.3f}s",
                f"  - Std Dev: {self.results.chinese_results['std_stream_laal']:.3f}s",
                ""
            ])
        
        if self.results.italian_results:
            report_lines.extend([
                f"Italian Translation Results:",
                f"  - Count: {self.results.italian_results['count']}",
                f"  - Average streamLAAL: {self.results.italian_results['avg_stream_laal']:.3f}s",
                f"  - Std Dev: {self.results.italian_results['std_stream_laal']:.3f}s",
                ""
            ])
        
        report_lines.extend([
            "=" * 80,
            f"Test completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total duration: {self.results.total_duration:.1f}s",
            "=" * 80
        ])
        
        # 写入报告文件
        async with aiofiles.open(report_path, 'w') as f:
            await f.write('\n'.join(report_lines))
        
        # 同时打印到控制台
        print('\n'.join(report_lines))
        
        logger.info(f"📊 Summary report saved to {report_path}")

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="InfiniSST Evaluation Framework")
    parser.add_argument("--num-users", type=int, default=16, help="Number of concurrent users")
    parser.add_argument("--language-split", type=float, default=0.5, help="Fraction of Chinese users (0.0-1.0)")
    parser.add_argument("--arrival-rate", type=float, default=2.0, help="Poisson arrival rate (users/second)")
    parser.add_argument("--test-duration", type=int, default=300, help="Test duration in seconds")
    parser.add_argument("--server-url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory")
    parser.add_argument("--use-dynamic-schedule", action="store_true", help="Enable dynamic scheduling")
    parser.add_argument("--max-batch-size", type=int, default=32, help="Maximum batch size")
    parser.add_argument("--batch-timeout", type=float, default=0.1, help="Batch timeout in seconds")
    
    args = parser.parse_args()
    
    # 创建测试配置
    config = TestConfig(
        num_users=args.num_users,
        language_split=args.language_split,
        arrival_rate=args.arrival_rate,
        test_duration=args.test_duration,
        server_url=args.server_url,
        output_dir=args.output_dir,
        use_dynamic_schedule=args.use_dynamic_schedule,
        max_batch_size=args.max_batch_size,
        batch_timeout=args.batch_timeout
    )
    
    # 运行评估
    framework = EvaluationFramework(config)
    results = await framework.run_evaluation()
    
    print(f"\n🎉 Evaluation completed!")
    print(f"Results saved to: {config.output_dir}")
    print(f"Overall streamLAAL: {results.avg_stream_laal:.3f}s ± {results.std_stream_laal:.3f}s")

if __name__ == "__main__":
    asyncio.run(main()) 