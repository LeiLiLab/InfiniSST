#!/usr/bin/env python3
"""
Test script for Ray-based InfiniSST API
测试Ray版本API功能的脚本
"""

import asyncio
import aiohttp
import json
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RayAPITester:
    """Ray API功能测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = None
        
    async def test_health_check(self):
        """测试健康检查端点"""
        logger.info("🧪 Testing health check...")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Health check passed: {data.get('status', 'unknown')}")
                        return True
                    else:
                        logger.error(f"❌ Health check failed: {response.status}")
                        return False
            except Exception as e:
                logger.error(f"❌ Health check error: {e}")
                return False
    
    async def test_ray_stats(self):
        """测试Ray统计信息端点"""
        logger.info("🧪 Testing Ray stats...")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/ray/stats") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Ray stats success: {data.get('success', False)}")
                        return True
                    else:
                        logger.warning(f"⚠️ Ray stats failed: {response.status}")
                        return False
            except Exception as e:
                logger.error(f"❌ Ray stats error: {e}")
                return False
    
    async def test_session_creation(self):
        """测试会话创建"""
        logger.info("🧪 Testing session creation...")
        
        async with aiohttp.ClientSession() as session:
            try:
                init_data = {
                    "agent_type": "InfiniSST",
                    "language_pair": "English -> Chinese",
                    "client_id": "ray_test_user"
                }
                
                async with session.post(f"{self.base_url}/init", json=init_data) as response:
                    if response.status == 200:
                        data = await response.json()
                        session_id = data.get("session_id")
                        if session_id:
                            self.session_id = session_id
                            logger.info(f"✅ Session created: {session_id}")
                            return True
                        else:
                            logger.error("❌ No session ID received")
                            return False
                    else:
                        logger.error(f"❌ Session creation failed: {response.status}")
                        text = await response.text()
                        logger.error(f"Response: {text}")
                        return False
            except Exception as e:
                logger.error(f"❌ Session creation error: {e}")
                return False
    
    async def test_queue_status(self):
        """测试队列状态查询"""
        if not self.session_id:
            logger.warning("⚠️ No session ID available for queue status test")
            return False
            
        logger.info("🧪 Testing queue status...")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/queue_status/{self.session_id}") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Queue status: {data.get('status', 'unknown')}")
                        return True
                    else:
                        logger.error(f"❌ Queue status failed: {response.status}")
                        return False
            except Exception as e:
                logger.error(f"❌ Queue status error: {e}")
                return False
    
    async def test_load_models(self):
        """测试模型加载"""
        logger.info("🧪 Testing model loading...")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(f"{self.base_url}/load_models") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Model loading: {data.get('success', False)}")
                        return True
                    else:
                        logger.warning(f"⚠️ Model loading failed: {response.status}")
                        return False
            except Exception as e:
                logger.error(f"❌ Model loading error: {e}")
                return False
    
    async def test_debug_endpoints(self):
        """测试调试端点"""
        logger.info("🧪 Testing debug endpoints...")
        
        endpoints = [
            "/debug/session_stats",
            "/debug/session_history"
        ]
        
        results = []
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                try:
                    async with session.get(f"{self.base_url}{endpoint}") as response:
                        if response.status == 200:
                            data = await response.json()
                            logger.info(f"✅ {endpoint}: OK")
                            results.append(True)
                        else:
                            logger.warning(f"⚠️ {endpoint}: {response.status}")
                            results.append(False)
                except Exception as e:
                    logger.error(f"❌ {endpoint} error: {e}")
                    results.append(False)
        
        return all(results)
    
    async def test_ping(self):
        """测试ping功能"""
        if not self.session_id:
            logger.warning("⚠️ No session ID available for ping test")
            return False
            
        logger.info("🧪 Testing ping...")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(f"{self.base_url}/ping", json={"session_id": self.session_id}) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Ping: {data.get('success', False)}")
                        return True
                    else:
                        logger.error(f"❌ Ping failed: {response.status}")
                        return False
            except Exception as e:
                logger.error(f"❌ Ping error: {e}")
                return False
    
    async def cleanup_session(self):
        """清理测试会话"""
        if not self.session_id:
            return
            
        logger.info("🧹 Cleaning up test session...")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(f"{self.base_url}/delete_session", 
                                      json={"session_id": self.session_id}) as response:
                    if response.status == 200:
                        logger.info("✅ Session cleaned up")
                    else:
                        logger.warning(f"⚠️ Session cleanup failed: {response.status}")
            except Exception as e:
                logger.error(f"❌ Session cleanup error: {e}")
    
    async def run_all_tests(self):
        """运行所有测试"""
        logger.info("🚀 Starting Ray API tests...")
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Ray Stats", self.test_ray_stats),
            ("Session Creation", self.test_session_creation),
            ("Queue Status", self.test_queue_status),
            ("Model Loading", self.test_load_models),
            ("Debug Endpoints", self.test_debug_endpoints),
            ("Ping", self.test_ping),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                result = await test_func()
                results[test_name] = result
                if result:
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                logger.error(f"💥 {test_name}: EXCEPTION - {e}")
                results[test_name] = False
            
            # Small delay between tests
            await asyncio.sleep(0.5)
        
        # Cleanup
        await self.cleanup_session()
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("📊 TEST SUMMARY")
        logger.info("="*50)
        
        passed = 0
        failed = 0
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name:20} {status}")
            if result:
                passed += 1
            else:
                failed += 1
        
        logger.info("="*50)
        logger.info(f"Total: {len(results)}, Passed: {passed}, Failed: {failed}")
        
        if failed == 0:
            logger.info("🎉 All tests passed! Ray API is working correctly.")
        else:
            logger.warning(f"⚠️ {failed} test(s) failed. Check the logs above for details.")
        
        return failed == 0

async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Ray-based InfiniSST API")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Base URL of the Ray API server")
    args = parser.parse_args()
    
    tester = RayAPITester(args.url)
    success = await tester.run_all_tests()
    
    if success:
        exit(0)
    else:
        exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 