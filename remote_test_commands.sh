#!/bin/bash
# 远程服务器全链路日志测试命令
# 请在远程服务器上直接运行这些命令

echo "🧪 InfiniSST 全链路日志测试 - 远程执行版本"
echo "======================================================"

# 1. 激活conda环境
echo "步骤 1: 激活conda环境"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

# 2. 进入项目目录
echo "步骤 2: 进入项目目录"
cd ~/infinisst-demo-v2

# 3. 检查服务器状态
echo "步骤 3: 检查服务器状态"
echo "🔍 检查服务器进程..."
ps aux | grep -E "(api\.py|scheduler|serve)" | grep -v grep

echo ""
echo "🔍 检查端口占用..."
netstat -tulpn 2>/dev/null | grep -E ":5000|:8000|:8080" || ss -tulpn 2>/dev/null | grep -E ":5000|:8000|:8080"

echo ""
echo "🔍 尝试连接API健康检查..."
curl -s http://localhost:5000/health | head -100 || echo "❌ 5000端口不可用"
curl -s http://localhost:8000/health | head -100 || echo "❌ 8000端口不可用"

# 4. 安装依赖（如果需要）
echo ""
echo "步骤 4: 检查并安装依赖"
python3 -c "import aiohttp" 2>/dev/null || pip install aiohttp

# 5. 运行测试脚本
echo ""
echo "步骤 5: 运行全链路日志测试"
if [ -f "test_full_pipeline_logs.py" ]; then
    python3 test_full_pipeline_logs.py
else
    echo "❌ test_full_pipeline_logs.py 文件不存在"
    echo "请先创建测试文件或同步代码"
fi

# 6. 简化的手动测试
echo ""
echo "步骤 6: 简化的手动测试（如果自动测试失败）"
echo "🔧 手动测试命令："
echo "   curl -X GET http://localhost:5000/status"
echo "   curl -X GET http://localhost:5000/diagnose"
echo ""
echo "🔧 如果服务器未启动，请运行："
echo "   cd ~/infinisst-demo-v2"
echo "   python3 serve/api.py &"
echo "   # 等待几秒后再次测试"

echo ""
echo "======================================================"
echo "测试完成！请查看上面的输出结果" 