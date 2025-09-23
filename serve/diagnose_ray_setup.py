#!/usr/bin/env python3
"""
Ray API Environment Diagnosis Script
Ray API环境诊断脚本
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def print_header(title):
    """打印标题"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def check_python_environment():
    """检查Python环境"""
    print_header("Python Environment Check")
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    
    # 检查PYTHONPATH
    pythonpath = os.environ.get('PYTHONPATH', 'Not set')
    print(f"PYTHONPATH: {pythonpath}")
    
    # 检查conda环境
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'Not in conda environment')
    print(f"Conda environment: {conda_env}")

def check_file_structure():
    """检查文件结构"""
    print_header("File Structure Check")
    
    project_root = "/home/jiaxuanluo/new-infinisst"
    
    required_files = [
        "serve/ray_api.py",
        "serve/ray_config.py", 
        "serve/ray_serving_system.py",
        "serve/test_ray_api.py",
        "serve/start_ray_api.sh"
    ]
    
    optional_files = [
        "serve/ray_config.json",
        "serve/static/index.html",
        "static/index.html"
    ]
    
    print("Required files:")
    for file_path in required_files:
        full_path = Path(project_root) / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - MISSING")
    
    print("\nOptional files:")
    for file_path in optional_files:
        full_path = Path(project_root) / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ⚠️  {file_path} - Not found")
    
    # 检查目录
    required_dirs = [
        "serve",
        "logs"
    ]
    
    print("\nDirectories:")
    for dir_path in required_dirs:
        full_path = Path(project_root) / dir_path
        if full_path.exists():
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ - MISSING")

def check_python_packages():
    """检查Python包"""
    print_header("Python Packages Check")
    
    required_packages = [
        "ray",
        "fastapi", 
        "uvicorn",
        "websockets",
        "aiohttp",
        "numpy",
        "torch"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            if package == "ray":
                import ray
                print(f"  ✅ {package} - version {ray.__version__}")
            elif package == "torch":
                import torch
                print(f"  ✅ {package} - version {torch.__version__}")
            else:
                print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - NOT INSTALLED")

def check_gpu_environment():
    """检查GPU环境"""
    print_header("GPU Environment Check")
    
    # 检查CUDA_VISIBLE_DEVICES
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_devices}")
    
    # 尝试运行nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '-L'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("Available GPUs:")
            for line in result.stdout.strip().split('\n')[:4]:
                print(f"  {line}")
        else:
            print("❌ nvidia-smi failed")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ nvidia-smi not available or timed out")
    
    # 检查PyTorch GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"PyTorch detects {gpu_count} GPU(s)")
            for i in range(min(gpu_count, 4)):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("❌ PyTorch cannot detect CUDA GPUs")
    except Exception as e:
        print(f"❌ Error checking PyTorch GPU: {e}")

def check_ray_installation():
    """检查Ray安装"""
    print_header("Ray Installation Check")
    
    try:
        import ray
        print(f"✅ Ray version: {ray.__version__}")
        
        # 尝试初始化Ray（测试模式）
        try:
            ray.init(local_mode=True, logging_level=logging.ERROR)
            print("✅ Ray initialization test: SUCCESS")
            ray.shutdown()
        except Exception as e:
            print(f"❌ Ray initialization test failed: {e}")
            
    except ImportError:
        print("❌ Ray not installed")
        print("Install with: pip install ray[default]")

def check_ports():
    """检查端口占用"""
    print_header("Port Check")
    
    ports_to_check = [8000, 8265, 6379]  # API, Ray Dashboard, Redis
    
    for port in ports_to_check:
        try:
            result = subprocess.run(['lsof', f'-i:{port}'], 
                                  capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                print(f"⚠️  Port {port} is in use:")
                for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                    print(f"    {line}")
            else:
                print(f"✅ Port {port} is available")
        except FileNotFoundError:
            print(f"⚠️  Cannot check port {port} (lsof not available)")

def generate_fix_suggestions():
    """生成修复建议"""
    print_header("Fix Suggestions")
    
    print("If you encountered issues, try these fixes:")
    print()
    
    print("1. 🔧 Fix missing files:")
    print("   - Ensure you're in the correct directory: /home/jiaxuanluo/new-infinisst")
    print("   - Run: git pull  # to get latest files")
    print()
    
    print("2. 🐍 Fix Python environment:")
    print("   - Activate conda: conda activate infinisst")
    print("   - Set PYTHONPATH: export PYTHONPATH=/home/jiaxuanluo/new-infinisst")
    print()
    
    print("3. 📦 Install missing packages:")
    print("   - Ray: pip install ray[default]")
    print("   - FastAPI: pip install fastapi uvicorn websockets aiohttp")
    print()
    
    print("4. 🔧 Fix directory issues:")
    print("   - Create directories: mkdir -p logs serve/static")
    print("   - Copy static files: cp -r static/* serve/static/ 2>/dev/null || true")
    print()
    
    print("5. 🚀 Start Ray API:")
    print("   - Simple start: bash serve/start_ray_api.sh")
    print("   - SLURM start: sbatch serve/ray_api.sh")
    print()
    
    print("6. 🧪 Test the API:")
    print("   - Run tests: python serve/test_ray_api.py")
    print("   - Health check: curl http://localhost:8000/health")

def main():
    """主函数"""
    print("🔍 Ray API Environment Diagnosis")
    print("诊断Ray API环境设置")
    
    # 切换到项目目录
    project_root = "/home/jiaxuanluo/new-infinisst"
    try:
        os.chdir(project_root)
        print(f"✅ Changed to project directory: {project_root}")
    except Exception as e:
        print(f"❌ Cannot change to project directory: {e}")
    
    # 运行所有检查
    check_python_environment()
    check_file_structure()
    check_python_packages()
    check_gpu_environment()
    check_ray_installation()
    check_ports()
    generate_fix_suggestions()
    
    print(f"\n{'='*60}")
    print("🎯 Diagnosis Complete")
    print("诊断完成")
    print('='*60)

if __name__ == "__main__":
    # 设置日志级别以避免Ray警告
    import logging
    logging.getLogger("ray").setLevel(logging.ERROR)
    
    main() 