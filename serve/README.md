# 实时语音翻译 Demo

这是一个基于FastAPI和WebSocket的实时语音翻译演示应用。

## 功能特点

- 支持InfiniSST翻译模型（StreamAtt暂时禁用）
- 支持多种语言方向（英译中、英译德、英译西）
- 可动态调整延迟倍数，平衡翻译速度和准确性
- 实时音频处理和翻译
- 基于WebSocket的低延迟通信
- 支持重置翻译状态，无需重新加载模型

## 安装依赖

```bash
pip install fastapi uvicorn python-multipart websockets soundfile numpy
```

## 运行服务器

```bash
cd serve
python run.py
```

服务器将在 http://localhost:8000 上启动。

## 使用方法

1. 打开浏览器访问 http://localhost:8000
2. 选择翻译模型（如InfiniSST）和语言方向（如English -> Chinese）
3. 点击"Load Model"加载模型
4. 选择延迟倍数（影响翻译的响应速度和准确性）
   - 1x: 最快响应，处理960*16个样本，但准确性可能较低
   - 2x: 默认设置，处理960*16*2个样本，平衡速度和准确性
   - 3x: 更准确，处理960*16*3个样本，但响应较慢
   - 4x: 最准确，处理960*16*4个样本，但响应最慢
5. 点击"Update Latency"应用新的延迟设置（无需重新加载模型）
6. 上传音频文件
7. 点击"Start Translation"开始翻译
8. 音频将自动播放，同时显示实时翻译结果
9. 翻译过程中可随时调整延迟倍数并点击"Update Latency"应用
10. 如需停止当前翻译并重置状态，点击"Reset Translation"（无需重新加载模型）
    - 重置后可以重新开始翻译或调整延迟倍数
    - 会话保持活跃，无需重新加载模型

## 技术说明

- 前端使用原生JavaScript和Web Audio API进行音频处理
- 后端使用FastAPI和WebSocket进行实时通信
- 音频数据以16kHz采样率处理
- 基础数据块大小为960*16个采样点，实际处理块大小为基础大小乘以延迟倍数
- 延迟倍数越大，一次处理的数据越多，翻译质量越高，但响应速度越慢
- 支持动态调整延迟倍数，无需重新加载模型
- 重置功能保留会话状态，允许在重置后继续使用相同的模型 