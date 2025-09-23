# InfiniSST Desktop Application

这是 InfiniSST 翻译系统的桌面版本，基于 Electron 构建，可以脱离浏览器独立运行。

## 功能特性

- 🖥️ **独立桌面应用**: 无需浏览器，直接运行
- 🎵 **多媒体支持**: 支持音频和视频文件翻译
- 🎤 **实时语音翻译**: 支持麦克风实时录音翻译
- 🌐 **YouTube 支持**: 支持 YouTube 视频下载和翻译
- 📱 **跨平台**: 支持 Windows、macOS 和 Linux
- 🎨 **现代界面**: 美观的用户界面，支持拖拽和调整大小
- ⚡ **高性能**: 本地运行，响应迅速

## 系统要求

### 基本要求
- **操作系统**: Windows 10+, macOS 10.14+, 或 Ubuntu 18.04+
- **内存**: 至少 8GB RAM (推荐 16GB+)
- **存储**: 至少 10GB 可用空间
- **网络**: 用于下载模型和 YouTube 视频

### GPU 要求 (推荐)
- **NVIDIA GPU**: 支持 CUDA 的显卡，至少 6GB 显存
- **CUDA**: 版本 11.0 或更高
- **cuDNN**: 对应的 cuDNN 版本

### Python 环境
- **Python**: 3.8 - 3.11
- **PyTorch**: 1.12.0 或更高版本
- **其他依赖**: 见 `serve/requirements.txt`

## 安装和运行

### 开发环境设置

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd InfiniSST
   ```

2. **安装 Python 依赖**
   ```bash
   cd serve
   pip install -r requirements.txt
   cd ..
   ```

3. **安装 Node.js 依赖**
   ```bash
   npm install
   ```

4. **下载模型文件**
   ```bash
   ./download_models.sh
   ```

5. **运行开发版本**
   ```bash
   npm run dev
   ```

### 生产环境构建

1. **构建应用**
   ```bash
   npm run build
   ```

2. **打包分发版本**
   ```bash
   # 构建当前平台的安装包
   npm run dist
   
   # 仅打包不创建安装程序
   npm run pack
   ```

构建完成后，安装包将在 `dist` 目录中：
- **Windows**: `.exe` 安装程序
- **macOS**: `.dmg` 磁盘映像
- **Linux**: `.AppImage` 可执行文件

## 使用说明

### 启动应用

1. **开发模式**: `npm run dev`
2. **生产模式**: 运行构建后的安装包

### 基本操作

1. **加载模型**
   - 选择翻译模型 (InfiniSST)
   - 选择语言对 (如 English → Chinese)
   - 设置延迟倍数 (1x-4x)
   - 点击 "Load Model" 按钮

2. **音频/视频翻译**
   - 点击 "Upload Video/Audio" 上传本地文件
   - 或使用菜单 File → Open Audio File (Ctrl/Cmd+O)
   - 播放媒体文件开始翻译

3. **YouTube 翻译**
   - 在输入框中粘贴 YouTube URL
   - 点击播放按钮下载并翻译

4. **实时语音翻译**
   - 点击 "Record Audio" 开始录音
   - 说话时会实时显示翻译结果
   - 再次点击停止录音

### 高级功能

- **翻译面板**: 可拖拽、调整大小、最小化
- **快捷键**: Ctrl/Cmd+O 打开文件
- **设置调整**: 运行时可调整延迟倍数
- **重置功能**: 随时重置翻译状态

## 配置说明

### 模型路径配置

在 `serve/api.py` 中修改模型路径：

```python
model_path = "/path/to/your/models/{}-{}/pytorch_model.bin"
lora_path = "/path/to/your/models/{}-{}/lora.bin"
```

### GPU 配置

设置环境变量指定使用的 GPU：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### 端口配置

默认后端服务运行在 8000-8100 端口范围内，会自动寻找可用端口。

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确认有足够的 GPU 内存
   - 查看控制台错误信息

2. **音频播放问题**
   - 确认系统音频设备正常
   - 检查文件格式是否支持
   - 尝试重新启动应用

3. **麦克风权限问题**
   - 确认应用有麦克风访问权限
   - 检查系统隐私设置
   - 重新启动应用

4. **YouTube 下载失败**
   - 检查网络连接
   - 确认 URL 格式正确
   - 某些视频可能有地区限制

### 日志查看

- **开发模式**: 控制台会显示详细日志
- **生产模式**: 
  - Windows: `%APPDATA%\infinisst-desktop\logs\`
  - macOS: `~/Library/Logs/infinisst-desktop/`
  - Linux: `~/.config/infinisst-desktop/logs/`

## 开发指南

### 项目结构

```
InfiniSST/
├── electron/           # Electron 主进程和预加载脚本
│   ├── main.js        # 主进程
│   ├── preload.js     # 预加载脚本
│   └── assets/        # 应用图标
├── serve/             # 后端 Python 服务
│   ├── api.py         # FastAPI 服务器
│   ├── static/        # 前端静态文件
│   └── requirements.txt
├── agents/            # 翻译模型代理
├── model/             # 模型相关代码
├── package.json       # Node.js 配置
└── README-Electron.md # 本文件
```

### 添加新功能

1. **前端功能**: 修改 `serve/static/index.html`
2. **后端 API**: 修改 `serve/api.py`
3. **Electron 集成**: 修改 `electron/main.js` 和 `electron/preload.js`

### 调试技巧

1. **前端调试**: 开发模式下自动打开开发者工具
2. **后端调试**: 查看控制台输出的后端日志
3. **进程调试**: 使用 `ps aux | grep python` 查看后端进程

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题，请通过以下方式联系：
- GitHub Issues
- 邮箱: [your-email@example.com] 