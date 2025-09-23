# InfiniSST Desktop Client

这是 InfiniSST 翻译系统的桌面客户端，基于 Electron 构建。该客户端连接到远程的 Python 后端服务器，实现了前后端分离的架构。

## 架构说明

```
┌─────────────────┐    HTTP/WebSocket    ┌─────────────────┐
│   Mac Desktop   │ ◄─────────────────► │  Linux Server   │
│                 │                      │                 │
│  Electron App   │                      │  Python Backend │
│  (Frontend)     │                      │  (InfiniSST)    │
└─────────────────┘                      └─────────────────┘
```

- **前端**: Mac 上运行的 Electron 应用，提供用户界面
- **后端**: Linux 服务器上运行的 Python 服务，处理 AI 翻译

## 功能特性

- 🖥️ **独立桌面应用**: 无需浏览器，直接运行
- 🌐 **远程连接**: 连接到远程 Linux 服务器
- 🎵 **多媒体支持**: 支持音频和视频文件翻译
- 🎤 **实时语音翻译**: 支持麦克风实时录音翻译
- 📺 **YouTube 支持**: 支持 YouTube 视频下载和翻译
- 🎨 **现代界面**: 美观的用户界面，支持拖拽和调整大小
- ⚡ **高性能**: 利用远程 GPU 资源

## 系统要求

### 客户端 (Mac)
- **操作系统**: macOS 10.14+
- **内存**: 至少 4GB RAM
- **存储**: 至少 1GB 可用空间
- **网络**: 稳定的网络连接到服务器

### 服务器 (Linux)
- **操作系统**: Ubuntu 18.04+ 或其他 Linux 发行版
- **内存**: 至少 16GB RAM (推荐 32GB+)
- **GPU**: NVIDIA GPU，至少 6GB 显存
- **CUDA**: 版本 11.0 或更高
- **Python**: 3.8 - 3.11
- **网络**: 公网 IP 或内网可访问

## 安装和运行

### 1. 服务器端设置

首先在 Linux 服务器上设置后端服务：

```bash
# 克隆项目
git clone <repository-url>
cd InfiniSST

# 安装 Python 依赖
cd serve
pip install -r requirements.txt

# 下载模型文件
cd ..
./download_models.sh

# 启动服务器 (允许外部访问)
cd serve
python3 api.py --host 0.0.0.0 --port 8001
```

### 2. 客户端设置

在 Mac 上设置 Electron 客户端：

```bash
# 克隆项目 (如果还没有)
git clone <repository-url>
cd InfiniSST

# 安装 Node.js 依赖
npm install

# 启动客户端
npm run electron-dev
```

### 3. 连接配置

启动客户端后，会弹出服务器配置对话框：

1. **Protocol**: 选择 `http` 或 `https`
2. **Host**: 输入服务器 IP 地址或域名
3. **Port**: 输入服务器端口 (默认 8001)
4. 点击 "Connect" 连接

## 使用方法

### 快速开始

1. **启动服务器**:
   ```bash
   # 在 Linux 服务器上
   cd InfiniSST/serve
   python3 api.py --host 0.0.0.0 --port 8001
   ```

2. **启动客户端**:
   ```bash
   # 在 Mac 上
   cd InfiniSST
   npm run electron-dev
   ```

3. **配置连接**: 在弹出的对话框中输入服务器信息

4. **开始翻译**: 加载模型后即可使用各种翻译功能

### 命令行选项

客户端支持命令行参数预设服务器信息：

```bash
# 使用环境变量
export INFINISST_HOST="your-server.com"
export INFINISST_PORT="8001"
export INFINISST_PROTOCOL="http"
npm run electron-dev

# 或使用启动脚本
./scripts/start-electron-client.sh --server your-server.com --port 8001
```

### 基本操作

1. **加载模型**
   - 选择翻译模型 (InfiniSST)
   - 选择语言对 (如 English → Chinese)
   - 设置延迟倍数 (1x-4x)
   - 点击 "Load Model" 按钮

2. **音频/视频翻译**
   - 点击 "Upload Video/Audio" 上传本地文件
   - 或使用菜单 File → Open Audio File (Cmd+O)
   - 播放媒体文件开始翻译

3. **YouTube 翻译**
   - 在输入框中粘贴 YouTube URL
   - 点击播放按钮下载并翻译

4. **实时语音翻译**
   - 点击 "Record Audio" 开始录音
   - 说话时会实时显示翻译结果

## 网络配置

### 防火墙设置

确保服务器防火墙允许客户端访问：

```bash
# Ubuntu/Debian
sudo ufw allow 8001

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=8001/tcp
sudo firewall-cmd --reload
```

### 内网访问

如果在内网环境中使用：

1. 确保客户端和服务器在同一网络
2. 使用服务器的内网 IP 地址
3. 检查网络路由和防火墙设置

### 公网访问

如果通过公网访问：

1. 配置服务器的公网 IP 或域名
2. 设置端口转发 (如果需要)
3. 考虑使用 HTTPS 加密连接
4. 配置适当的安全措施

## 开发指南

### 项目结构

```
InfiniSST/
├── electron/           # Electron 客户端
│   ├── main.js        # 主进程
│   ├── preload.js     # 预加载脚本
│   └── assets/        # 应用图标
├── serve/             # Python 后端服务
│   ├── api.py         # FastAPI 服务器
│   ├── static/        # 前端静态文件
│   └── requirements.txt
├── agents/            # 翻译模型代理
├── model/             # 模型相关代码
├── scripts/           # 启动脚本
├── package.json       # Node.js 配置
└── README-Electron-Remote.md
```

### 开发模式

```bash
# 启动开发模式
npm run electron-dev

# 构建应用
npm run build

# 打包分发
npm run dist
```

### 调试技巧

1. **客户端调试**: 开发模式下自动打开开发者工具
2. **服务器调试**: 查看服务器控制台输出
3. **网络调试**: 使用浏览器开发者工具查看网络请求

## 故障排除

### 连接问题

1. **无法连接到服务器**
   - 检查服务器是否正在运行
   - 验证 IP 地址和端口
   - 检查防火墙设置
   - 测试网络连通性: `ping server-ip`

2. **连接超时**
   - 检查网络延迟
   - 增加连接超时时间
   - 确认服务器负载正常

3. **WebSocket 连接失败**
   - 检查代理服务器设置
   - 确认 WebSocket 支持
   - 查看浏览器控制台错误

### 性能问题

1. **翻译延迟高**
   - 检查网络延迟
   - 调整延迟倍数设置
   - 确认服务器 GPU 资源

2. **音频传输问题**
   - 检查网络带宽
   - 确认音频格式支持
   - 查看服务器日志

### 常见错误

1. **模型加载失败**
   - 检查服务器模型文件
   - 确认 GPU 内存充足
   - 查看服务器错误日志

2. **权限问题**
   - 确认麦克风访问权限
   - 检查文件读取权限
   - 验证网络访问权限

## 部署建议

### 生产环境

1. **使用 HTTPS**: 配置 SSL 证书
2. **负载均衡**: 使用 Nginx 或其他反向代理
3. **监控**: 设置服务器监控和日志
4. **备份**: 定期备份模型和配置文件

### 安全考虑

1. **网络安全**: 使用 VPN 或专用网络
2. **访问控制**: 实现用户认证和授权
3. **数据加密**: 加密敏感数据传输
4. **定期更新**: 保持系统和依赖更新

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 联系方式

如有问题，请通过以下方式联系：
- GitHub Issues
- 邮箱: [your-email@example.com] 