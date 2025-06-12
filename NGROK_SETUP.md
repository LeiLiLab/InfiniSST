# InfiniSST Ngrok 远程测试指南

本指南将帮助你使用ngrok将InfiniSST翻译应用暴露到公网，进行远程测试。

## 🚀 快速开始

### 方法一：自动化脚本（推荐）

1. **启动ngrok隧道和后端服务器**：
   ```bash
   ./start-ngrok-test.sh
   ```
   
2. **复制ngrok提供的HTTPS URL**（类似 `https://abc123.ngrok.io`）

3. **在新终端窗口启动Electron应用**：
   ```bash
   ./start-electron-remote.sh https://abc123.ngrok.io
   ```

### 方法二：手动步骤

1. **启动后端服务器**：
   ```bash
   cd serve
   source env/bin/activate
   python api.py --host 0.0.0.0 --port 8001
   ```

2. **在新终端启动ngrok**：
   ```bash
   ngrok http 8001
   ```

3. **复制ngrok URL并启动Electron**：
   ```bash
   ./start-electron-remote.sh https://your-ngrok-url.ngrok.io
   ```

## 📋 详细步骤

### 1. 安装和配置ngrok

如果还没有安装ngrok：
```bash
brew install ngrok/ngrok/ngrok
```

（可选）注册ngrok账户并设置authtoken：
```bash
ngrok config add-authtoken YOUR_AUTH_TOKEN
```

### 2. 启动服务

运行自动化脚本：
```bash
./start-ngrok-test.sh
```

脚本会：
- ✅ 检查ngrok是否安装
- ✅ 检查后端服务器是否运行，如果没有则自动启动
- ✅ 启动ngrok隧道
- ✅ 显示公网访问URL

### 3. 获取ngrok URL

在ngrok输出中找到类似这样的信息：
```
Forwarding    https://abc123.ngrok.io -> http://localhost:8001
```

复制HTTPS URL（推荐使用HTTPS而不是HTTP）。

### 4. 启动Electron应用

在新终端窗口中：
```bash
./start-electron-remote.sh https://abc123.ngrok.io
```

## 🔧 配置说明

### 环境变量

- `REMOTE_SERVER_URL`: 远程服务器URL
- `ELECTRON_IS_DEV`: 开发模式标志

### 自动检测功能

Electron应用会自动检测是否使用远程连接：
- 如果URL包含`ngrok`或`https://`，会显示"远程连接已建立"
- 否则显示本地连接状态

## 🌐 远程访问

一旦ngrok隧道建立，你可以：

1. **分享URL给其他人**：他们可以通过浏览器访问你的应用
2. **从任何设备访问**：手机、平板、其他电脑等
3. **测试网络功能**：WebSocket连接、文件上传等

## ⚠️ 注意事项

### 安全性
- ngrok免费版的URL是公开的，任何人都可以访问
- 不要在生产环境中使用ngrok
- 避免在公网暴露敏感数据

### 性能
- ngrok会增加一些延迟
- 免费版有带宽限制
- 连接可能不如本地稳定

### URL管理
- 免费版每次重启ngrok都会生成新的URL
- 付费版可以使用固定的自定义域名

## 🛠️ 故障排除

### 常见问题

1. **"Backend server not running"**
   - 确保Python虚拟环境已激活
   - 检查端口8001是否被占用
   - 手动启动服务器进行测试

2. **"Cannot connect to remote server"**
   - 检查ngrok隧道是否正常运行
   - 确认URL格式正确（包含https://）
   - 测试URL是否可以在浏览器中访问

3. **Electron应用无法连接**
   - 确保使用了正确的ngrok URL
   - 检查防火墙设置
   - 查看Electron控制台的错误信息

### 调试技巧

1. **检查ngrok状态**：
   ```bash
   curl -s http://localhost:4040/api/tunnels | jq
   ```

2. **测试后端连接**：
   ```bash
   curl https://your-ngrok-url.ngrok.io
   ```

3. **查看ngrok日志**：
   访问 http://localhost:4040 查看ngrok管理界面

## 📱 移动设备测试

使用ngrok URL，你可以在移动设备上测试应用：

1. 在手机浏览器中打开ngrok URL
2. 测试麦克风权限和录音功能
3. 验证翻译功能是否正常工作

## 🔄 停止服务

按 `Ctrl+C` 停止ngrok和后端服务器。自动化脚本会自动清理所有进程。

## 💡 高级用法

### 自定义端口
```bash
ngrok http 8080  # 使用不同端口
```

### 添加认证
```bash
ngrok http 8001 --auth="username:password"
```

### 使用配置文件
创建 `ngrok.yml` 配置文件进行更复杂的设置。

---

## 📞 获取帮助

如果遇到问题：
1. 检查本文档的故障排除部分
2. 查看ngrok官方文档：https://ngrok.com/docs
3. 检查Electron和后端的控制台日志 