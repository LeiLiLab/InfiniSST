# 🧪 测试模式 (Test Mode)

这个目录包含用于自动化测试的视频文件。

## 如何启用测试模式

### 方法1：URL参数
在浏览器中访问时添加 `?test=true` 参数：
```
http://localhost:8001/?test=true
```

### 方法2：手动触发
在浏览器开发者控制台中执行：
```javascript
triggerTestMode()
```

## 测试模式功能

当测试模式启用时，系统会自动：

1. **🤖 自动加载模型**
   - 调用 Load Model 按钮的逻辑
   - 等待模型加载完成
   - 创建翻译会话

2. **📼 自动加载测试视频**
   - 从服务器获取 `0000AAAA.mp4` 测试视频
   - 创建媒体播放器
   - 设置翻译事件监听器

3. **▶️ 尝试自动播放**
   - 自动播放测试视频（取决于浏览器的自动播放策略）
   - 如果被浏览器阻止，会显示提示点击播放按钮

## 测试视频文件

系统会在以下位置搜索测试视频：
- `serve/static/test_video/0000AAAA.mp4`
- `~/Downloads/0000AAAA.mp4`
- `/tmp/0000AAAA.mp4`
- `test_video/0000AAAA.mp4`

允许的测试视频文件名：
- `0000AAAA.mp4` (主要测试文件)
- `test.mp4`
- `sample.mp4`

## 状态提示

测试模式会在状态栏显示进度：
- `TEST MODE: Auto-loading model...`
- `TEST MODE: Loading test video...`
- `TEST MODE: Video loaded - Click play to start translation`
- `TEST MODE: Auto-playing and translating...`

## 调试

测试模式的详细日志会在浏览器控制台中显示，以 `🧪 [TEST MODE]` 前缀标识。

## 回退方案

如果服务器上找不到测试视频文件，系统会：
1. 显示错误信息
2. 高亮文件上传按钮
3. 提示手动上传 `~/Downloads/0000AAAA.mp4` 