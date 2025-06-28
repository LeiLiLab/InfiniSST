const { contextBridge, ipcRenderer } = require('electron');

console.log('=== Preload Script Starting ===');
console.log('Process PID:', process.pid);
console.log('Current URL will be:', process.env.REMOTE_SERVER_URL || 'http://localhost:8001');

try {
  // 使用一个标志来防止重复暴露
  if (global._electronAPIExposed) {
    console.log('⚠️ electronAPI already exposed, skipping re-exposure');
    console.log('This indicates the preload script was executed multiple times');
    return;
  }
  
  console.log('🔧 Exposing electronAPI to main world...');
  global._electronAPIExposed = true;
  
  // 暴露安全的 API 给渲染进程
  contextBridge.exposeInMainWorld('electronAPI', {
    // 获取后端服务端口
    getBackendPort: () => ipcRenderer.invoke('get-backend-port'),
    
    // 显示文件选择对话框
    showOpenDialog: () => ipcRenderer.invoke('show-open-dialog'),
    
    // 监听菜单事件
    onMenuOpenFile: (callback) => {
      ipcRenderer.on('menu-open-file', callback);
    },
    
    // 翻译窗口控制
    showTranslationWindow: () => ipcRenderer.invoke('show-translation-window'),
    hideTranslationWindow: () => ipcRenderer.invoke('hide-translation-window'),
    closeTranslationWindow: () => ipcRenderer.invoke('close-translation-window'),
    minimizeTranslationWindow: () => ipcRenderer.invoke('minimize-translation-window'),
    setWindowSize: (width, height) => ipcRenderer.invoke('set-translation-window-size', width, height),
    getWindowBounds: () => ipcRenderer.invoke('get-translation-window-bounds'),
    setWindowBounds: (bounds) => ipcRenderer.invoke('set-translation-window-bounds', bounds),
    
    // 翻译数据更新
    updateTranslation: (translationData) => ipcRenderer.invoke('update-translation', translationData),
    updateTranslationStatus: (statusData) => ipcRenderer.invoke('update-translation-status', statusData),
    
    // 翻译窗口样式更新
    updateTranslationStyle: (styleData) => ipcRenderer.invoke('update-translation-style', styleData),
    
    // 重置翻译
    resetTranslation: () => ipcRenderer.invoke('reset-translation-from-window'),
    
    // 麦克风权限管理
    checkMicrophonePermission: () => ipcRenderer.invoke('check-microphone-permission'),
    requestMicrophonePermission: () => ipcRenderer.invoke('request-microphone-permission'),
    
    // BlackHole音频驱动管理（替代屏幕录制权限）
    checkBlackHoleInstalled: async () => {
      console.log('📡 Preload: checkBlackHoleInstalled called, invoking IPC...');
      try {
        console.log('📡 Preload: About to call ipcRenderer.invoke with channel "check-blackhole-installed"');
        const result = await ipcRenderer.invoke('check-blackhole-installed');
        console.log('📡 Preload: IPC invoke completed, result:', result);
        return result;
      } catch (error) {
        console.error('📡 Preload: IPC invoke failed:', error);
        throw error;
      }
    },
    installBlackHole: () => {
      console.log('📡 Preload: installBlackHole called, invoking IPC...');
      return ipcRenderer.invoke('install-blackhole');
    },
    testSystemProfiler: () => {
      console.log('📡 Preload: testSystemProfiler called, invoking IPC...');
      return ipcRenderer.invoke('test-system-profiler');
    },
    testIPC: () => {
      console.log('📡 Preload: testIPC called, invoking IPC...');
      return ipcRenderer.invoke('test-ipc');
    },
    
    // 监听翻译窗口事件（用于翻译窗口）
    onTranslationUpdate: (callback) => {
      ipcRenderer.on('translation-update', callback);
    },
    onStatusUpdate: (callback) => {
      ipcRenderer.on('status-update', callback);
    },
    onTranslationStyleUpdate: (callback) => {
      ipcRenderer.on('translation-style-update', callback);
    },
    onResetTranslation: (callback) => {
      ipcRenderer.on('reset-translation', callback);
    },
    
    // 移除监听器
    removeAllListeners: (channel) => {
      ipcRenderer.removeAllListeners(channel);
    },
    
    // 平台信息
    platform: process.platform,
    
    // 版本信息
    versions: {
      node: process.versions.node,
      chrome: process.versions.chrome,
      electron: process.versions.electron
    }
  });
  
  console.log('✅ electronAPI exposed successfully');
  console.log('Available methods:', Object.keys(window.electronAPI || {}));
  
} catch (error) {
  console.error('Error in preload script:', error);
}

// 在页面加载完成后注入一些桌面应用特有的样式和功能
window.addEventListener('DOMContentLoaded', () => {
  console.log('DOMContentLoaded in preload');
  
  // 添加桌面应用标识
  document.body.classList.add('electron-app');
  
  // 添加平台特定的样式类
  document.body.classList.add(`platform-${process.platform}`);
  
  // 禁用右键菜单（可选）
  // document.addEventListener('contextmenu', (e) => {
  //   e.preventDefault();
  // });
  
  // 禁用拖拽文件到窗口（防止意外导航）
  document.addEventListener('dragover', (e) => {
    e.preventDefault();
  });
  
  document.addEventListener('drop', (e) => {
    e.preventDefault();
  });
  
  console.log('Preload DOM setup complete');
}); 