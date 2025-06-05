const { contextBridge, ipcRenderer } = require('electron');

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

// 在页面加载完成后注入一些桌面应用特有的样式和功能
window.addEventListener('DOMContentLoaded', () => {
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
}); 