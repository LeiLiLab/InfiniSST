const { app, BrowserWindow, Menu, dialog, shell, ipcMain, systemPreferences } = require('electron');
const path = require('path');
const isDev = process.env.ELECTRON_IS_DEV === 'true' || require('electron-is-dev');

console.log('=== Electron Main Process Starting ===');
console.log('isDev:', isDev);
console.log('__dirname:', __dirname);

let mainWindow;
let translationWindow;
let backendUrl = null;

// 配置选项
const CONFIG = {
  // 远程服务器配置
  REMOTE_SERVER: {
    host: process.env.INFINISST_HOST || 'localhost',
    port: process.env.INFINISST_PORT || 8001,
    protocol: process.env.INFINISST_PROTOCOL || 'http'
  },
  // 开发模式下的默认配置
  DEV_SERVER: {
    host: 'localhost',
    port: 8001,
    protocol: 'http'
  }
};

console.log('CONFIG:', CONFIG);

// 创建主窗口
function createWindow() {
  console.log('Creating main window...');
  
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      preload: path.join(__dirname, 'preload.js'),
      webSecurity: false // 允许跨域请求到远程服务器
    },
    icon: path.join(__dirname, 'assets', 'icon.png'),
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    show: false // 先不显示，等加载完成后再显示
  });

  console.log('Main window created');

  // 设置应用菜单
  createMenu();

  // 窗口准备好后显示
  mainWindow.once('ready-to-show', () => {
    console.log('Main window ready to show');
    mainWindow.show();
    
    // 开发模式下打开开发者工具
    if (isDev) {
      console.log('Opening DevTools in development mode');
      mainWindow.webContents.openDevTools();
    }
  });

  // 窗口关闭时的处理
  mainWindow.on('closed', () => {
    console.log('Main window closed');
    mainWindow = null;
  });

  // 处理外部链接
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    console.log('Opening external URL:', url);
    shell.openExternal(url);
    return { action: 'deny' };
  });

  // 监听页面加载事件
  mainWindow.webContents.on('did-start-loading', () => {
    console.log('Page started loading');
  });

  mainWindow.webContents.on('did-finish-load', () => {
    console.log('Page finished loading');
  });

  mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription, validatedURL) => {
    console.error('Page failed to load:', {
      errorCode,
      errorDescription,
      validatedURL
    });
  });

  // 连接到后端服务
  connectToBackend();
}

// 连接到后端服务 - 简化版本用于调试
async function connectToBackend() {
  console.log('Starting backend connection...');
  
  try {
    // 开发模式下直接使用本地服务器配置，跳过对话框
    if (isDev) {
      const serverConfig = CONFIG.DEV_SERVER;
      backendUrl = `${serverConfig.protocol}://${serverConfig.host}:${serverConfig.port}`;
      console.log(`Using development server: ${backendUrl}`);
      
      // 测试连接
      console.log('Testing backend connection...');
      const isConnected = await testBackendConnection(backendUrl);
      
      if (!isConnected) {
        console.error('Backend connection failed');
        const retry = await dialog.showMessageBox(mainWindow, {
          type: 'error',
          title: 'Connection Failed',
          message: 'Failed to connect to the backend server',
          detail: `Could not connect to ${backendUrl}. Please ensure the API server is running on port 8001.`,
          buttons: ['Retry', 'Continue Anyway', 'Quit'],
          defaultId: 0
        });
        
        if (retry.response === 0) {
          connectToBackend();
          return;
        } else if (retry.response === 2) {
          app.quit();
          return;
        }
        // Continue anyway (response === 1)
      }
      
      // 加载前端页面
      console.log('Loading main page...');
      mainWindow.loadURL(backendUrl);
      
    } else {
      // 生产模式显示配置对话框
      const serverConfig = await showServerConfigDialog();
      if (!serverConfig) {
        app.quit();
        return;
      }

      backendUrl = `${serverConfig.protocol}://${serverConfig.host}:${serverConfig.port}`;
      console.log(`Connecting to backend server at: ${backendUrl}`);
      
      // 测试连接
      const isConnected = await testBackendConnection(backendUrl);
      if (!isConnected) {
        const retry = await dialog.showMessageBox(mainWindow, {
          type: 'error',
          title: 'Connection Failed',
          message: 'Failed to connect to the backend server',
          detail: `Could not connect to ${backendUrl}. Please check if the server is running.`,
          buttons: ['Retry', 'Quit'],
          defaultId: 0
        });
        
        if (retry.response === 0) {
          connectToBackend();
        } else {
          app.quit();
        }
        return;
      }
      
      // 连接成功，加载前端页面
      mainWindow.loadURL(backendUrl);
    }
    
  } catch (error) {
    console.error('Error connecting to backend server:', error);
    dialog.showErrorBox('Connection Error', `Failed to connect to backend server: ${error.message}`);
    app.quit();
  }
}

// 显示服务器配置对话框
async function showServerConfigDialog() {
  return new Promise((resolve) => {
    const configWindow = new BrowserWindow({
      width: 500,
      height: 400,
      modal: true,
      parent: mainWindow,
      resizable: false,
      webPreferences: {
        nodeIntegration: true,
        contextIsolation: false
      },
      title: 'Server Configuration'
    });

    // 创建配置页面内容
    const configHtml = `
    <!DOCTYPE html>
    <html>
    <head>
      <title>Server Configuration</title>
      <style>
        body {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          padding: 30px;
          background: #f5f5f5;
          margin: 0;
        }
        .container {
          background: white;
          padding: 30px;
          border-radius: 10px;
          box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h2 {
          color: #333;
          margin-bottom: 20px;
          text-align: center;
        }
        .form-group {
          margin-bottom: 15px;
        }
        label {
          display: block;
          margin-bottom: 5px;
          font-weight: 500;
          color: #555;
        }
        input, select {
          width: 100%;
          padding: 10px;
          border: 1px solid #ddd;
          border-radius: 5px;
          font-size: 14px;
          box-sizing: border-box;
        }
        .button-group {
          display: flex;
          gap: 10px;
          margin-top: 20px;
        }
        button {
          flex: 1;
          padding: 12px;
          border: none;
          border-radius: 5px;
          font-size: 14px;
          cursor: pointer;
          font-weight: 500;
        }
        .btn-primary {
          background: #007AFF;
          color: white;
        }
        .btn-secondary {
          background: #8E8E93;
          color: white;
        }
        .btn-primary:hover {
          background: #0056CC;
        }
        .btn-secondary:hover {
          background: #6D6D70;
        }
        .preset-buttons {
          display: flex;
          gap: 10px;
          margin-bottom: 20px;
        }
        .preset-btn {
          padding: 8px 16px;
          border: 1px solid #007AFF;
          background: white;
          color: #007AFF;
          border-radius: 5px;
          cursor: pointer;
          font-size: 12px;
        }
        .preset-btn:hover {
          background: #007AFF;
          color: white;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <h2>InfiniSST Server Configuration</h2>
        
        <div class="preset-buttons">
          <button class="preset-btn" onclick="setPreset('local')">Local Development</button>
          <button class="preset-btn" onclick="setPreset('remote')">Remote Server</button>
          <button class="preset-btn" onclick="setPreset('ngrok')">Ngrok Tunnel</button>
        </div>
        
        <form id="configForm">
          <div class="form-group">
            <label for="protocol">Protocol:</label>
            <select id="protocol">
              <option value="http">HTTP</option>
              <option value="https">HTTPS</option>
            </select>
          </div>
          
          <div class="form-group">
            <label for="host">Host:</label>
            <input type="text" id="host" value="${CONFIG.REMOTE_SERVER.host}" placeholder="e.g., your-server.com or 192.168.1.100">
          </div>
          
          <div class="form-group">
            <label for="port">Port:</label>
            <input type="number" id="port" value="${CONFIG.REMOTE_SERVER.port}" min="1" max="65535">
          </div>
          
          <div class="button-group">
            <button type="button" class="btn-secondary" onclick="cancel()">Cancel</button>
            <button type="submit" class="btn-primary">Connect</button>
          </div>
        </form>
      </div>
      
      <script>
        const { ipcRenderer } = require('electron');
        
        function setPreset(type) {
          const protocolEl = document.getElementById('protocol');
          const hostEl = document.getElementById('host');
          const portEl = document.getElementById('port');
          
          if (type === 'local') {
            protocolEl.value = 'http';
            hostEl.value = 'localhost';
            portEl.value = '8001';
          } else if (type === 'remote') {
            protocolEl.value = 'http';
            hostEl.value = '';
            portEl.value = '8001';
            hostEl.focus();
          } else if (type === 'ngrok') {
            protocolEl.value = 'https';
            hostEl.value = 'infinisst.ngrok.app';
            portEl.value = '443';
          }
        }
        
        function cancel() {
          ipcRenderer.send('config-result', null);
        }
        
        document.getElementById('configForm').addEventListener('submit', (e) => {
          e.preventDefault();
          const config = {
            protocol: document.getElementById('protocol').value,
            host: document.getElementById('host').value,
            port: parseInt(document.getElementById('port').value)
          };
          ipcRenderer.send('config-result', config);
        });
        
        // 设置默认值
        document.getElementById('protocol').value = '${CONFIG.REMOTE_SERVER.protocol}';
      </script>
    </body>
    </html>
    `;

    configWindow.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(configHtml));

    ipcMain.once('config-result', (event, config) => {
      configWindow.close();
      resolve(config);
    });

    configWindow.on('closed', () => {
      resolve(null);
    });
  });
}

// 请求麦克风权限 (macOS)
async function requestMicrophonePermission() {
  if (process.platform === 'darwin') {
    try {
      console.log('Requesting microphone permission on macOS...');
      const microphonePermission = await systemPreferences.askForMediaAccess('microphone');
      console.log('Microphone permission granted:', microphonePermission);
      
      if (!microphonePermission) {
        console.warn('Microphone permission denied by user');
        const result = await dialog.showMessageBox(mainWindow, {
          type: 'warning',
          title: 'Microphone Permission Required',
          message: 'InfiniSST needs microphone access to provide real-time translation.',
          detail: 'Please grant microphone permission in System Preferences > Security & Privacy > Privacy > Microphone, then restart the application.',
          buttons: ['Open System Preferences', 'Continue Without Microphone', 'Quit'],
          defaultId: 0
        });
        
        if (result.response === 0) {
          // 打开系统偏好设置
          shell.openExternal('x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone');
        } else if (result.response === 2) {
          app.quit();
          return false;
        }
      }
      
      return microphonePermission;
    } catch (error) {
      console.error('Error requesting microphone permission:', error);
      return false;
    }
  } else {
    // 非macOS平台，假设权限已授予
    console.log('Non-macOS platform, assuming microphone permission is available');
    return true;
  }
}

// 测试后端连接
function testBackendConnection(url, maxAttempts = 3) {
  return new Promise((resolve) => {
    let attempts = 0;
    
    const testConnection = () => {
      const http = require('http');
      const https = require('https');
      const urlObj = new URL(url);
      const client = urlObj.protocol === 'https:' ? https : http;
      
      const req = client.get(url, (res) => {
        resolve(true);
      });
      
      req.on('error', () => {
        attempts++;
        if (attempts >= maxAttempts) {
          resolve(false);
        } else {
          setTimeout(testConnection, 1000);
        }
      });
      
      req.setTimeout(5000, () => {
        req.destroy();
        attempts++;
        if (attempts >= maxAttempts) {
          resolve(false);
        } else {
          setTimeout(testConnection, 1000);
        }
      });
    };
    
    testConnection();
  });
}

// 创建应用菜单
function createMenu() {
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'Open Audio/Video File...',
          accelerator: 'CmdOrCtrl+O',
          click: () => {
            if (mainWindow) {
              mainWindow.webContents.send('menu-open-file');
            }
          }
        },
        { type: 'separator' },
        {
          label: 'Debug Microphone',
          click: () => {
            if (mainWindow && backendUrl) {
              mainWindow.loadURL(`${backendUrl}/debug-microphone.html`);
            }
          }
        },
        {
          label: 'Back to Main App',
          click: () => {
            if (mainWindow && backendUrl) {
              mainWindow.loadURL(backendUrl);
            }
          }
        },
        { type: 'separator' },
        {
          label: 'Quit',
          accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
          click: () => {
            app.quit();
          }
        }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' }
      ]
    },
    {
      label: 'Translation',
      submenu: [
        {
          label: 'Show Translation Window',
          accelerator: 'CmdOrCtrl+T',
          click: () => {
            createTranslationWindow();
          }
        },
        {
          label: 'Hide Translation Window',
          accelerator: 'CmdOrCtrl+Shift+T',
          click: () => {
            if (translationWindow) {
              translationWindow.hide();
            }
          }
        },
        {
          label: 'Close Translation Window',
          click: () => {
            if (translationWindow) {
              translationWindow.close();
              translationWindow = null;
            }
          }
        }
      ]
    },
    {
      label: 'Window',
      submenu: [
        { role: 'minimize' },
        { role: 'close' }
      ]
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'About InfiniSST',
          click: () => {
            dialog.showMessageBox(mainWindow, {
              type: 'info',
              title: 'About InfiniSST',
              message: 'InfiniSST Translation Desktop',
              detail: 'Simultaneous end-to-end speech translation powered by LLM\nVersion 1.0.0'
            });
          }
        },
        {
          label: 'Learn More',
          click: () => {
            shell.openExternal('https://github.com/your-repo/InfiniSST');
          }
        }
      ]
    }
  ];

  // macOS 特殊处理
  if (process.platform === 'darwin') {
    template.unshift({
      label: app.getName(),
      submenu: [
        { role: 'about' },
        { type: 'separator' },
        { role: 'services' },
        { type: 'separator' },
        { role: 'hide' },
        { role: 'hideOthers' },
        { role: 'unhide' },
        { type: 'separator' },
        { role: 'quit' }
      ]
    });

    // Window menu
    template[4].submenu = [
      { role: 'close' },
      { role: 'minimize' },
      { role: 'zoom' },
      { type: 'separator' },
      { role: 'front' }
    ];
  }

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

// 应用准备就绪
app.whenReady().then(async () => {
  // 在创建窗口之前请求麦克风权限
  await requestMicrophonePermission();
  
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// 所有窗口关闭时
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// IPC 处理
ipcMain.handle('get-backend-url', () => {
  return backendUrl;
});

// 检查麦克风权限状态
ipcMain.handle('check-microphone-permission', async () => {
  if (process.platform === 'darwin') {
    try {
      const status = systemPreferences.getMediaAccessStatus('microphone');
      console.log('Current microphone permission status:', status);
      return status;
    } catch (error) {
      console.error('Error checking microphone permission:', error);
      return 'unknown';
    }
  } else {
    return 'granted'; // 非macOS平台假设已授权
  }
});

// 请求麦克风权限
ipcMain.handle('request-microphone-permission', async () => {
  return await requestMicrophonePermission();
});

ipcMain.handle('show-open-dialog', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Audio Files', extensions: ['mp3', 'wav', 'flac', 'm4a', 'aac'] },
      { name: 'Video Files', extensions: ['mp4', 'avi', 'mov', 'mkv', 'webm'] },
      { name: 'All Files', extensions: ['*'] }
    ]
  });
  return result;
});

// 创建翻译窗口
function createTranslationWindow() {
  if (translationWindow) {
    translationWindow.focus();
    return;
  }

  translationWindow = new BrowserWindow({
    width: 600,
    height: 300,
    minWidth: 400,
    minHeight: 200,
    maxWidth: 1000,
    maxHeight: 600,
    alwaysOnTop: true,
    skipTaskbar: true,
    resizable: true,
    frame: true,
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    title: 'InfiniSST Translation',
    show: false
  });

  // 加载翻译窗口页面
  translationWindow.loadFile(path.join(__dirname, 'translation-window.html'));

  // 窗口准备好后显示
  translationWindow.once('ready-to-show', () => {
    translationWindow.show();
    
    // 设置窗口位置到右下角
    const { screen } = require('electron');
    const primaryDisplay = screen.getPrimaryDisplay();
    const { width, height } = primaryDisplay.workAreaSize;
    
    translationWindow.setPosition(
      width - translationWindow.getBounds().width - 20,
      height - translationWindow.getBounds().height - 20
    );
    
    // 窗口显示后发送初始状态
    setTimeout(() => {
      console.log('Sending initial status to translation window');
      translationWindow.webContents.send('status-update', {
        text: '翻译窗口已准备就绪，请加载模型开始翻译',
        type: 'ready'
      });
    }, 200);
  });

  // 窗口关闭时的处理
  translationWindow.on('closed', () => {
    translationWindow = null;
  });

  // 防止窗口被最小化时失去置顶状态
  translationWindow.on('minimize', () => {
    translationWindow.restore();
  });
}

// IPC 处理器
ipcMain.handle('show-translation-window', () => {
  createTranslationWindow();
});

ipcMain.handle('hide-translation-window', () => {
  if (translationWindow) {
    translationWindow.hide();
  }
});

ipcMain.handle('close-translation-window', () => {
  if (translationWindow) {
    translationWindow.close();
    translationWindow = null;
  }
});

ipcMain.handle('update-translation', (event, translationData) => {
  console.log('Main process received translation update:', translationData?.text?.substring(0, 50) + '...');
  if (translationWindow && translationWindow.webContents) {
    // 确保窗口已完全加载
    if (translationWindow.webContents.isLoading()) {
      console.log('Translation window still loading, waiting...');
      translationWindow.webContents.once('did-finish-load', () => {
        console.log('Translation window loaded, sending translation update');
        translationWindow.webContents.send('translation-update', translationData);
      });
    } else {
      console.log('Sending translation update to translation window');
      translationWindow.webContents.send('translation-update', translationData);
    }
  } else {
    console.log('Translation window not available for translation update');
  }
});

ipcMain.handle('update-translation-status', (event, statusData) => {
  console.log('Main process received status update:', statusData);
  if (translationWindow && translationWindow.webContents) {
    // 确保窗口已完全加载
    if (translationWindow.webContents.isLoading()) {
      console.log('Translation window still loading, waiting...');
      translationWindow.webContents.once('did-finish-load', () => {
        console.log('Translation window loaded, sending status update');
        translationWindow.webContents.send('status-update', statusData);
      });
    } else {
      console.log('Sending status update to translation window');
      translationWindow.webContents.send('status-update', statusData);
    }
  } else {
    console.log('Translation window not available for status update');
  }
});

ipcMain.handle('reset-translation-from-window', (event) => {
  // 向主窗口发送重置翻译的请求
  if (mainWindow && mainWindow.webContents) {
    mainWindow.webContents.executeJavaScript(`
      if (typeof resetTranslationFromElectron === 'function') {
        resetTranslationFromElectron();
      }
    `);
  }
  
  // 向翻译窗口发送重置确认
  if (translationWindow && translationWindow.webContents) {
    translationWindow.webContents.send('reset-translation');
  }
});

// 处理未捕获的异常
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  dialog.showErrorBox('Unexpected Error', error.message);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
}); 