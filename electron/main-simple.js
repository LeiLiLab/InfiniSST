const { app, BrowserWindow, Menu, dialog, shell, ipcMain } = require('electron');
const path = require('path');

console.log('=== Simple Electron Main Process Starting ===');

let mainWindow;
let translationWindow;

// 获取服务器URL（本地或远程）
function getServerUrl() {
  const remoteUrl = process.env.REMOTE_SERVER_URL;
  if (remoteUrl) {
    console.log('Using remote server URL:', remoteUrl);
    return remoteUrl;
  } else {
    console.log('Using local server URL: http://localhost:8001');
    return 'http://localhost:8001';
  }
}

function createWindow() {
  console.log('Creating main window...');
  
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
      webSecurity: false
    },
    show: false
  });

  console.log('Main window created');

  // 使用动态URL（本地或远程）
  const mainUrl = getServerUrl();
  console.log(`Loading: ${mainUrl}`);
  
  mainWindow.loadURL(mainUrl).catch(error => {
    console.error('Failed to load URL:', error);
    // 如果无法连接到服务器，显示错误页面
    mainWindow.loadFile(path.join(__dirname, 'error.html'));
  });

  mainWindow.once('ready-to-show', () => {
    console.log('Main window ready to show');
    mainWindow.show();
    
    // 只在开发模式下打开开发者工具
    if (process.env.ELECTRON_IS_DEV === 'true') {
      mainWindow.webContents.openDevTools();
    }
  });

  mainWindow.on('closed', () => {
    console.log('Main window closed');
    mainWindow = null;
  });

  // 监听页面事件
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
}

// 创建翻译窗口
function createTranslationWindow() {
  if (translationWindow) {
    console.log('Translation window already exists, focusing...');
    translationWindow.focus();
    return translationWindow;
  }

  console.log('Creating translation window...');

  translationWindow = new BrowserWindow({
    width: 600,
    height: 300,
    alwaysOnTop: true,
    resizable: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    title: 'InfiniSST Translation'
  });

  translationWindow.loadFile(path.join(__dirname, 'translation-window.html'));

  translationWindow.once('ready-to-show', () => {
    translationWindow.show();
    
    // 只在开发模式且非远程连接时开启开发者工具
    if (process.env.ELECTRON_IS_DEV === 'true' && !process.env.REMOTE_SERVER_URL) {
      console.log('Opening DevTools for translation window');
      translationWindow.webContents.openDevTools();
    } else if (process.env.REMOTE_SERVER_URL) {
      console.log('Remote connection detected, skipping DevTools for translation window');
    }
    
    // 窗口显示后发送初始状态
    setTimeout(() => {
      console.log('Sending initial status to translation window');
      if (translationWindow && translationWindow.webContents) {
        const serverUrl = getServerUrl();
        const isRemote = serverUrl.includes('ngrok') || serverUrl.includes('https://');
        const statusText = isRemote ? 
          '远程连接已建立，请加载模型开始翻译' : 
          '翻译窗口已准备就绪，请加载模型开始翻译';
          
        translationWindow.webContents.send('status-update', {
          text: statusText,
          type: 'ready'
        });
      }
    }, 1000);
  });

  translationWindow.on('closed', () => {
    translationWindow = null;
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

// 应用事件
app.whenReady().then(() => {
  createWindow();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

console.log('Electron app setup complete'); 