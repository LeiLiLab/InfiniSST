#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

console.log('=== InfiniSST Electron Debug Launcher ===\n');

// 检查必需的文件
const requiredFiles = [
  'electron/main.js',
  'electron/preload.js', 
  'electron/translation-window.html',
  'package.json'
];

console.log('Checking required files...');
let allFilesExist = true;

for (const file of requiredFiles) {
  if (fs.existsSync(file)) {
    console.log(`✓ ${file}`);
  } else {
    console.log(`✗ ${file} - NOT FOUND`);
    allFilesExist = false;
  }
}

if (!allFilesExist) {
  console.log('\n❌ Some required files are missing. Please check your project structure.');
  process.exit(1);
}

// 检查node_modules
if (!fs.existsSync('node_modules')) {
  console.log('\n⚠️  node_modules not found. Please run: npm install');
  process.exit(1);
}

console.log('\n✅ All required files found');

// 检查API服务器是否运行
const http = require('http');

function checkApiServer() {
  return new Promise((resolve) => {
    const req = http.request({
      hostname: 'localhost',
      port: 8001,
      path: '/',
      timeout: 3000
    }, (res) => {
      console.log('✅ API server is running on port 8001');
      resolve(true);
    });

    req.on('error', (error) => {
      console.log('⚠️  API server not detected on port 8001');
      console.log('   You can still run Electron, but some features may not work');
      resolve(false);
    });

    req.on('timeout', () => {
      console.log('⚠️  API server check timed out');
      resolve(false);
    });

    req.end();
  });
}

async function startElectron() {
  console.log('\nChecking API server...');
  await checkApiServer();
  
  console.log('\n🚀 Starting Electron in development mode...');
  console.log('📝 Check the terminal output for any errors\n');
  
  // 设置环境变量
  const env = {
    ...process.env,
    ELECTRON_IS_DEV: 'true',
    NODE_ENV: 'development'
  };
  
  // 启动Electron
  const electronProcess = spawn('npx', ['electron', '.'], {
    stdio: 'inherit',
    env: env,
    shell: true
  });
  
  electronProcess.on('close', (code) => {
    console.log(`\nElectron process exited with code ${code}`);
  });
  
  electronProcess.on('error', (error) => {
    console.error('Failed to start Electron:', error);
  });
}

startElectron(); 