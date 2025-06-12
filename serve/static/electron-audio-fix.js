// Electron专用音频处理模块 - 安全版本
// 避免音频反馈循环和内存泄漏

class ElectronAudioProcessor {
    constructor() {
        this.mediaRecorder = null;
        this.micStream = null;
        this.ws = null;
        this.isProcessing = false;
        this.audioChunks = [];
        
        // 调试计数器
        this.dataAvailableCount = 0;
        this.chunksSentCount = 0;
        this.lastLogTime = 0;
        this.errorCount = 0;
        
        // 安全配置
        this.config = {
            mimeType: 'audio/webm;codecs=opus',
            audioBitsPerSecond: 64000, // 降低比特率减少内存压力
            timeslice: 200, // 增加时间片减少事件频率
            targetSampleRate: 16000,
            baseChunkSize: 960 * 16,
            logInterval: 2000, // 每2秒打印一次状态
            maxErrorCount: 5, // 最大错误次数
            maxChunkSize: 1024 * 1024 // 最大块大小 1MB
        };
        
        console.log('🎵 ElectronAudioProcessor (安全版本) created');
    }
    
    async initializeAudio(micStream, websocket) {
        try {
            console.log('🚀 Starting Electron audio processor (安全版本) initialization...');
            
            this.micStream = micStream;
            this.ws = websocket;
            this.errorCount = 0;
            
            // 检查MediaRecorder支持
            if (!MediaRecorder.isTypeSupported(this.config.mimeType)) {
                console.warn('⚠️ WebM/Opus not supported, trying fallback...');
                this.config.mimeType = 'audio/webm';
                if (!MediaRecorder.isTypeSupported(this.config.mimeType)) {
                    this.config.mimeType = '';
                    console.log('📝 Using default MediaRecorder format');
                }
            }
            
            console.log(`🎵 MediaRecorder format: ${this.config.mimeType || 'default'}`);
            
            // 创建MediaRecorder with 安全配置
            const options = {
                audioBitsPerSecond: this.config.audioBitsPerSecond
            };
            
            if (this.config.mimeType) {
                options.mimeType = this.config.mimeType;
            }
            
            this.mediaRecorder = new MediaRecorder(micStream, options);
            console.log(`🔧 MediaRecorder created with safe options:`, options);
            
            // 设置事件处理器
            this.setupMediaRecorderEvents();
            
            // 开始录制
            this.mediaRecorder.start(this.config.timeslice);
            console.log(`✅ MediaRecorder started with timeslice: ${this.config.timeslice}ms`);
            
            this.isProcessing = true;
            console.log('🎉 Electron audio processor (安全版本) initialized successfully!');
            
            // 开始状态监控
            this.startStatusMonitoring();
            
            return true;
        } catch (error) {
            console.error('❌ Error initializing Electron audio processor:', error);
            this.cleanup();
            throw error;
        }
    }
    
    setupMediaRecorderEvents() {
        console.log('🎵 Setting up MediaRecorder event handlers...');
        
        this.mediaRecorder.ondataavailable = (event) => {
            try {
                this.dataAvailableCount++;
                
                // 每2秒打印一次详细状态
                const now = Date.now();
                if (now - this.lastLogTime >= this.config.logInterval) {
                    console.log(`🎵 MediaRecorder dataavailable event #${this.dataAvailableCount}`);
                    console.log(`📡 WebSocket state: ${this.ws ? this.ws.readyState : 'null'} (1=OPEN)`);
                    console.log(`🔧 MediaRecorder state: ${this.mediaRecorder ? this.mediaRecorder.state : 'null'}`);
                    console.log(`❌ Error count: ${this.errorCount}/${this.config.maxErrorCount}`);
                    this.lastLogTime = now;
                }
                
                if (event.data && event.data.size > 0) {
                    // 检查数据大小是否合理
                    if (event.data.size > this.config.maxChunkSize) {
                        console.warn(`⚠️ Audio chunk too large: ${event.data.size} bytes, skipping`);
                        return;
                    }
                    
                    console.log(`🎤 Got audio data: size=${event.data.size} bytes, type=${event.data.type}`);
                    
                    // 异步处理音频数据，避免阻塞
                    this.processAudioBlobSafely(event.data);
                } else {
                    console.log(`🔇 No audio data in event (size: ${event.data ? event.data.size : 'null'})`);
                }
            } catch (error) {
                this.handleError('ondataavailable', error);
            }
        };
        
        this.mediaRecorder.onstart = () => {
            console.log('🎬 MediaRecorder started');
            this.errorCount = 0; // 重置错误计数
        };
        
        this.mediaRecorder.onstop = () => {
            console.log('🛑 MediaRecorder stopped');
        };
        
        this.mediaRecorder.onerror = (event) => {
            this.handleError('MediaRecorder', event.error);
        };
        
        this.mediaRecorder.onpause = () => {
            console.log('⏸️ MediaRecorder paused');
        };
        
        this.mediaRecorder.onresume = () => {
            console.log('▶️ MediaRecorder resumed');
        };
        
        console.log('✅ MediaRecorder event handlers configured');
    }
    
    async processAudioBlobSafely(blob) {
        try {
            if (!this.isProcessing) {
                console.log('🛑 Processing stopped, ignoring audio blob');
                return;
            }
            
            console.log(`🔧 Processing audio blob: size=${blob.size}, type=${blob.type}`);
            
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                // 使用setTimeout确保异步处理，避免阻塞主线程
                setTimeout(async () => {
                    try {
                        const arrayBuffer = await blob.arrayBuffer();
                        console.log(`📦 Converted blob to ArrayBuffer: ${arrayBuffer.byteLength} bytes`);
                        
                        // 发送音频数据
                        if (this.ws && this.ws.readyState === WebSocket.OPEN && this.isProcessing) {
                            this.ws.send(arrayBuffer);
                            this.chunksSentCount++;
                            console.log(`✅ Sent audio blob #${this.chunksSentCount} (${arrayBuffer.byteLength} bytes)`);
                        } else {
                            console.warn(`⚠️ Cannot send audio: WebSocket state changed to ${this.ws ? this.ws.readyState : 'null'}`);
                        }
                    } catch (error) {
                        this.handleError('processAudioBlob async', error);
                    }
                }, 0);
            } else {
                console.warn(`⚠️ Cannot send audio: WebSocket state is ${this.ws ? this.ws.readyState : 'null'}`);
            }
            
        } catch (error) {
            this.handleError('processAudioBlob', error);
        }
    }
    
    handleError(source, error) {
        this.errorCount++;
        console.error(`❌ Error in ${source} (#${this.errorCount}):`, error);
        
        if (this.errorCount >= this.config.maxErrorCount) {
            console.error(`🚨 Too many errors (${this.errorCount}), stopping audio processor`);
            this.stop();
            
            // 通知上层应用
            if (typeof window !== 'undefined' && window.updateStatus) {
                window.updateStatus('Audio processing failed due to too many errors', 'error');
            }
        }
    }
    
    startStatusMonitoring() {
        console.log('📊 Starting status monitoring...');
        
        const monitor = () => {
            if (!this.isProcessing) {
                console.log('📊 Status monitoring stopped');
                return;
            }
            
            try {
                console.log(`📊 Status Report:`);
                console.log(`   - Data available events: ${this.dataAvailableCount}`);
                console.log(`   - Chunks sent: ${this.chunksSentCount}`);
                console.log(`   - Errors: ${this.errorCount}/${this.config.maxErrorCount}`);
                console.log(`   - WebSocket state: ${this.ws ? this.ws.readyState : 'null'}`);
                console.log(`   - MediaRecorder state: ${this.mediaRecorder ? this.mediaRecorder.state : 'null'}`);
                console.log(`   - Processing: ${this.isProcessing}`);
                
                // 检查是否有问题
                if (this.dataAvailableCount === 0) {
                    console.warn('⚠️ WARNING: No dataavailable events received! MediaRecorder may not be working.');
                }
                
                if (this.ws && this.ws.readyState !== WebSocket.OPEN) {
                    console.warn(`⚠️ WARNING: WebSocket not open (state: ${this.ws.readyState})`);
                }
                
                if (this.mediaRecorder && this.mediaRecorder.state !== 'recording') {
                    console.warn(`⚠️ WARNING: MediaRecorder not recording (state: ${this.mediaRecorder.state})`);
                }
                
                // 检查错误率
                if (this.errorCount > 0) {
                    console.warn(`⚠️ WARNING: ${this.errorCount} errors occurred`);
                }
                
                // 5秒后再次检查
                if (this.isProcessing) {
                    setTimeout(monitor, 5000);
                }
            } catch (error) {
                console.error('❌ Error in status monitoring:', error);
            }
        };
        
        // 3秒后开始监控，给初始化更多时间
        setTimeout(monitor, 3000);
    }
    
    stop() {
        console.log('🛑 Stopping Electron audio processor...');
        
        this.isProcessing = false;
        
        // 打印最终统计
        console.log(`📊 Final Statistics:`);
        console.log(`   - Total dataavailable events: ${this.dataAvailableCount}`);
        console.log(`   - Total chunks sent: ${this.chunksSentCount}`);
        console.log(`   - Total errors: ${this.errorCount}`);
        
        this.cleanup();
        
        console.log('✅ Electron audio processor stopped');
    }
    
    cleanup() {
        console.log('🧹 Cleaning up audio processor resources...');
        
        // 停止MediaRecorder
        if (this.mediaRecorder) {
            try {
                if (this.mediaRecorder.state === 'recording') {
                    this.mediaRecorder.stop();
                    console.log('✅ MediaRecorder stopped');
                }
            } catch (e) {
                console.error('❌ Error stopping MediaRecorder:', e);
            }
            this.mediaRecorder = null;
        }
        
        // 清理资源
        this.audioChunks = [];
        this.ws = null;
        this.micStream = null;
        
        console.log('✅ Audio processor cleanup completed');
    }
}

// 导出给全局使用
window.ElectronAudioProcessor = ElectronAudioProcessor;
console.log('🎵 ElectronAudioProcessor (安全版本) class loaded and available globally'); 