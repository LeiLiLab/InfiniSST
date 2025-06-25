class ElectronAudioProcessor {
    constructor() {
        this.audioContext = null;
        this.workletNode = null;
        this.micStream = null;
        this.sourceNode = null;
        this.ws = null;
        this.isProcessing = false;
        this.resampledBuffer = new Float32Array();
        this.chunksSentCount = 0;
        this.errorCount = 0;
        this.lastIdleResetTime = 0;

        this.config = {
            targetSampleRate: 16000,
            baseChunkSize: 960 * 16,
            maxErrorCount: 5,
            idleResetThrottleMs: 1000
        };

        console.log('🎵 ElectronAudioProcessor with AudioWorklet created');
    }

    async initializeAudio(audioSource, websocket, sourceType = 'microphone') {
        try {
            console.log('🚀 Initializing AudioWorklet-based processor for', sourceType, '...');
            if (sourceType === 'microphone') {
                this.micStream = audioSource;
            } else if (sourceType === 'media') {
                this.mediaElement = audioSource;
            }
            this.audioSource = audioSource;
            this.sourceType = sourceType;
            this.ws = websocket;
            this.isProcessing = true;
            this.errorCount = 0;
            this.chunksSentCount = 0;
            this.resampledBuffer = new Float32Array();

            // 创建AudioContext并检查状态
            this.audioContext = new AudioContext({ sampleRate: 48000 });
            console.log('🎙️ AudioContext created, state:', this.audioContext.state);
            
            // 确保AudioContext处于运行状态
            if (this.audioContext.state === 'suspended') {
                console.log('▶️ Resuming suspended AudioContext...');
                await this.audioContext.resume();
            }
            
            // 在Electron环境中使用正确的模块路径
            let audioProcessorUrl;
            let urlBuildMethod = 'unknown';
            try {
                // 尝试多种URL构建方式
                if (window.location && window.location.origin) {
                    audioProcessorUrl = window.location.origin + '/static/audio-processor.js';
                    urlBuildMethod = 'window.location.origin';
                } else {
                    // 备用方案
                    const protocol = window.location.protocol || 'http:';
                    const host = window.location.host || 'localhost';
                    audioProcessorUrl = `${protocol}//${host}/static/audio-processor.js`;
                    urlBuildMethod = 'manual construction';
                }
            } catch (urlError) {
                console.warn('⚠️ Error building URL, using relative path:', urlError);
                audioProcessorUrl = '/static/audio-processor.js';
                urlBuildMethod = 'relative path fallback';
            }
            
            console.log('📁 Loading AudioWorklet module:', {
                url: audioProcessorUrl,
                method: urlBuildMethod,
                currentLocation: window.location?.href,
                isElectron: typeof window !== 'undefined' && !!window.electronAPI
            });
            
            // 添加重试机制加载AudioWorklet模块
            let retryCount = 0;
            const maxRetries = 5; // 增加重试次数
            
            while (retryCount < maxRetries) {
                try {
                    await this.audioContext.audioWorklet.addModule(audioProcessorUrl);
                    console.log('✅ AudioWorklet module loaded successfully on attempt', retryCount + 1);
                    break;
                } catch (moduleError) {
                    retryCount++;
                    console.warn(`⚠️ AudioWorklet module load attempt ${retryCount}/${maxRetries} failed:`, {
                        error: moduleError.message,
                        name: moduleError.name,
                        url: audioProcessorUrl,
                        audioContextState: this.audioContext.state
                    });
                    
                    if (retryCount >= maxRetries) {
                        console.error('❌ All AudioWorklet attempts failed, will try ScriptProcessor fallback');
                        throw new Error(`Failed to load AudioWorklet module after ${maxRetries} attempts: ${moduleError.message}`);
                    }
                    
                    // 等待递增的时间后重试
                    const waitTime = 200 * retryCount;
                    console.log(`⏳ Waiting ${waitTime}ms before retry...`);
                    await new Promise(resolve => setTimeout(resolve, waitTime));
                }
            }
            
            console.log('🔧 Creating AudioWorkletNode...');
            this.workletNode = new AudioWorkletNode(this.audioContext, 'pcm-processor');

            // Create appropriate source node based on source type
            if (this.sourceType === 'microphone') {
                this.sourceNode = this.audioContext.createMediaStreamSource(this.micStream);
                console.log('📱 Created MediaStreamSource for microphone');
            } else if (this.sourceType === 'media') {
                this.sourceNode = this.audioContext.createMediaElementSource(this.mediaElement);
                console.log('🎵 Created MediaElementSource for media file');
            }
            
            this.sourceNode.connect(this.workletNode);
            this.workletNode.connect(this.audioContext.destination);
            
            console.log('🔍 Audio connection debug:', {
                sourceType: this.sourceType,
                hasSourceNode: !!this.sourceNode,
                hasWorkletNode: !!this.workletNode,
                hasDestination: !!this.audioContext.destination
            });
            
            // 添加直接连接，确保用户能听到音频（特别是对于媒体文件）
            if (this.sourceType === 'media') {
                this.sourceNode.connect(this.audioContext.destination);
                console.log('🔊 Added direct audio connection for media playback');
            } else {
                console.log('🎵 Source type is not media, skipping direct connection. Source type:', this.sourceType);
            }

            const resampleRatio = this.config.targetSampleRate / this.audioContext.sampleRate;
            // 使用全局延迟倍数（与传统浏览器方式保持一致）
            const currentLatencyMultiplier = (typeof window !== 'undefined' && window.currentLatencyMultiplier) ? window.currentLatencyMultiplier : 2;
            const targetChunkSize = this.config.baseChunkSize * currentLatencyMultiplier;
            console.log(`🎯 AudioWorklet using latency multiplier: ${currentLatencyMultiplier}x, target chunk size: ${targetChunkSize}`);

            // 添加本地缓冲区来模拟ScriptProcessor的行为
            let localAudioBuffer = new Float32Array();
            const SCRIPT_PROCESSOR_BUFFER_SIZE = 4096; // 与传统浏览器方式保持一致

            this.workletNode.port.onmessage = (event) => {
                if (!this.isProcessing || !event.data) return;

                const input = event.data;
                
                // 音频活动检测（与传统浏览器方式保持一致）
                let hasSound = false;
                let volumeSum = 0;
                for (let i = 0; i < input.length; i++) {
                    const volume = Math.abs(input[i]);
                    volumeSum += volume;
                    if (volume > 0.01) {
                        hasSound = true;
                    }
                }
                
                // 如果检测到声音，重置idle timer（使用节流机制）
                if (hasSound && typeof window !== 'undefined' && window.resetIdleTimer) {
                    const now = Date.now();
                    if (now - this.lastIdleResetTime >= this.config.idleResetThrottleMs) {
                        window.resetIdleTimer();
                        this.lastIdleResetTime = now;
                        console.log('🔄 Idle timer reset (throttled)');
                    }
                }
                
                // 更新音量指示器（主要用于麦克风模式）
                if (this.sourceType === 'microphone' && typeof document !== 'undefined') {
                    const volumeLevel = document.getElementById('volumeLevel');
                    if (volumeLevel) {
                        const averageVolume = volumeSum / input.length;
                        const volumePercent = Math.min(100, Math.round(averageVolume * 1000));
                        volumeLevel.style.width = volumePercent + '%';
                    }
                }
                
                // 将新数据添加到本地缓冲区
                const newLocalBuffer = new Float32Array(localAudioBuffer.length + input.length);
                newLocalBuffer.set(localAudioBuffer);
                newLocalBuffer.set(input, localAudioBuffer.length);
                localAudioBuffer = newLocalBuffer;

                // 当本地缓冲区达到ScriptProcessor的大小时才处理
                while (localAudioBuffer.length >= SCRIPT_PROCESSOR_BUFFER_SIZE) {
                    const processingChunk = localAudioBuffer.slice(0, SCRIPT_PROCESSOR_BUFFER_SIZE);
                    localAudioBuffer = localAudioBuffer.slice(SCRIPT_PROCESSOR_BUFFER_SIZE);

                    // 保持与传统浏览器方式一致，处理所有音频数据

                    // 重采样处理（与传统浏览器方式完全一致）
                    const resampledLength = Math.floor(processingChunk.length * resampleRatio);
                    const resampledChunk = new Float32Array(resampledLength);
                    for (let i = 0; i < resampledLength; i++) {
                        const originalIndex = Math.floor(i / resampleRatio);
                        resampledChunk[i] = processingChunk[originalIndex];
                    }

                    const newBuffer = new Float32Array(this.resampledBuffer.length + resampledChunk.length);
                    newBuffer.set(this.resampledBuffer);
                    newBuffer.set(resampledChunk, this.resampledBuffer.length);
                    this.resampledBuffer = newBuffer;

                    while (this.resampledBuffer.length >= targetChunkSize) {
                        const chunk = this.resampledBuffer.slice(0, targetChunkSize);
                        try {
                            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                                this.ws.send(chunk.buffer);
                                this.chunksSentCount++;
                                if (this.chunksSentCount % 5 === 0) {
                                    console.log(`✅ Sent chunk #${this.chunksSentCount} (${chunk.byteLength} bytes) [BufferMode]`);
                                }
                            }
                        } catch (err) {
                            this.handleError('send_chunk', err);
                        }
                        this.resampledBuffer = this.resampledBuffer.slice(targetChunkSize);
                    }
                }
            };

            console.log('🎉 AudioWorkletNode processor initialized!');
            return true;
        } catch (error) {
            console.error('❌ Error initializing AudioWorklet processor:', error);
            console.log('🔄 Attempting fallback to ScriptProcessor...');
            
            try {
                return await this.initializeWithScriptProcessor(this.audioSource, websocket, this.sourceType);
            } catch (fallbackError) {
                console.error('❌ Fallback ScriptProcessor also failed:', fallbackError);
                this.cleanup();
                throw new Error(`Both AudioWorklet and ScriptProcessor failed. AudioWorklet: ${error.message}, ScriptProcessor: ${fallbackError.message}`);
            }
        }
    }

    async initializeWithScriptProcessor(audioSource, websocket, sourceType = 'microphone') {
        console.log('🔧 Initializing with ScriptProcessor fallback for', sourceType, '...');
        console.warn('⚠️ Using deprecated ScriptProcessor because AudioWorklet failed to load');
        
        try {
            // 清理之前的AudioContext
            if (this.audioContext) {
                try {
                    await this.audioContext.close();
                } catch (e) {
                    console.warn('⚠️ Error closing previous AudioContext:', e);
                }
            }
            
            if (sourceType === 'microphone') {
                this.micStream = audioSource;
            } else if (sourceType === 'media') {
                this.mediaElement = audioSource;
            }
            this.audioSource = audioSource;
            this.sourceType = sourceType;
            this.ws = websocket;
            this.isProcessing = true;
            this.errorCount = 0;
            this.chunksSentCount = 0;
            this.resampledBuffer = new Float32Array();

            // 创建新的AudioContext
            this.audioContext = new AudioContext({ sampleRate: 48000 });
            console.log('🎙️ AudioContext created for ScriptProcessor, state:', this.audioContext.state);
            
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }

            // 使用ScriptProcessor代替AudioWorklet
            const scriptProcessor = this.audioContext.createScriptProcessor(4096, 1, 1);
            this.workletNode = scriptProcessor; // 保持接口一致性
            
            // Create appropriate source node based on source type
            if (this.sourceType === 'microphone') {
                this.sourceNode = this.audioContext.createMediaStreamSource(this.micStream);
                console.log('📱 Created MediaStreamSource for microphone [ScriptProcessor]');
            } else if (this.sourceType === 'media') {
                this.sourceNode = this.audioContext.createMediaElementSource(this.mediaElement);
                console.log('🎵 Created MediaElementSource for media file [ScriptProcessor]');
            }
            
            this.sourceNode.connect(scriptProcessor);
            scriptProcessor.connect(this.audioContext.destination);
            
            console.log('🔍 Audio connection debug [ScriptProcessor]:', {
                sourceType: this.sourceType,
                hasSourceNode: !!this.sourceNode,
                hasScriptProcessor: !!scriptProcessor,
                hasDestination: !!this.audioContext.destination
            });
            
            // 添加直接连接，确保用户能听到音频（特别是对于媒体文件）
            if (this.sourceType === 'media') {
                this.sourceNode.connect(this.audioContext.destination);
                console.log('🔊 Added direct audio connection for media playback [ScriptProcessor]');
            } else {
                console.log('🎵 Source type is not media, skipping direct connection [ScriptProcessor]. Source type:', this.sourceType);
            }

            const resampleRatio = this.config.targetSampleRate / this.audioContext.sampleRate;
            // 使用全局延迟倍数（与传统浏览器方式保持一致）
            const currentLatencyMultiplier = (typeof window !== 'undefined' && window.currentLatencyMultiplier) ? window.currentLatencyMultiplier : 2;
            const targetChunkSize = this.config.baseChunkSize * currentLatencyMultiplier;
            console.log(`🎯 ScriptProcessor using latency multiplier: ${currentLatencyMultiplier}x, target chunk size: ${targetChunkSize}`);

            // ScriptProcessor使用onaudioprocess而不是port.onmessage
            scriptProcessor.onaudioprocess = (event) => {
                if (!this.isProcessing) return;

                const inputData = event.inputBuffer.getChannelData(0);
                
                // 音频活动检测（与传统浏览器方式保持一致）
                let hasSound = false;
                let volumeSum = 0;
                for (let i = 0; i < inputData.length; i++) {
                    const volume = Math.abs(inputData[i]);
                    volumeSum += volume;
                    if (volume > 0.01) {
                        hasSound = true;
                    }
                }
                
                // 如果检测到声音，重置idle timer（使用节流机制）
                if (hasSound && typeof window !== 'undefined' && window.resetIdleTimer) {
                    const now = Date.now();
                    if (now - this.lastIdleResetTime >= this.config.idleResetThrottleMs) {
                        window.resetIdleTimer();
                        this.lastIdleResetTime = now;
                        console.log('🔄 Idle timer reset (throttled)');
                    }
                }
                
                // 更新音量指示器（主要用于麦克风模式）
                if (this.sourceType === 'microphone' && typeof document !== 'undefined') {
                    const volumeLevel = document.getElementById('volumeLevel');
                    if (volumeLevel) {
                        const averageVolume = volumeSum / inputData.length;
                        const volumePercent = Math.min(100, Math.round(averageVolume * 1000));
                        volumeLevel.style.width = volumePercent + '%';
                    }
                }
                
                // 重采样逻辑与AudioWorklet相同
                const resampledLength = Math.floor(inputData.length * resampleRatio);
                const resampledChunk = new Float32Array(resampledLength);
                for (let i = 0; i < resampledLength; i++) {
                    const originalIndex = Math.floor(i / resampleRatio);
                    resampledChunk[i] = inputData[originalIndex];
                }

                const newBuffer = new Float32Array(this.resampledBuffer.length + resampledChunk.length);
                newBuffer.set(this.resampledBuffer);
                newBuffer.set(resampledChunk, this.resampledBuffer.length);
                this.resampledBuffer = newBuffer;

                while (this.resampledBuffer.length >= targetChunkSize) {
                    const chunk = this.resampledBuffer.slice(0, targetChunkSize);
                    try {
                        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                            this.ws.send(chunk.buffer);
                            this.chunksSentCount++;
                            if (this.chunksSentCount % 5 === 0) {
                                console.log(`✅ Sent chunk #${this.chunksSentCount} (${chunk.byteLength} bytes) via ScriptProcessor`);
                            }
                        }
                    } catch (err) {
                        this.handleError('send_chunk', err);
                    }
                    this.resampledBuffer = this.resampledBuffer.slice(targetChunkSize);
                }
            };

            console.log('🎉 ScriptProcessor fallback initialized successfully!');
            return true;
        } catch (error) {
            console.error('❌ Error initializing ScriptProcessor fallback:', error);
            this.cleanup();
            throw error;
        }
    }

    handleError(source, error) {
        this.errorCount++;
        console.error(`❌ Error in ${source} (#${this.errorCount}):`, error);
        if (this.errorCount >= this.config.maxErrorCount) {
            console.error(`🚨 Too many errors (${this.errorCount}), stopping audio processor`);
            this.stop();
            if (typeof window !== 'undefined' && window.updateStatus) {
                window.updateStatus('Audio processing failed due to too many errors', 'error');
            }
        }
    }

    stop() {
        console.log('🛑 Stopping Electron audio processor...');
        this.isProcessing = false;

        if (this.resampledBuffer && this.resampledBuffer.length > 0) {
            const currentLatencyMultiplier = (typeof window !== 'undefined' && window.currentLatencyMultiplier) ? window.currentLatencyMultiplier : 2;
            const finalChunkSize = this.config.baseChunkSize * currentLatencyMultiplier;
            const finalChunk = new Float32Array(finalChunkSize);
            finalChunk.set(this.resampledBuffer.slice(0, finalChunkSize));
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(finalChunk.buffer);
                console.log(`✅ Sent final partial chunk (${finalChunk.byteLength} bytes)`);
            }
        }

        this.cleanup();
        console.log('✅ Electron audio processor stopped');
    }

    cleanup() {
        console.log('🧹 Cleaning up audio processor resources...');
        this.ws = null;
        this.micStream = null;
        this.mediaElement = null;
        this.audioSource = null;
        this.sourceType = null;

        if (this.sourceNode) {
            try { this.sourceNode.disconnect(); } catch (e) {}
            this.sourceNode = null;
        }
        if (this.workletNode) {
            try { this.workletNode.disconnect(); } catch (e) {}
            this.workletNode = null;
        }
        if (this.audioContext) {
            try { this.audioContext.close(); } catch (e) {}
            this.audioContext = null;
        }

        console.log('✅ Audio processor cleanup completed');
    }
}

window.ElectronAudioProcessor = ElectronAudioProcessor;
console.log('🎵 ElectronAudioProcessor with AudioWorklet loaded globally');