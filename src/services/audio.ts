import { AudioSource } from '../types';

class AudioService {
  private audioContext: AudioContext | null = null;
  private audioSource: MediaStreamAudioSourceNode | null = null;
  private processor: ScriptProcessorNode | null = null;
  private resampledBuffer: Float32Array = new Float32Array();
  private isProcessing: boolean = false;
  private onAudioDataCallback: ((data: ArrayBuffer) => void) | null = null;
  private onVolumeCallback: ((volume: number) => void) | null = null;

  private readonly targetSampleRate = 16000;
  private readonly baseChunkSize = 960 * 16;
  private readonly segmentSize = 4096;

  constructor() {
    this.processAudio = this.processAudio.bind(this);
  }

  public async initializeAudio(source: AudioSource): Promise<void> {
    try {
      this.audioContext = new AudioContext();
      console.log('AudioContext created with sample rate:', this.audioContext.sampleRate);

      if (source.type === 'mic') {
        this.audioSource = this.audioContext.createMediaStreamSource(source.stream);
      }

      if (this.audioSource) {
        this.processor = this.audioContext.createScriptProcessor(this.segmentSize, 1, 1);
        this.processor.onaudioprocess = this.processAudio;
        this.audioSource.connect(this.processor);
        this.processor.connect(this.audioContext.destination);
      }
    } catch (error) {
      console.error('Error initializing audio:', error);
      throw error;
    }
  }

  public startProcessing(): void {
    this.isProcessing = true;
  }

  public stopProcessing(): void {
    this.isProcessing = false;
  }

  public setOnAudioDataCallback(callback: (data: ArrayBuffer) => void): void {
    this.onAudioDataCallback = callback;
  }

  public setOnVolumeCallback(callback: (volume: number) => void): void {
    this.onVolumeCallback = callback;
  }

  public async cleanup(): Promise<void> {
    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }

    if (this.audioSource) {
      this.audioSource.disconnect();
      this.audioSource = null;
    }

    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
    }

    this.resampledBuffer = new Float32Array();
    this.isProcessing = false;
  }

  private processAudio(e: AudioProcessingEvent): void {
    if (!this.isProcessing || !this.audioContext) return;

    const inputData = e.inputBuffer.getChannelData(0);
    
    // Calculate volume level
    let volumeSum = 0;
    for (let i = 0; i < inputData.length; i++) {
      volumeSum += Math.abs(inputData[i]);
    }
    const averageVolume = volumeSum / inputData.length;
    this.onVolumeCallback?.(averageVolume);

    // Resample audio data
    const resampleRatio = this.targetSampleRate / this.audioContext.sampleRate;
    const resampledLength = Math.floor(inputData.length * resampleRatio);
    const resampledChunk = new Float32Array(resampledLength);

    for (let i = 0; i < resampledLength; i++) {
      const originalIndex = Math.floor(i / resampleRatio);
      resampledChunk[i] = inputData[originalIndex];
    }

    // Add resampled data to buffer
    const newBuffer = new Float32Array(this.resampledBuffer.length + resampledChunk.length);
    newBuffer.set(this.resampledBuffer);
    newBuffer.set(resampledChunk, this.resampledBuffer.length);
    this.resampledBuffer = newBuffer;

    // Send data when buffer is full enough
    const targetChunkSize = this.baseChunkSize;
    while (this.resampledBuffer.length >= targetChunkSize) {
      const chunk = this.resampledBuffer.slice(0, targetChunkSize);
      this.onAudioDataCallback?.(chunk.buffer);
      this.resampledBuffer = this.resampledBuffer.slice(targetChunkSize);
    }
  }
}

export const audioService = new AudioService(); 