class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    console.log('🎵 PCMProcessor AudioWorklet created');
  }

  process(inputs, outputs, parameters) {
    try {
      const input = inputs[0];
      if (input && input.length > 0) {
        const channelData = input[0];
        if (channelData && channelData.length > 0) {
          this.port.postMessage(channelData);
        }
      }
      return true;
    } catch (error) {
      console.error('❌ Error in AudioWorklet process:', error);
      return false;
    }
  }
}

console.log('📦 Registering PCMProcessor AudioWorklet...');
registerProcessor('pcm-processor', PCMProcessor);
console.log('✅ PCMProcessor AudioWorklet registered successfully');