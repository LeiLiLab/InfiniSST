import { TranslationService } from '../types';

// Configuration object that can be overridden at runtime
const config = {
  baseUrl: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000',  // Use environment variable with fallback
  defaultLanguagePair: process.env.REACT_APP_DEFAULT_LANGUAGE_PAIR || 'English -> Chinese',
  defaultLatencyMultiplier: Number(process.env.REACT_APP_DEFAULT_LATENCY_MULTIPLIER) || 2,
  enableDebugLogging: process.env.NODE_ENV === 'development',  // Enable debug logging in development mode
};

// Function to update configuration at runtime
export const updateConfig = (newConfig: Partial<typeof config>) => {
  Object.assign(config, newConfig);
  if (config.enableDebugLogging) {
    console.log('Translation service config updated:', config);
  }
};

// Function to get current configuration
export const getConfig = () => ({ ...config });

class TranslationServiceImpl implements TranslationService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = config.baseUrl;
    if (config.enableDebugLogging) {
      console.log('Translation service initialized with config:', config);
    }
  }

  public async initializeSession(targetLanguage: string): Promise<string> {
    try {
      if (config.enableDebugLogging) {
        console.log('Initializing translation session with target language:', targetLanguage);
      }

      const response = await fetch(`${this.baseUrl}/init`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          agent_type: 'InfiniSST',
          language_pair: config.defaultLanguagePair,
          latency_multiplier: config.defaultLatencyMultiplier,
          client_id: Math.random().toString(36).substring(2, 15),  // Generate a random client ID
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to initialize session: ${response.statusText}`);
      }

      const data = await response.json();
      if (config.enableDebugLogging) {
        console.log('Session initialized:', data);
      }
      return data.session_id;
    } catch (error) {
      console.error('Error initializing translation session:', error);
      throw error;
    }
  }

  public async sendAudioData(
    sessionId: string,
    data: ArrayBuffer,
    callback: (text: string, translated: string) => void
  ): Promise<void> {
    try {
      if (config.enableDebugLogging) {
        console.log('Sending audio data for session:', sessionId);
      }

      const response = await fetch(`${this.baseUrl}/wss/${sessionId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/octet-stream',
        },
        body: data,
      });

      if (!response.ok) {
        throw new Error(`Failed to send audio data: ${response.statusText}`);
      }

      const result = await response.json();
      if (config.enableDebugLogging) {
        console.log('Received translation:', result);
      }
      callback(result.text, result.translatedText);
    } catch (error) {
      console.error('Error sending audio data:', error);
      throw error;
    }
  }

  public async endSession(sessionId: string): Promise<void> {
    try {
      if (config.enableDebugLogging) {
        console.log('Ending session:', sessionId);
      }

      const response = await fetch(`${this.baseUrl}/delete_session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ session_id: sessionId }),
      });

      if (!response.ok) {
        throw new Error(`Failed to end session: ${response.statusText}`);
      }

      if (config.enableDebugLogging) {
        console.log('Session ended successfully');
      }
    } catch (error) {
      console.error('Error ending translation session:', error);
      throw error;
    }
  }
}

export const translationService = new TranslationServiceImpl(); 