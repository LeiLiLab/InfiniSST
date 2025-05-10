export interface TranslationSession {
  sessionId: string;
  agentType: string;
  languagePair: string;
  latencyMultiplier: number;
  isReady: boolean;
}

export interface TranslationState {
  text: string;
  translatedText: string;
  subtitleStyle: SubtitleStyle;
}

export interface AudioSource {
  type: 'mic';
  stream: MediaStream;
}

export interface SubtitleStyle {
  fontSize: string;
  fontFamily: string;
  color: string;
  backgroundColor: string;
  padding: string;
  borderRadius: string;
}

export interface TranslationHistory {
  id: string;
  timestamp: number;
  sourceText: string;
  targetText: string;
  languagePair: string;
  duration: number;
}

export interface TranslationService {
  initializeSession: (targetLanguage: string) => Promise<string>;
  sendAudioData: (sessionId: string, data: ArrayBuffer, callback: (text: string, translated: string) => void) => Promise<void>;
  endSession: (sessionId: string) => Promise<void>;
}

export interface AudioService {
  initializeAudio: (source: AudioSource) => Promise<void>;
  startProcessing: () => void;
  stopProcessing: () => void;
  setOnAudioDataCallback: (callback: (data: ArrayBuffer) => void) => void;
  setOnVolumeCallback: (callback: (volume: number) => void) => void;
  cleanup: () => Promise<void>;
} 