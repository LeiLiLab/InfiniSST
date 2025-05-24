export interface TranslationSession {
  sessionId: string;
  agentType: string;
  languagePair: string;
  latencyMultiplier: number;
  isReady: boolean;
}

export interface TranslationState {
  text: string;
  isPaused: boolean;
  isInitialized: boolean;
  translatedText?: string;
  subtitleStyle: SubtitleStyle;
}

export interface AudioSource {
  type: 'mic';
  stream: MediaStream;
}

export type Position = 'top' | 'bottom';

export interface SubtitleStyle {
  fontSize: number;
  fontFamily: string;
  fontColor: string;
  color: string;
  backgroundColor: string;
  padding: string;
  borderRadius: string;
  subtitlePosition: Position;
  opacity: number;
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