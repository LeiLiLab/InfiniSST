import { useState, useEffect, useCallback } from 'react';
import { TranslationSession, TranslationState, AudioSource, SubtitleStyle } from '../types';
import { websocketService } from '../services/websocket';
import { audioService } from '../services/audio';
import { apiService } from '../services/api';

export const useTranslation = () => {
  const [session, setSession] = useState<TranslationSession | null>(null);
  const [translationState, setTranslationState] = useState<TranslationState>({
    text: '',
    isPaused: true,
    isInitialized: false,
  });
  const [status, setStatus] = useState<string>('');
  const [error, setError] = useState<string>('');
  const [volume, setVolume] = useState<number>(0);
  const [subtitleStyle, setSubtitleStyle] = useState<SubtitleStyle>({
    fontSize: 24,
    fontColor: '#ffffff',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    position: 'bottom',
    opacity: 1,
  });

  // Initialize translation session
  const initializeSession = useCallback(async (
    agentType: string,
    languagePair: string,
    latencyMultiplier: number,
    clientId: string
  ) => {
    try {
      setStatus('Initializing session...');
      const newSession = await apiService.initializeSession(
        agentType,
        languagePair,
        latencyMultiplier,
        clientId
      );
      setSession(newSession);
      setStatus('Session initialized');
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to initialize session');
    }
  }, []);

  // Connect WebSocket and start translation
  const startTranslation = useCallback(async (source: AudioSource) => {
    if (!session) {
      setError('No active session');
      return;
    }

    try {
      // Connect WebSocket
      await websocketService.connect(session.sessionId);
      
      // Set up WebSocket callbacks
      websocketService.setOnMessageCallback((text) => {
        setTranslationState(prev => ({ ...prev, text }));
      });
      
      websocketService.setOnErrorCallback((error) => {
        setError(error);
      });
      
      websocketService.setOnStatusCallback((status) => {
        setStatus(status);
      });

      // Initialize audio processing
      await audioService.initializeAudio(source);
      
      // Set up audio callbacks
      audioService.setOnAudioDataCallback((data) => {
        websocketService.sendAudioData(data);
      });
      
      audioService.setOnVolumeCallback((volume) => {
        setVolume(volume);
      });

      // Start processing
      audioService.startProcessing();
      setTranslationState(prev => ({ ...prev, isPaused: false, isInitialized: true }));
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to start translation');
    }
  }, [session]);

  // Pause/resume translation
  const togglePause = useCallback(() => {
    if (translationState.isInitialized) {
      if (translationState.isPaused) {
        audioService.startProcessing();
      } else {
        audioService.stopProcessing();
      }
      setTranslationState(prev => ({ ...prev, isPaused: !prev.isPaused }));
    }
  }, [translationState.isInitialized, translationState.isPaused]);

  // Reset translation
  const resetTranslation = useCallback(async () => {
    if (!session) return;

    try {
      await apiService.resetTranslation(session.sessionId);
      setTranslationState({
        text: '',
        isPaused: true,
        isInitialized: false,
      });
      setStatus('Translation reset');
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to reset translation');
    }
  }, [session]);

  // Update latency
  const updateLatency = useCallback(async (latencyMultiplier: number) => {
    if (!session) return;

    try {
      await apiService.updateLatency(session.sessionId, latencyMultiplier);
      setSession(prev => prev ? { ...prev, latencyMultiplier } : null);
      setStatus('Latency updated');
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to update latency');
    }
  }, [session]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      websocketService.disconnect();
      audioService.cleanup();
    };
  }, []);

  return {
    session,
    translationState,
    status,
    error,
    volume,
    subtitleStyle,
    setSubtitleStyle,
    initializeSession,
    startTranslation,
    togglePause,
    resetTranslation,
    updateLatency,
  };
}; 