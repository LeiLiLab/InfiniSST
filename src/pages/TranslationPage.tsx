import React, { useState, useCallback, useEffect, ChangeEvent } from 'react';
import {
  Container,
  Box,
  Typography,
  Paper,
  Grid,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  Drawer,
  useTheme,
  useMediaQuery,
  CircularProgress,
  Tooltip,
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  Stop,
  Settings,
  Mic,
  MicOff,
  VolumeUp,
} from '@mui/icons-material';
import { useTranslation } from '../hooks/useTranslation';
import { Subtitle } from '../components/Subtitle';
import { AudioVisualizer } from '../components/AudioVisualizer';
import { SubtitleSettings } from '../components/SubtitleSettings';
import { AudioSource, SubtitleStyle } from '../types';
import { audioService } from '../services/audio';
import { translationService } from '../services/translation';
import { styled } from '@mui/material/styles';
import { TranslationDisplay } from '../components/TranslationDisplay';

const LANGUAGE_PAIRS = [
  { value: 'English -> Chinese', label: 'English → Chinese' },
  { value: 'English -> German', label: 'English → German' },
  { value: 'English -> Spanish', label: 'English → Spanish' },
];

const LATENCY_OPTIONS = [
  { value: 1, label: '1x (Fastest)' },
  { value: 2, label: '2x (Default)' },
  { value: 3, label: '3x (More Accurate)' },
  { value: 4, label: '4x (Most Accurate)' },
];

export const TranslationPage: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const [languagePair, setLanguagePair] = useState(LANGUAGE_PAIRS[0].value);
  const [latencyMultiplier, setLatencyMultiplier] = useState(2);
  const [isMicActive, setIsMicActive] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isTranslating, setIsTranslating] = useState<boolean>(false);
  const [volume, setVolume] = useState<number>(0);
  const [currentText, setCurrentText] = useState<string>('');
  const [translatedText, setTranslatedText] = useState<string>('');
  const [targetLanguage, setTargetLanguage] = useState<string>('zh');
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [subtitleStyle, setSubtitleStyle] = useState<SubtitleStyle>({
    fontSize: '24px',
    fontFamily: 'Arial',
    color: '#ffffff',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    padding: '10px',
    borderRadius: '5px',
  });
  const [translationText, setTranslationText] = useState<string>('');
  const [showTranslation, setShowTranslation] = useState<boolean>(true);

  const {
    session,
    translationState,
    status,
    setSubtitleStyle: setTranslationSubtitleStyle,
    initializeSession,
    startTranslation,
    togglePause,
    resetTranslation,
    updateLatency,
  } = useTranslation();

  const handleMicToggle = async () => {
    if (!isMicActive) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        setIsMicActive(true);
        if (!session) {
          // Initialize session if not already done
          const clientId = Math.random().toString(36).substring(2, 15);
          await initializeSession('InfiniSST', languagePair, latencyMultiplier, clientId);
        }
        await startTranslation({ type: 'mic', stream });
      } catch (error) {
        console.error('Error accessing microphone:', error);
      }
    } else {
      setIsMicActive(false);
      resetTranslation();
    }
  };

  const handleLatencyChange = async (value: number) => {
    setLatencyMultiplier(value);
    if (session) {
      await updateLatency(value);
    }
  };

  const handleVolumeChange = useCallback((newVolume: number) => {
    setVolume(newVolume);
  }, []);

  const handleLanguageChange = useCallback((language: string) => {
    setTargetLanguage(language);
  }, []);

  const handleSubtitleStyleChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setSubtitleStyle((prev: SubtitleStyle) => ({
      ...prev,
      [name]: value
    }));
  }, []);

  const handleTranslationUpdate = useCallback((text: string, translated: string) => {
    setCurrentText(text);
    setTranslatedText(translated);
  }, []);

  const toggleRecording = useCallback(async () => {
    try {
      if (!isRecording) {
        setError(null);
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const source: AudioSource = { type: 'mic', stream };
        
        await audioService.initializeAudio(source);
        audioService.setOnAudioDataCallback(async (data: ArrayBuffer) => {
          if (!sessionId) {
            const newSessionId = await translationService.initializeSession(targetLanguage);
            setSessionId(newSessionId);
          }
          await translationService.sendAudioData(sessionId!, data, handleTranslationUpdate);
        });
        
        audioService.setOnVolumeCallback(handleVolumeChange);
        audioService.startProcessing();
        setIsRecording(true);
        setIsTranslating(true);
      } else {
        audioService.stopProcessing();
        await audioService.cleanup();
        if (sessionId) {
          await translationService.endSession(sessionId);
          setSessionId(null);
        }
        setIsRecording(false);
        setIsTranslating(false);
        setCurrentText('');
        setTranslatedText('');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred';
      setError(errorMessage);
      setIsRecording(false);
      setIsTranslating(false);
    }
  }, [isRecording, sessionId, targetLanguage, handleTranslationUpdate, handleVolumeChange]);

  useEffect(() => {
    return () => {
      if (isRecording) {
        audioService.stopProcessing();
        audioService.cleanup();
        if (sessionId) {
          translationService.endSession(sessionId);
        }
      }
    };
  }, [isRecording, sessionId]);

  const handleTranslation = (text: string, translated: string) => {
    setTranslationText(translated);
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Simultaneous Speech Translation
        </Typography>

        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <Paper sx={{ p: 2, mb: 2 }}>
              <Grid container spacing={2} alignItems="center">
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth>
                    <InputLabel>Language Pair</InputLabel>
                    <Select
                      value={languagePair}
                      label="Language Pair"
                      onChange={(e) => setLanguagePair(e.target.value)}
                      disabled={!!session}
                    >
                      {LANGUAGE_PAIRS.map((pair) => (
                        <MenuItem key={pair.value} value={pair.value}>
                          {pair.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth>
                    <InputLabel>Latency</InputLabel>
                    <Select
                      value={latencyMultiplier}
                      label="Latency"
                      onChange={(e) => handleLatencyChange(e.target.value as number)}
                      disabled={!session}
                    >
                      {LATENCY_OPTIONS.map((option) => (
                        <MenuItem key={option.value} value={option.value}>
                          {option.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
            </Paper>

            <Paper sx={{ p: 2, mb: 2 }}>
              <Button
                variant="outlined"
                startIcon={isMicActive ? <MicOff /> : <Mic />}
                onClick={handleMicToggle}
                fullWidth
                disabled={!!session && !isMicActive}
              >
                {isMicActive ? 'Stop Microphone' : 'Start Microphone'}
              </Button>
            </Paper>

            <Paper sx={{ p: 2, mb: 2 }}>
              <AudioVisualizer
                volume={volume}
                isActive={translationState.isInitialized && !translationState.isPaused}
              />
            </Paper>

            <Paper sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" sx={{ flexGrow: 1 }}>
                  Translation Output
                </Typography>
                <IconButton
                  onClick={() => setIsSettingsOpen(true)}
                  disabled={!translationState.isInitialized}
                >
                  <Settings />
                </IconButton>
              </Box>

              <Box sx={{ minHeight: '100px', p: 2, bgcolor: 'grey.100', borderRadius: 1 }}>
                <Typography>{translationState.text || 'Translation will appear here...'}</Typography>
              </Box>

              <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                <Button
                  variant="contained"
                  startIcon={translationState.isPaused ? <PlayArrow /> : <Pause />}
                  onClick={togglePause}
                  disabled={!translationState.isInitialized}
                >
                  {translationState.isPaused ? 'Resume' : 'Pause'}
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<Stop />}
                  onClick={resetTranslation}
                  disabled={!translationState.isInitialized}
                >
                  Reset
                </Button>
              </Box>
            </Paper>
          </Grid>

          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Status
              </Typography>
              <Typography color={error ? 'error' : 'textPrimary'}>
                {error || status || 'Ready'}
              </Typography>
            </Paper>
          </Grid>
        </Grid>
      </Box>

      <Drawer
        anchor={isMobile ? 'bottom' : 'right'}
        open={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
      >
        <Box sx={{ width: isMobile ? 'auto' : 400 }}>
          <SubtitleSettings
            style={translationState.subtitleStyle}
            onChange={handleSubtitleStyleChange}
          />
        </Box>
      </Drawer>

      <Subtitle
        text={translationState.text}
        style={translationState.subtitleStyle}
      />

      {showTranslation && (
        <TranslationDisplay
          text={translationText}
          onClose={() => setShowTranslation(false)}
          initialPosition={{ x: window.innerWidth - 320, y: 20 }}
        />
      )}

      <Button
        variant="outlined"
        onClick={() => setShowTranslation(!showTranslation)}
        sx={{ mt: 2 }}
      >
        {showTranslation ? 'Hide Translation' : 'Show Translation'}
      </Button>
    </Container>
  );
}; 