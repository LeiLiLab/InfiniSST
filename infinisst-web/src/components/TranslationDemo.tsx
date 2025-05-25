import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  Box, 
  TextField, 
  Button, 
  Typography, 
  Paper, 
  Container, 
  Grid,
  IconButton,
  Slider,
  Tooltip,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  useTheme,
  ThemeProvider,
  createTheme,
  alpha,
  SelectChangeEvent,
  styled,
  Menu,
  Snackbar,
  Alert
} from '@mui/material';
import MicIcon from '@mui/icons-material/Mic';
import StopIcon from '@mui/icons-material/Stop';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import YouTubeIcon from '@mui/icons-material/YouTube';
import DragIndicatorIcon from '@mui/icons-material/DragIndicator';
import LanguageIcon from '@mui/icons-material/Language';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import TuneIcon from '@mui/icons-material/Tune';
import { motion, AnimatePresence } from 'framer-motion';
import apiService, { Translation, InitSessionResponse, QueueStatusResponse } from '../services/api';
import './TranslationDemo.css';

// Create a custom theme for the application
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#2196f3',
    },
    secondary: {
      main: '#f50057',
    },
    background: {
      default: '#0a1929',
      paper: '#132f4c',
    },
    text: {
      primary: '#ffffff',
      secondary: '#b0bec5',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
      letterSpacing: '-0.02em',
    },
    h5: {
      fontWeight: 600,
      letterSpacing: '-0.01em',
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          padding: '10px 20px',
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0 4px 20px 0 rgba(0,0,0,0.12)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
  },
});

// Styled components
const StyledUploadButton = styled(Button)<{component?: React.ElementType}>(({ theme }) => ({
  background: `linear-gradient(45deg, ${theme.palette.primary.main} 30%, ${theme.palette.primary.light} 90%)`,
  height: '56px',
  transition: 'all 0.3s ease-in-out',
  '&:hover': {
    transform: 'translateY(-2px)',
  },
}));

const StyledRecordButton = styled(Button)(({ theme }) => ({
  background: `linear-gradient(45deg, ${theme.palette.secondary.main} 30%, ${theme.palette.secondary.light} 90%)`,
  height: '56px',
  transition: 'all 0.3s ease-in-out',
  '&:hover': {
    transform: 'translateY(-2px)',
  },
}));

const StyledYouTubeField = styled(TextField)(({ theme }) => ({
  '& .MuiOutlinedInput-root': {
    height: '56px',
    transition: 'all 0.3s ease-in-out',
    backdropFilter: 'blur(10px)',
    background: alpha(theme.palette.background.paper, 0.8),
    '&:hover': {
      '& .MuiOutlinedInput-notchedOutline': {
        borderColor: theme.palette.primary.main,
      },
    },
    '&.Mui-focused': {
      '& .MuiOutlinedInput-notchedOutline': {
        borderColor: theme.palette.primary.main,
        borderWidth: 2,
      },
    },
  },
  '& .MuiInputLabel-outlined': {
    transform: 'translate(14px, 18px) scale(1)',
    '&.MuiInputLabel-shrink': {
      transform: 'translate(14px, -6px) scale(0.75)',
    },
  },
}));

const StyledIconButton = styled(IconButton)(({ theme }) => ({
  backgroundColor: alpha(theme.palette.primary.main, 0.1),
  color: theme.palette.primary.main,
  transition: 'all 0.3s ease-in-out',
  '&:hover': {
    backgroundColor: alpha(theme.palette.primary.main, 0.2),
    transform: 'scale(1.1)',
  },
}));

const StyledMediaContainer = styled(Paper)(({ theme }) => ({
  overflow: 'hidden',
  position: 'relative',
  background: `linear-gradient(180deg, ${alpha(theme.palette.background.paper, 0.6)} 0%, ${alpha(theme.palette.background.default, 0.9)} 100%)`,
  backdropFilter: 'blur(10px)',
  boxShadow: '0 10px 30px rgba(0, 0, 0, 0.2)',
  border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
  height: '400px',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
}));

// Create an Input component specifically for file upload
const VisuallyHiddenInput = styled('input')({
  clip: 'rect(0 0 0 0)',
  clipPath: 'inset(50%)',
  height: 1,
  overflow: 'hidden',
  position: 'absolute',
  bottom: 0,
  left: 0,
  whiteSpace: 'nowrap',
  width: 1,
});

// Latency multiplier options
const LATENCY_OPTIONS = [
  { value: 1, label: '1x (Fastest)' },
  { value: 2, label: '2x (Default)' },
  { value: 3, label: '3x (More Accurate)' },
  { value: 4, label: '4x (Most Accurate)' }
];

const TranslationDemo: React.FC = () => {
  const [mediaUrl, setMediaUrl] = useState<string>('');
  const [youtubeUrl, setYoutubeUrl] = useState<string>('');
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [translations, setTranslations] = useState<Translation[]>([
    { time: 0, text: "Welcome to InfiniSST Translation Demo" }
  ]);
  const [currentTranslation, setCurrentTranslation] = useState<string>("Welcome to InfiniSST Translation Demo");
  const [barPosition, setBarPosition] = useState<{ top: number, left: number }>({ top: 0, left: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStartPos, setDragStartPos] = useState<{ x: number, y: number }>({ x: 0, y: 0 });
  const [targetLanguage, setTargetLanguage] = useState<string>('en');
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  
  // New state variables for new features
  const [latencyMultiplier, setLatencyMultiplier] = useState<number>(2);
  const [modelLoaded, setModelLoaded] = useState<boolean>(false);
  const [modelLoading, setModelLoading] = useState<boolean>(false);
  const [modelFeedback, setModelFeedback] = useState<{message: string, severity: 'success' | 'error' | 'info' | 'warning'} | null>(null);
  const [containerWidth, setContainerWidth] = useState<number>(0);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const mediaRef = useRef<HTMLVideoElement | HTMLAudioElement | null>(null);
  const translationBarRef = useRef<HTMLDivElement>(null);
  const mediaContainerRef = useRef<HTMLDivElement>(null);
  
  // State for managing the session
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [queuePosition, setQueuePosition] = useState<number>(0);
  const [isQueued, setIsQueued] = useState(false);
  // Ref to prevent duplicate session initialization
  const isInitializingRef = useRef(false);
  //check worker ready
  const workerReadyRef = useRef(false);

  // State for managing audio
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  // State for managing the translation
  const [translation, setTranslation] = useState<string>("");
  const [isTranslating, setIsTranslating] = useState(false);

  // References
  const wsRef = useRef<WebSocket | null>(null);
  const pingIntervalRef = useRef<number | null>(null);
  const queueCheckIntervalRef = useRef<number | null>(null);

  // Auto-load model when component mounts
  useEffect(() => {
    // Check for orphaned sessions and clean them up
    const pendingSessions = apiService.getPendingDeleteSessions();
    Object.keys(pendingSessions).forEach(async (orphanedSessionId) => {
      await apiService.deleteSession(orphanedSessionId);
      apiService.removePendingDeleteSession(orphanedSessionId);
    });

    // Initialize session
    initSession();

    // Cleanup on unmount
    return () => {
      cleanupSession();
    };
  }, []);

  // Function to load the model
  const loadModel = async () => {
    try {
      setModelLoading(true);
      setModelFeedback({
        message: 'Loading translation model...',
        severity: 'info'
      });

      // Simulate model loading with a delay (replace with actual API call)
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Successfully loaded
      setModelLoaded(true);
      setModelLoading(false);
      setModelFeedback({
        message: 'Translation model loaded successfully',
        severity: 'success'
      });

      // Clear success message after 3 seconds
      setTimeout(() => {
        setModelFeedback(null);
      }, 3000);

    } catch (error) {
      console.error('Error loading model:', error);
      setModelLoading(false);
      setModelFeedback({
        message: 'Error loading translation model',
        severity: 'error'
      });
    }
  };

  // Initialize WebSocket connection when session ID is available
  useEffect(() => {
    if (sessionId) {
      connectWebSocket();
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [sessionId]);

  // Initialize translation session with anti-reentry logic
  const initSession = async () => {
    if (isInitializingRef.current || sessionId) {
      console.log("Skipping duplicate initSession call.");
      return;
    }
    isInitializingRef.current = true;
    setIsConnecting(true);

    try {
      const response = await apiService.initSession(
        'InfiniSST',
        'English -> Chinese',
        latencyMultiplier
      );

      if (response.error) {
        console.error('Failed to initialize session:', response.error);
        setModelFeedback({
          message: `Failed to initialize session: ${response.error}`,
          severity: 'error'
        });
        return;
      }

      console.log('Session initialized:', response);
      setSessionId(response.session_id);
      apiService.storeSessionId(response.session_id);

      if (response.queued) {
        setIsQueued(true);
        setQueuePosition(response.queue_position);
        setModelFeedback({
          message: `Waiting in queue... Position: ${response.queue_position}`,
          severity: 'warning'
        });
        startQueueCheck(response.session_id);
      } else if (response.initializing) {
        setModelFeedback({
          message: 'Model is initializing, please wait...',
          severity: 'info'
        });
        setIsQueued(true);
        startQueueCheck(response.session_id);
      } else {
        setIsQueued(false);
        setModelLoaded(true);
        setModelFeedback({
          message: 'Model loaded successfully',
          severity: 'success'
        });
        startPingInterval(response.session_id);
      }
    } catch (error) {
      console.error('Error initializing session:', error);
      setModelFeedback({
        message: `Error initializing session: ${error instanceof Error ? error.message : String(error)}`,
        severity: 'error'
      });
    } finally {
      isInitializingRef.current = false;
      setIsConnecting(false);
    }
  };
  
  // Start checking queue status
  const startQueueCheck = (sessionId: string) => {
    // Clear any existing interval
    if (queueCheckIntervalRef.current) {
      window.clearInterval(queueCheckIntervalRef.current);
    }
    
    // Check queue status immediately
    checkQueueStatus(sessionId);
    
    // Set up interval to check every 2 seconds
    queueCheckIntervalRef.current = window.setInterval(() => {
      checkQueueStatus(sessionId);
    }, 2000);
  };
  
  // Check queue status
  const checkQueueStatus = async (sessionId: string) => {
    try {
      const status = await apiService.checkQueueStatus(sessionId);
      
      if (status.status === 'active') {
        // Session is now active
        setIsQueued(false);
        setQueuePosition(0);
        setModelLoaded(true);
        setModelFeedback({
          message: 'Model loaded successfully. Ready to translate.',
          severity: 'success'
        });
        
        // Stop checking queue status
        if (queueCheckIntervalRef.current) {
          window.clearInterval(queueCheckIntervalRef.current);
          queueCheckIntervalRef.current = null;
        }
        
        // Start ping interval
        startPingInterval(sessionId);
        
        // Connect WebSocket
        connectWebSocket();
      } else if (status.status === 'initializing') {
        setModelFeedback({
          message: 'Model is initializing, please wait...',
          severity: 'info'
        });
      } else if (status.status === 'queued') {
        setQueuePosition(status.queue_position);
        setModelFeedback({
          message: `Waiting in queue... Position: ${status.queue_position}`,
          severity: 'warning'
        });
      } else {
        // Session not found
        setModelFeedback({
          message: 'Error: Session not found',
          severity: 'error'
        });
        
        // Stop checking
        if (queueCheckIntervalRef.current) {
          window.clearInterval(queueCheckIntervalRef.current);
          queueCheckIntervalRef.current = null;
        }
      }
    } catch (error) {
      console.error('Error checking queue status:', error);
    }
  };
  
  // Connect to WebSocket
  const connectWebSocket = () => {
    if (!sessionId) return;
    
    const ws = apiService.createWebSocket(sessionId);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      setIsConnecting(false);
    };
    
    ws.onmessage = (event) => {
      const message = event.data;
      
      if (message.startsWith('INITIALIZING:')) {
        console.log('Worker initializing:', message);
        setModelFeedback({
          message: message.substring('INITIALIZING:'.length).trim(),
          severity: 'info'
        });
      } else if (message.startsWith('READY:')) {
        console.log('Worker ready:', message);
        workerReadyRef.current = true;
        setIsConnecting(false);
        setModelFeedback({
          message: 'Ready to translate',
          severity: 'success'
        });
      } else if (message.startsWith('ERROR:')) {
        console.error('Error from worker:', message);
        setModelFeedback({
          message: `Error: ${message.substring(6)}`,
          severity: 'error'
        });
      } else {
        // Translation output
        setTranslation(message);
        setCurrentTranslation(message);
      }
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
      
      // Try to reconnect after a delay if session is still active
      if (sessionId && modelLoaded && !isQueued) {
        setTimeout(() => {
          connectWebSocket();
        }, 3000);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      ws.close();
    };
    
    wsRef.current = ws;
  };
  
  // Start ping interval to keep session alive
  const startPingInterval = (id: string) => {
    // Clear any existing interval
    if (pingIntervalRef.current) {
      window.clearInterval(pingIntervalRef.current);
    }
    
    // Send ping every 5 seconds to keep session alive
    pingIntervalRef.current = window.setInterval(() => {
      apiService.sendPing(id).catch(error => {
        console.error('Error sending ping:', error);
      });
    }, 5000);
  };
  
  // Clean up session on component unmount
  const cleanupSession = () => {
    // Clear ping interval
    if (pingIntervalRef.current) {
      window.clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }
    
    // Clear queue check interval
    if (queueCheckIntervalRef.current) {
      window.clearInterval(queueCheckIntervalRef.current);
      queueCheckIntervalRef.current = null;
    }
    
    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    // Delete session
    if (sessionId) {
      // Store as pending delete in case the request fails
      apiService.storePendingDeleteSession(sessionId);
      
      apiService.deleteSession(sessionId).then(response => {
        if (response.success) {
          apiService.removePendingDeleteSession(sessionId);
        }
      }).catch(error => {
        console.error('Error deleting session:', error);
      });
      
      // Remove stored session ID
      apiService.removeStoredSessionId();
    }
  };
  
  // Upload and process audio file
  const uploadFile = async (file: File) => {
    if (!file || !isConnected || !wsRef.current || !workerReadyRef.current) {
      setModelFeedback({
        message: 'Please wait for the model to be ready',
        severity: 'warning'
      });
      return;
    }
    
    setIsUploading(true);
    setIsTranslating(true);
    
    try {
      // Read the file as ArrayBuffer
      const arrayBuffer = await file.arrayBuffer();
      
      // Create an AudioContext
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      
      // Decode the audio data
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      
      // Get the raw audio data (mono channel)
      const rawData = audioBuffer.getChannelData(0);
      
      // Resample to 16kHz if needed
      const sampleRate = audioBuffer.sampleRate;
      const targetSampleRate = 16000;
      
      let processedData: Float32Array;
      if (sampleRate !== targetSampleRate) {
        const resampleRatio = targetSampleRate / sampleRate;
        const resampledLength = Math.floor(rawData.length * resampleRatio);
        processedData = new Float32Array(resampledLength);
        
        for (let i = 0; i < resampledLength; i++) {
          const originalIndex = Math.floor(i / resampleRatio);
          processedData[i] = rawData[originalIndex];
        }
      } else {
        processedData = rawData;
      }
      
      // Split the audio into chunks and send them over WebSocket
      const chunkSize = 16000 * latencyMultiplier; // Adjust chunk size based on latency
      const numChunks = Math.ceil(processedData.length / chunkSize);
      
      for (let i = 0; i < numChunks; i++) {
        const start = i * chunkSize;
        const end = Math.min(start + chunkSize, processedData.length);
        const chunk = processedData.slice(start, end);
        
        // Send the chunk over WebSocket
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(chunk.buffer);
        }
        
        // Wait a short time between chunks to simulate real-time streaming
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
      setIsUploading(false);
      
      // Send EOF signal after a short delay
      setTimeout(() => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          // Send an empty chunk to signal end of stream
          const emptyChunk = new Float32Array(0);
          wsRef.current.send(emptyChunk.buffer);
        }
        setIsTranslating(false);
      }, 500);
      
    } catch (error) {
      console.error('Error processing audio:', error);
      setModelFeedback({
        message: `Error processing audio: ${error instanceof Error ? error.message : String(error)}`,
        severity: 'error'
      });
      setIsUploading(false);
      setIsTranslating(false);
    }
  };
  
  // Handle file uploads (video or audio)
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setMediaUrl(URL.createObjectURL(file));
      
      // Reset translation when new file is selected
      setTranslation("");
      setCurrentTranslation("Ready to translate");
      
      // Auto-upload the file if WebSocket is connected
      if (file && isConnected && sessionId) {
        console.log("File selected, waiting for playback to start");
        uploadFile(file);
      }else{
        console.log("File selected, but not connected",isConnected,sessionId);
      }
    }
  };
  
  // Handle YouTube URL input
  const handleYoutubeSubmit = async () => {
    if (youtubeUrl) {
      setIsProcessing(true);
      // Extract YouTube video ID
      const videoId = youtubeUrl.split('v=')[1]?.split('&')[0];
      if (videoId) {
        const embedUrl = `https://www.youtube.com/embed/${videoId}?enablejsapi=1`;
        setMediaUrl(embedUrl);
        
        // YouTube translation is not supported in the current backend
        setModelFeedback({
          message: 'YouTube translation is not yet supported. Please upload an audio/video file instead.',
          severity: 'warning'
        });
        setIsProcessing(false);
      }
    }
  };
  
  // Handle microphone recording
  const handleRecordToggle = async () => {
    if (!isRecording) {
      // Check if session is ready
      if (!sessionId || !isConnected || !modelLoaded) {
        setModelFeedback({
          message: 'Please wait for the model to load before recording',
          severity: 'warning'
        });
        return;
      }
      
      try {
        // Start recording
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        setIsRecording(true);
        
        // Create audio context for processing
        const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(stream);
        const processor = audioContext.createScriptProcessor(4096, 1, 1);
        
        // Store references for cleanup
        (window as any).currentStream = stream;
        (window as any).currentAudioContext = audioContext;
        (window as any).currentProcessor = processor;
        
        processor.onaudioprocess = (e) => {
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            const inputData = e.inputBuffer.getChannelData(0);
            
            // Resample to 16kHz if needed
            const sampleRate = audioContext.sampleRate;
            const targetSampleRate = 16000;
            const resampleRatio = targetSampleRate / sampleRate;
            
            const resampledLength = Math.floor(inputData.length * resampleRatio);
            const resampledData = new Float32Array(resampledLength);
            
            for (let i = 0; i < resampledLength; i++) {
              const originalIndex = Math.floor(i / resampleRatio);
              resampledData[i] = inputData[originalIndex];
            }
            
            // Send audio data through WebSocket
            wsRef.current.send(resampledData.buffer);
          }
        };
        
        source.connect(processor);
        processor.connect(audioContext.destination);
        
        setModelFeedback({
          message: 'Recording... Speak into your microphone',
          severity: 'info'
        });
      } catch (error) {
        console.error('Error accessing microphone:', error);
        setModelFeedback({
          message: 'Error accessing microphone. Please check permissions.',
          severity: 'error'
        });
      }
    } else {
      // Stop recording
      setIsRecording(false);
      
      // Clean up audio resources
      if ((window as any).currentStream) {
        (window as any).currentStream.getTracks().forEach((track: MediaStreamTrack) => track.stop());
        (window as any).currentStream = null;
      }
      
      if ((window as any).currentProcessor) {
        (window as any).currentProcessor.disconnect();
        (window as any).currentProcessor = null;
      }
      
      if ((window as any).currentAudioContext) {
        (window as any).currentAudioContext.close();
        (window as any).currentAudioContext = null;
      }
      
      setModelFeedback({
        message: 'Recording stopped',
        severity: 'success'
      });
    }
  };
  
  // Handle time update for syncing translations
  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
      mediaRef.current = videoRef.current;
    } else if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
      mediaRef.current = audioRef.current;
    }
    
    // Find the current translation based on time
    const current = translations.find((t, index) => {
      const nextTranslation = translations[index + 1];
      return t.time <= currentTime && (!nextTranslation || currentTime < nextTranslation.time);
    });
    
    if (current) {
      setCurrentTranslation(current.text);
    }
  };
  
  // Handle play event
  const handlePlay = () => {
    setIsPlaying(true);
  };
  
  // Handle pause event
  const handlePause = () => {
    setIsPlaying(false);
  };
  
  // Set up media references when the URL changes
  useEffect(() => {
    if (mediaUrl) {
      if (mediaUrl.includes('youtube.com/embed')) {
        // YouTube embed - need to handle differently
        mediaRef.current = null;
      } else {
        // Local media file
        const isVideo = mediaUrl.includes('video') || 
                        mediaUrl.endsWith('.mp4') || 
                        mediaUrl.endsWith('.webm');
        
        if (isVideo && videoRef.current) {
          mediaRef.current = videoRef.current;
        } else if (audioRef.current) {
          mediaRef.current = audioRef.current;
        }
      }
    }
  }, [mediaUrl]);
  
  // Handle translation bar dragging (make the entire bar draggable)
  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (translationBarRef.current) {
      setIsDragging(true);
      setDragStartPos({ 
        x: e.clientX - barPosition.left, 
        y: e.clientY - barPosition.top 
      });
    }
  };
  
  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (isDragging && translationBarRef.current) {
      const newLeft = e.clientX - dragStartPos.x;
      const newTop = e.clientY - dragStartPos.y;
      
      setBarPosition({
        left: Math.max(0, newLeft),
        top: Math.max(0, newTop)
      });
    }
  }, [isDragging, dragStartPos]);
  
  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);
  
  // Set up and remove event listeners for dragging
  useEffect(() => {
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [handleMouseMove, handleMouseUp]);
  
  // Handle language change
  const handleLanguageChange = (event: SelectChangeEvent) => {
    setTargetLanguage(event.target.value);
  };
  
  // Handle language menu selection
  const handleLanguageMenuSelect = (language: string) => {
    setTargetLanguage(language);
    setAnchorEl(null);
  };

  // Get and update the media container width for matching translation bar width
  useEffect(() => {
    const updateWidth = () => {
      if (mediaContainerRef.current) {
        const width = mediaContainerRef.current.offsetWidth;
        setContainerWidth(width);
      }
    };

    // Initial update
    updateWidth();

    // Update on window resize
    window.addEventListener('resize', updateWidth);
    return () => window.removeEventListener('resize', updateWidth);
  }, []);

  // Center the translation bar initially
  useEffect(() => {
    const centerBar = () => {
      if (translationBarRef.current && containerWidth > 0) {
        const barWidth = translationBarRef.current.offsetWidth;
        const windowWidth = window.innerWidth;
        const windowHeight = window.innerHeight;
        
        setBarPosition({
          left: (windowWidth - barWidth) / 2,
          top: windowHeight / 2 - 100 // Position it a bit above center
        });
      }
    };
    
    // Center when container width is known and on window resize
    centerBar();
    window.addEventListener('resize', centerBar);
    return () => window.removeEventListener('resize', centerBar);
  }, [containerWidth]);
  
  // Handle file selection
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null;
    setSelectedFile(file);
    
    // Reset translation when new file is selected
    setTranslation("");
    
    // Auto-upload the file if WebSocket is connected
    if (file && isConnected) {
      uploadFile(file);
    }
  };
  
  // Reset translation
  const resetTranslation = async () => {
    if (!sessionId) {
      setModelFeedback({
        message: 'No active session to reset',
        severity: 'warning'
      });
      return;
    }
    
    try {
      const response = await apiService.resetTranslation(sessionId);
      
      if (response.success) {
        // Clear translation
        setTranslation("");
        setCurrentTranslation("Translation reset. Ready for new input.");
        
        setModelFeedback({
          message: response.message || 'Translation reset successfully',
          severity: 'success'
        });
      } else {
        setModelFeedback({
          message: `Failed to reset: ${response.error}`,
          severity: 'error'
        });
      }
    } catch (error) {
      console.error('Error resetting translation:', error);
      setModelFeedback({
        message: `Error resetting: ${error instanceof Error ? error.message : String(error)}`,
        severity: 'error'
      });
    }
  };
  
  // Update latency multiplier
  const updateLatency = async (multiplier: number) => {
    if (!sessionId) return;
    
    setLatencyMultiplier(multiplier);
    
    try {
      const response = await apiService.updateLatency(sessionId, multiplier);
      
      if (response.success) {
        setModelFeedback({
          message: `Latency updated to ${multiplier}x`,
          severity: 'success'
        });
        
        // Clear the message after 2 seconds
        setTimeout(() => {
          setModelFeedback(null);
        }, 2000);
      } else {
        console.error('Failed to update latency:', response.error);
        setModelFeedback({
          message: `Failed to update latency: ${response.error}`,
          severity: 'error'
        });
      }
    } catch (error) {
      console.error('Error updating latency:', error);
      setModelFeedback({
        message: `Error updating latency: ${error instanceof Error ? error.message : String(error)}`,
        severity: 'error'
      });
    }
  };
  
  return (
    <ThemeProvider theme={darkTheme}>
      <Box
        sx={{
          minHeight: '100vh',
          background: 'linear-gradient(135deg, #0a1929 0%, #051321 100%)',
          backgroundSize: 'cover',
          py: 5,
        }}
      >
        <Container maxWidth="lg">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Typography
              variant="h4"
              align="center"
              gutterBottom
              sx={{
                mb: 4,
                background: 'linear-gradient(45deg, #2196f3, #21cbf3)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                fontWeight: 'bold',
              }}
            >
              InfiniSST Translation Demo
            </Typography>

            <Paper 
              elevation={0}
              sx={{ 
                p: 4, 
                mb: 5, 
                borderRadius: 4,
                background: alpha(darkTheme.palette.background.paper, 0.8),
                backdropFilter: 'blur(8px)',
                border: `1px solid ${alpha('#fff', 0.1)}`,
              }}
            >
              <Grid container spacing={3} alignItems="center">
                {/* File upload input */}
                <Grid item xs={12} md={4}>
                  <StyledUploadButton
                    variant="contained"
                    startIcon={<CloudUploadIcon />}
                    fullWidth
                    disabled={isProcessing || isRecording}
                    component="label"
                  >
                    Upload Video/Audio
                    <VisuallyHiddenInput 
                      type="file"
                      accept="video/*,audio/*"
                      onChange={handleFileUpload}
                    />
                  </StyledUploadButton>
                </Grid>
                
                {/* YouTube URL input */}
                <Grid item xs={12} md={5}>
                  <Box display="flex" alignItems="center">
                    <StyledYouTubeField
                      label="YouTube URL"
                      variant="outlined"
                      fullWidth
                      value={youtubeUrl}
                      onChange={(e) => setYoutubeUrl(e.target.value)}
                      placeholder="https://www.youtube.com/watch?v=..."
                      InputProps={{
                        startAdornment: <YouTubeIcon sx={{ mr: 1, color: 'red' }} />,
                      }}
                      disabled={isProcessing || isRecording}
                    />
                    <StyledIconButton 
                      color="primary" 
                      onClick={handleYoutubeSubmit}
                      sx={{ ml: 1, width: 56, height: 56 }}
                      disabled={!youtubeUrl || isProcessing || isRecording}
                    >
                      <PlayArrowIcon />
                    </StyledIconButton>
                  </Box>
                </Grid>
                
                {/* Microphone recording */}
                <Grid item xs={12} md={3}>
                  <StyledRecordButton
                    variant="contained"
                    color={isRecording ? "secondary" : "primary"}
                    startIcon={isRecording ? <StopIcon /> : <MicIcon />}
                    onClick={handleRecordToggle}
                    fullWidth
                    disabled={isProcessing || !modelLoaded}
                  >
                    {isRecording ? "Stop Recording" : "Record Audio"}
                  </StyledRecordButton>
                </Grid>
              </Grid>
            </Paper>
            
            {/* Media player */}
            <AnimatePresence>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                <StyledMediaContainer ref={mediaContainerRef}>
                  {isProcessing && (
                    <Box 
                      sx={{ 
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        display: 'flex', 
                        flexDirection: 'column', 
                        alignItems: 'center',
                        justifyContent: 'center',
                        backgroundColor: 'rgba(0,0,0,0.7)',
                        zIndex: 10,
                      }}
                    >
                      <CircularProgress size={60} thickness={4} sx={{ mb: 2 }} />
                      <Typography variant="h6" color="primary.light">
                        Processing your media...
                      </Typography>
                    </Box>
                  )}
                  
                  {mediaUrl ? (
                    mediaUrl.includes('youtube.com/embed') ? (
                      <iframe
                        width="100%"
                        height="100%"
                        src={mediaUrl}
                        title="YouTube video player"
                        frameBorder="0"
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                        allowFullScreen
                        style={{ position: 'absolute', top: 0, left: 0 }}
                      ></iframe>
                    ) : mediaUrl.includes('video') || mediaUrl.endsWith('.mp4') || mediaUrl.endsWith('.webm') ? (
                      <video
                        ref={videoRef}
                        src={mediaUrl}
                        controls
                        width="100%"
                        height="100%"
                        style={{ objectFit: 'contain' }}
                        onTimeUpdate={handleTimeUpdate}
                        onPlay={handlePlay}
                        onPause={handlePause}
                      ></video>
                    ) : (
                      <Box 
                        sx={{ 
                          width: '100%',
                          height: '100%',
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'center',
                          justifyContent: 'center',
                        }}
                      >
                        <Box 
                          sx={{ 
                            width: '80%',
                            padding: 3,
                            borderRadius: 4,
                            backgroundColor: alpha(darkTheme.palette.background.paper, 0.7),
                            backdropFilter: 'blur(8px)',
                            textAlign: 'center',
                          }}
                        >
                          <Typography variant="body1" color="text.secondary" gutterBottom>
                            Audio Playback
                          </Typography>
                          <audio
                            ref={audioRef}
                            src={mediaUrl}
                            controls
                            style={{ width: '100%', marginTop: 16 }}
                            onTimeUpdate={handleTimeUpdate}
                            onPlay={handlePlay}
                            onPause={handlePause}
                          ></audio>
                        </Box>
                      </Box>
                    )
                  ) : (
                    <Box 
                      sx={{ 
                        width: '100%', 
                        height: '100%',
                        display: 'flex', 
                        flexDirection: 'column',
                        alignItems: 'center', 
                        justifyContent: 'center',
                        p: 3,
                      }}
                    >
                      <Box 
                        component="img"
                        src="https://cdn-icons-png.flaticon.com/512/2913/2913967.png"
                        sx={{ width: 80, height: 80, opacity: 0.5, mb: 3 }}
                      />
                      <Typography variant="h6" color="text.secondary" align="center" gutterBottom>
                        No media loaded
                      </Typography>
                      <Typography variant="body2" color="text.secondary" align="center">
                        Upload a video/audio file, enter a YouTube URL, or record audio to start
                      </Typography>
                    </Box>
                  )}
                </StyledMediaContainer>
              </motion.div>
            </AnimatePresence>
          </motion.div>
        </Container>
        
        {/* Movable translation bar with enhanced features */}
        <AnimatePresence>
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 50 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            ref={translationBarRef}
            className="translation-bar"
            style={{
              position: 'fixed',
              top: `${barPosition.top}px`,
              left: `${barPosition.left}px`,
              width: containerWidth > 0 ? `${containerWidth}px` : 'auto',
              height: '70px', // Increased height for better visibility
              cursor: 'move',
              caretColor: 'transparent' // Prevents text selection cursor
            }}
            onMouseDown={handleMouseDown} // Make entire bar draggable
          >
            {/* Latency multiplier dropdown (new) */}
            <div className="latency-selector">
              <FormControl variant="standard" size="small" sx={{ minWidth: 130 }}>
                <InputLabel id="latency-select-label" sx={{ color: 'rgba(255, 255, 255, 0.7)', fontSize: '12px' }}>
                  Latency
                </InputLabel>
                <Select
                  labelId="latency-select-label"
                  value={latencyMultiplier}
                  onChange={(e) => {
                    const value = Number(e.target.value);
                    updateLatency(value);
                  }}
                  sx={{ 
                    color: 'white',
                    '.MuiSelect-icon': { color: 'rgba(255, 255, 255, 0.7)' },
                    '&:before': { borderColor: 'rgba(255, 255, 255, 0.3)' },
                    fontSize: '12px'
                  }}
                  MenuProps={{
                    sx: { 
                      '& .MuiMenuItem-root': {
                        fontSize: '12px'
                      }
                    }
                  }}
                >
                  {LATENCY_OPTIONS.map((option) => (
                    <MenuItem key={option.value} value={option.value}>
                      {option.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </div>
            <div className="drag-handle">
              <DragIndicatorIcon />
            </div>
            <div className="translation-text">
              {modelLoading ? (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <CircularProgress size={20} thickness={4} color="inherit" />
                  <span>Loading translation model...</span>
                </Box>
              ) : (
                currentTranslation
              )}
            </div>
            <div className="language-selector">
              <StyledIconButton 
                color="primary"
                size="small"
                aria-controls="language-menu"
                aria-haspopup="true"
                onClick={(event) => setAnchorEl(event.currentTarget)}
              >
                <LanguageIcon fontSize="small" />
              </StyledIconButton>
              <Menu
                id="language-menu"
                anchorEl={anchorEl}
                keepMounted
                open={Boolean(anchorEl)}
                onClose={() => setAnchorEl(null)}
              >
                <MenuItem value="en" onClick={() => handleLanguageMenuSelect('en')}>English</MenuItem>
                <MenuItem value="es" onClick={() => handleLanguageMenuSelect('es')}>Spanish</MenuItem>
                <MenuItem value="fr" onClick={() => handleLanguageMenuSelect('fr')}>French</MenuItem>
                <MenuItem value="de" onClick={() => handleLanguageMenuSelect('de')}>German</MenuItem>
                <MenuItem value="zh" onClick={() => handleLanguageMenuSelect('zh')}>Chinese</MenuItem>
                <MenuItem value="ja" onClick={() => handleLanguageMenuSelect('ja')}>Japanese</MenuItem>
              </Menu>
            </div>
          </motion.div>
        </AnimatePresence>
        
        {/* Toast notifications for model feedback */}
        {modelFeedback && (
          <Snackbar 
            open={true}
            autoHideDuration={6000}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
          >
            <Alert 
              severity={modelFeedback.severity}
              elevation={6}
              variant="filled"
            >
              {modelFeedback.message}
            </Alert>
          </Snackbar>
        )}
      </Box>
    </ThemeProvider>
  );
};

export default TranslationDemo; 