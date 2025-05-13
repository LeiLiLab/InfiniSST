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
  Menu
} from '@mui/material';
import MicIcon from '@mui/icons-material/Mic';
import StopIcon from '@mui/icons-material/Stop';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import YouTubeIcon from '@mui/icons-material/YouTube';
import DragIndicatorIcon from '@mui/icons-material/DragIndicator';
import LanguageIcon from '@mui/icons-material/Language';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import { motion, AnimatePresence } from 'framer-motion';
import apiService, { Translation } from '../services/api';
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
  const [barPosition, setBarPosition] = useState<{ top: number, left: number }>({ top: 100, left: 20 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStartPos, setDragStartPos] = useState<{ x: number, y: number }>({ x: 0, y: 0 });
  const [targetLanguage, setTargetLanguage] = useState<string>('en');
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const mediaRef = useRef<HTMLVideoElement | HTMLAudioElement | null>(null);
  const translationBarRef = useRef<HTMLDivElement>(null);
  
  // Handle file uploads (video or audio)
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setIsProcessing(true);
      const url = URL.createObjectURL(file);
      setMediaUrl(url);
      
      try {
        // Send to backend for translation
        const translationResults = await apiService.translateAudio(file, targetLanguage);
        setTranslations(translationResults);
      } catch (error) {
        console.error('Error processing file:', error);
      } finally {
        setIsProcessing(false);
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
        
        try {
          // Send to backend for processing
          await apiService.translateYoutubeVideo(videoId, targetLanguage);
          // For demo, we'll use mock translations since we can't easily get real-time 
          // translations from the YouTube iframe
          setTranslations([
            { time: 1.5, text: "Hello, welcome to the YouTube translation demo" },
            { time: 3.2, text: "This is an example of a YouTube video translation" },
            { time: 6.8, text: "The translation appears as if in sync with the video" },
            { time: 10.5, text: "You can drag this bar to reposition it" },
            { time: 15.2, text: "Similar to lyrics display in music apps" }
          ]);
        } catch (error) {
          console.error('Error processing YouTube video:', error);
        } finally {
          setIsProcessing(false);
        }
      }
    }
  };
  
  // Handle microphone recording
  const handleRecordToggle = () => {
    if (!isRecording) {
      // Start recording
      navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
          const mediaRecorder = new MediaRecorder(stream);
          const audioChunks: BlobPart[] = [];
          
          // Initialize WebSocket connection for real-time translation
          apiService.startRealtimeTranslation(
            targetLanguage,
            (translation) => {
              // Add new translation to list
              setTranslations(prev => {
                const updatedTranslations = [...prev, translation];
                updatedTranslations.sort((a, b) => a.time - b.time);
                return updatedTranslations;
              });
            },
            (error) => {
              console.error('WebSocket error:', error);
            }
          );
          
          mediaRecorder.addEventListener('dataavailable', event => {
            audioChunks.push(event.data);
            
            // Send audio chunk for real-time translation
            const audioBlob = new Blob([event.data], { type: 'audio/webm' });
            apiService.sendAudioChunk(audioBlob);
          });
          
          mediaRecorder.addEventListener('stop', () => {
            // Stop WebSocket connection
            apiService.stopRealtimeTranslation();
            
            // Create audio blob and URL
            const audioBlob = new Blob(audioChunks);
            const audioUrl = URL.createObjectURL(audioBlob);
            setMediaUrl(audioUrl);
            setIsRecording(false);
          });
          
          // Start recording with 1-second intervals for chunks
          mediaRecorder.start(1000);
          (window as any).mediaRecorder = mediaRecorder;
          setIsRecording(true);
        })
        .catch(error => {
          console.error('Error accessing microphone:', error);
        });
    } else {
      // Stop recording
      const mediaRecorder = (window as any).mediaRecorder;
      if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
      }
    }
  };
  
  // Handle media playback time updates
  const handleTimeUpdate = () => {
    if (mediaRef.current) {
      const currentMediaTime = mediaRef.current.currentTime;
      setCurrentTime(currentMediaTime);
      
      // Find the current translation based on time
      const currentTranslation = translations.find(
        (t, index) => 
          currentMediaTime >= t.time && 
          (index === translations.length - 1 || currentMediaTime < translations[index + 1].time)
      );
      
      if (currentTranslation) {
        setCurrentTranslation(currentTranslation.text);
      } else {
        setCurrentTranslation('');
      }
    }
  };
  
  // Handle media play state
  const handlePlay = () => {
    setIsPlaying(true);
  };
  
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
  
  // Handle translation bar dragging
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
                    disabled={isProcessing}
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
                <StyledMediaContainer>
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
        
        {/* Movable translation bar */}
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
            }}
          >
            <div 
              className="drag-handle"
              onMouseDown={handleMouseDown}
            >
              <DragIndicatorIcon />
            </div>
            <div className="translation-text">
              {currentTranslation}
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
      </Box>
    </ThemeProvider>
  );
};

export default TranslationDemo; 