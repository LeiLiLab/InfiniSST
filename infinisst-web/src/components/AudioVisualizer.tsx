import React, { useEffect, useRef } from 'react';
import { styled } from '@mui/material/styles';

interface AudioVisualizerProps {
  volume: number;
  isActive: boolean;
}

const VisualizerContainer = styled('div')({
  width: '100%',
  height: '60px',
  backgroundColor: '#f5f5f5',
  borderRadius: '4px',
  overflow: 'hidden',
  position: 'relative',
});

const VolumeBar = styled('div')<{ volume: number; isActive: boolean }>(({ volume, isActive }) => ({
  position: 'absolute',
  bottom: 0,
  left: 0,
  width: '100%',
  height: `${Math.min(100, volume * 1000)}%`,
  backgroundColor: isActive ? '#4CAF50' : '#9e9e9e',
  transition: 'height 0.1s ease',
}));

const VolumeText = styled('div')({
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  color: '#666',
  fontSize: '12px',
  textAlign: 'center',
  width: '100%',
});

export const AudioVisualizer: React.FC<AudioVisualizerProps> = ({ volume, isActive }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      const container = containerRef.current;
      const height = container.clientHeight;
      const volumeHeight = Math.min(height, volume * height * 10);
      
      // Add a subtle animation effect
      container.style.setProperty('--volume-height', `${volumeHeight}px`);
    }
  }, [volume]);

  return (
    <VisualizerContainer ref={containerRef}>
      <VolumeBar volume={volume} isActive={isActive} />
      <VolumeText>
        {isActive ? 'Audio Level' : 'No Audio Input'}
      </VolumeText>
    </VisualizerContainer>
  );
}; 