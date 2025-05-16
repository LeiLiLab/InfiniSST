import React from 'react';
import { styled } from '@mui/material/styles';
import { SubtitleStyle, Position } from '../types';

interface SubtitleProps {
  text: string;
  style: SubtitleStyle;
}

const SubtitleContainer = styled('div')<{ style: SubtitleStyle }>(({ style }) => ({
  position: 'fixed',
  left: '50%',
  transform: 'translateX(-50%)',
  bottom: style.subtitlePosition === 'bottom' ? '5%' : 'auto',
  top: style.subtitlePosition === 'top' ? '5%' : 'auto',
  padding: style.padding,
  borderRadius: style.borderRadius,
  backgroundColor: style.backgroundColor,
  color: style.fontColor,
  fontSize: `${style.fontSize}px`,
  opacity: style.opacity,
  maxWidth: '80%',
  textAlign: 'center',
  transition: 'all 0.3s ease',
  zIndex: 1000,
}));

export const Subtitle: React.FC<SubtitleProps> = ({ text, style }) => {
  if (!text) return null;

  return (
    <SubtitleContainer style={style}>
      {text}
    </SubtitleContainer>
  );
}; 