import React, { useState, useRef, useEffect } from 'react';
import { Box, Paper, Typography, IconButton, Theme } from '@mui/material';
import { styled } from '@mui/material/styles';
import DragIndicatorIcon from '@mui/icons-material/DragIndicator';
import CloseIcon from '@mui/icons-material/Close';

// 创建一个可拖动的容器组件
const DraggablePaper = styled(Paper)(({ theme }: { theme: Theme }) => ({
  position: 'fixed',
  padding: theme.spacing(2),
  backgroundColor: 'rgba(255, 255, 255, 0.9)',
  backdropFilter: 'blur(8px)',
  borderRadius: theme.spacing(1),
  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
  minWidth: '300px',
  maxWidth: '600px',
  maxHeight: '80vh',
  overflow: 'auto',
  cursor: 'move',
  userSelect: 'none',
  zIndex: 1000,
  '&:hover': {
    boxShadow: '0 6px 12px rgba(0, 0, 0, 0.15)',
  },
}));

// 拖动句柄样式
const DragHandle = styled(Box)(({ theme }: { theme: Theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: theme.spacing(1),
  borderBottom: `1px solid ${theme.palette.divider}`,
  marginBottom: theme.spacing(1),
  cursor: 'move',
}));

// 内容区域样式
const ContentBox = styled(Box)(({ theme }: { theme: Theme }) => ({
  padding: theme.spacing(1),
  maxHeight: 'calc(80vh - 100px)',
  overflow: 'auto',
  '&::-webkit-scrollbar': {
    width: '8px',
  },
  '&::-webkit-scrollbar-track': {
    background: 'rgba(0, 0, 0, 0.05)',
    borderRadius: '4px',
  },
  '&::-webkit-scrollbar-thumb': {
    background: 'rgba(0, 0, 0, 0.2)',
    borderRadius: '4px',
    '&:hover': {
      background: 'rgba(0, 0, 0, 0.3)',
    },
  },
}));

interface TranslationDisplayProps {
  text: string;
  onClose?: () => void;
  initialPosition?: { x: number; y: number };
}

export const TranslationDisplay: React.FC<TranslationDisplayProps> = ({
  text,
  onClose,
  initialPosition = { x: 20, y: 20 },
}) => {
  const [position, setPosition] = useState(initialPosition);
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  // 处理拖动开始
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.target instanceof HTMLElement && e.target.closest('.drag-handle')) {
      setIsDragging(true);
      const rect = containerRef.current?.getBoundingClientRect();
      if (rect) {
        setDragOffset({
          x: e.clientX - rect.left,
          y: e.clientY - rect.top,
        });
      }
    }
  };

  // 处理拖动
  const handleMouseMove = (e: MouseEvent) => {
    if (isDragging && containerRef.current) {
      const x = e.clientX - dragOffset.x;
      const y = e.clientY - dragOffset.y;
      
      // 确保不会拖出屏幕
      const maxX = window.innerWidth - (containerRef.current.offsetWidth || 0);
      const maxY = window.innerHeight - (containerRef.current.offsetHeight || 0);
      
      setPosition({
        x: Math.max(0, Math.min(x, maxX)),
        y: Math.max(0, Math.min(y, maxY)),
      });
    }
  };

  // 处理拖动结束
  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // 添加和移除全局事件监听器
  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging]);

  return (
    <DraggablePaper
      ref={containerRef}
      style={{
        left: `${position.x}px`,
        top: `${position.y}px`,
        opacity: isDragging ? 0.8 : 1,
      }}
      onMouseDown={handleMouseDown}
    >
      <DragHandle className="drag-handle">
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <DragIndicatorIcon color="action" />
          <Typography variant="subtitle2" color="text.secondary">
            Translation
          </Typography>
        </Box>
        {onClose && (
          <IconButton
            size="small"
            onClick={onClose}
            sx={{ '&:hover': { backgroundColor: 'rgba(0, 0, 0, 0.04)' } }}
          >
            <CloseIcon fontSize="small" />
          </IconButton>
        )}
      </DragHandle>
      <ContentBox>
        <Typography
          variant="body1"
          sx={{
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            lineHeight: 1.6,
          }}
        >
          {text}
        </Typography>
      </ContentBox>
    </DraggablePaper>
  );
}; 