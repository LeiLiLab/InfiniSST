import React from 'react';
import {
  Box,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Typography,
  SelectChangeEvent,
} from '@mui/material';
import { SubtitleStyle } from '../types';

interface SubtitleSettingsProps {
  style: SubtitleStyle;
  onChange: (style: SubtitleStyle) => void;
}

export const SubtitleSettings: React.FC<SubtitleSettingsProps> = ({ style, onChange }) => {
  const handleChange = (field: keyof SubtitleStyle) => (
    event: React.ChangeEvent<HTMLInputElement | { value: unknown }>
  ) => {
    const value = event.target.value;
    onChange({
      ...style,
      [field]: value,
    });
  };

  const handleSelectChange = (field: keyof SubtitleStyle) => (
    event: SelectChangeEvent
  ) => {
    const value = event.target.value;
    onChange({
      ...style,
      [field]: value,
    });
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Subtitle Settings
      </Typography>

      <Box sx={{ mb: 2 }}>
        <Typography gutterBottom>Font Size</Typography>
        <Slider
          value={Number(style.fontSize)}
          onChange={(_, value) => onChange({ ...style, fontSize: value as number })}
          min={12}
          max={48}
          step={1}
          valueLabelDisplay="auto"
        />
      </Box>

      <Box sx={{ mb: 2 }}>
        <Typography gutterBottom>Opacity</Typography>
        <Slider
          value={style.opacity}
          onChange={(_, value) => onChange({ ...style, opacity: value as number })}
          min={0}
          max={1}
          step={0.1}
          valueLabelDisplay="auto"
        />
      </Box>

      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>Position</InputLabel>
        <Select
          value={style.subtitlePosition}
          label="Position"
          onChange={handleSelectChange('subtitlePosition')}
        >
          <MenuItem value="top">Top</MenuItem>
          <MenuItem value="bottom">Bottom</MenuItem>
        </Select>
      </FormControl>

      <Box sx={{ mb: 2 }}>
        <TextField
          fullWidth
          label="Font Color"
          type="color"
          value={style.fontColor}
          onChange={handleChange('fontColor')}
          sx={{ '& input': { height: '40px' } }}
        />
      </Box>

      <Box sx={{ mb: 2 }}>
        <TextField
          fullWidth
          label="Background Color"
          type="color"
          value={style.backgroundColor}
          onChange={handleChange('backgroundColor')}
          sx={{ '& input': { height: '40px' } }}
        />
      </Box>
    </Box>
  );
}; 