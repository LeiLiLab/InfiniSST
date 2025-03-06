import numpy as np

import torch

from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
from transformers.utils import logging

logger = logging.get_logger(__name__)

def new_torch_extract_fbank_features(self, waveform: np.array, device: str = "cpu") -> np.ndarray:
    waveform = torch.from_numpy(waveform).type(torch.float32)

    window = torch.hann_window(self.n_fft)
    if device != "cpu":
        waveform = waveform.to(device)
        window = window.to(device)
    stft = torch.stft(waveform, self.n_fft, self.hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    mel_filters = torch.from_numpy(self.mel_filters).type(torch.float32)
    if device != "cpu":
        mel_filters = mel_filters.to(device)
    mel_spec = mel_filters.T @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    # if waveform.dim() == 2:
    #     max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
    #     log_spec = torch.maximum(log_spec, max_val - 8.0)
    # else:
    #     log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    if device != "cpu":
        log_spec = log_spec.detach().cpu()
    return log_spec.numpy()

def patch_qwen2ac():
    WhisperFeatureExtractor._torch_extract_fbank_features = new_torch_extract_fbank_features