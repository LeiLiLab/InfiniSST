import torch
import torchaudio
import librosa
import numpy as np
from typing import List, Dict, Any, Optional
import copy


def split_audio_into_chunks(audio_tensor: torch.Tensor, chunk_duration: float = 2.0, 
                          sample_rate: int = 48000, overlap: float = 0.0) -> List[torch.Tensor]:
    """
    Split audio tensor into fixed-duration chunks.
    
    Args:
        audio_tensor: Input audio tensor, shape (1, samples) or (samples,)
        chunk_duration: Duration of each chunk in seconds
        sample_rate: Sample rate of the audio
        overlap: Overlap between chunks in seconds (0 = no overlap)
        
    Returns:
        List of audio chunks as tensors
    """
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.squeeze()
    
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap * sample_rate)
    step_samples = chunk_samples - overlap_samples
    
    total_samples = audio_tensor.shape[0]
    chunks = []
    
    for start in range(0, total_samples - chunk_samples + 1, step_samples):
        end = start + chunk_samples
        chunk = audio_tensor[start:end]
        chunks.append(chunk.unsqueeze(0))  # Add channel dimension back
    
    # Handle the last chunk if it's shorter than chunk_duration
    if total_samples % step_samples != 0:
        last_start = total_samples - chunk_samples
        if last_start > 0 and last_start not in range(0, total_samples - chunk_samples + 1, step_samples):
            last_chunk = audio_tensor[last_start:]
            if last_chunk.shape[0] >= int(0.5 * chunk_samples):  # Only keep if >= 50% of chunk duration
                # Pad to exact chunk length if needed
                if last_chunk.shape[0] < chunk_samples:
                    padding = chunk_samples - last_chunk.shape[0]
                    last_chunk = torch.nn.functional.pad(last_chunk, (0, padding))
                chunks.append(last_chunk.unsqueeze(0))
    
    return chunks


def split_audio_from_path(audio_path: str, chunk_duration: float = 2.0, 
                         target_sr: int = 48000, overlap: float = 0.0) -> List[torch.Tensor]:
    """
    Load audio from file path and split into chunks.
    
    Args:
        audio_path: Path to audio file
        chunk_duration: Duration of each chunk in seconds
        target_sr: Target sample rate
        overlap: Overlap between chunks in seconds
        
    Returns:
        List of audio chunks as tensors
    """
    try:
        # Load audio using librosa for better compatibility
        waveform, sr = librosa.load(audio_path, sr=target_sr)
        audio_tensor = torch.tensor(waveform, dtype=torch.float32)
        return split_audio_into_chunks(audio_tensor, chunk_duration, target_sr, overlap)
    except Exception as e:
        print(f"Error loading audio from {audio_path}: {e}")
        return []


def create_chunked_samples(samples: List[Dict[str, Any]], chunk_duration: float = 2.0, 
                          target_sr: int = 48000, overlap: float = 0.0) -> List[Dict[str, Any]]:
    """
    Create chunked samples from sentence-level samples.
    
    Args:
        samples: List of sample dictionaries containing audio tensors or paths
        chunk_duration: Duration of each chunk in seconds
        target_sr: Target sample rate
        overlap: Overlap between chunks in seconds
        
    Returns:
        List of chunked sample dictionaries
    """
    chunked_samples = []
    
    for sample_idx, sample in enumerate(samples):
        audio_chunks = []
        
        # Get audio chunks from tensor or path
        if "audio_tensor" in sample and sample["audio_tensor"] is not None:
            audio_chunks = split_audio_into_chunks(
                sample["audio_tensor"], chunk_duration, target_sr, overlap
            )
        elif "audio" in sample:
            audio_path = sample["audio"] if isinstance(sample["audio"], str) else sample["audio"].get("path")
            if audio_path:
                audio_chunks = split_audio_from_path(
                    audio_path, chunk_duration, target_sr, overlap
                )
        
        # Create new samples for each chunk
        for chunk_idx, chunk in enumerate(audio_chunks):
            chunked_sample = copy.deepcopy(sample)
            
            # Update audio tensor
            chunked_sample["audio_tensor"] = chunk
            
            # Update metadata
            original_id = sample.get("segment_id", f"sample_{sample_idx}")
            chunked_sample["segment_id"] = f"{original_id}_chunk_{chunk_idx}"
            chunked_sample["original_segment_id"] = original_id
            chunked_sample["chunk_index"] = chunk_idx
            chunked_sample["chunk_duration"] = chunk_duration
            
            # Calculate chunk timing if available
            if "begin_time" in sample and "end_time" in sample:
                original_duration = sample["end_time"] - sample["begin_time"]
                chunk_start_ratio = chunk_idx * (chunk_duration - overlap) / original_duration
                chunk_end_ratio = min(1.0, (chunk_idx * (chunk_duration - overlap) + chunk_duration) / original_duration)
                
                chunked_sample["begin_time"] = sample["begin_time"] + chunk_start_ratio * original_duration
                chunked_sample["end_time"] = sample["begin_time"] + chunk_end_ratio * original_duration
                chunked_sample["original_begin_time"] = sample["begin_time"]
                chunked_sample["original_end_time"] = sample["end_time"]
            
            # Keep original text but mark as chunked
            chunked_sample["is_chunked"] = True
            chunked_sample["total_chunks"] = len(audio_chunks)
            
            chunked_samples.append(chunked_sample)
    
    return chunked_samples


def load_and_chunk_preprocessed_samples(json_path: str, chunk_duration: float = 2.0,
                                       target_sr: int = 48000, overlap: float = 0.0,
                                       max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load preprocessed samples and convert them to chunks.
    
    Args:
        json_path: Path to preprocessed samples JSON file
        chunk_duration: Duration of each chunk in seconds
        target_sr: Target sample rate
        overlap: Overlap between chunks in seconds
        max_samples: Maximum number of original samples to process (None for all)
        
    Returns:
        List of chunked sample dictionaries
    """
    from new_giga_speech import load_preprocessed_samples
    
    # Load original samples
    samples = load_preprocessed_samples(json_path, with_tensor=True)
    
    if max_samples is not None:
        samples = samples[:max_samples]
    
    print(f"Loaded {len(samples)} sentence-level samples")
    
    # Convert to chunks
    chunked_samples = create_chunked_samples(samples, chunk_duration, target_sr, overlap)
    
    print(f"Created {len(chunked_samples)} chunks from {len(samples)} original samples")
    print(f"Average chunks per sample: {len(chunked_samples) / len(samples):.2f}")
    
    return chunked_samples


if __name__ == "__main__":
    # Test the chunking functionality
    test_path = "data/test_preprocessed_samples_merged.json"
    chunked_samples = load_and_chunk_preprocessed_samples(
        test_path, 
        chunk_duration=2.0, 
        target_sr=48000, 
        overlap=0.0,
        max_samples=2
    )
    
    print(f"\nFirst chunk sample keys: {list(chunked_samples[0].keys())}")
    print(f"First chunk audio tensor shape: {chunked_samples[0]['audio_tensor'].shape}")
    print(f"Chunk duration at 48kHz: {chunked_samples[0]['audio_tensor'].shape[-1] / 48000:.2f}s")


