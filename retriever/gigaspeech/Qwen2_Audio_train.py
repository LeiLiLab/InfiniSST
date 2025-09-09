import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from typing import List, Union
import warnings
warnings.filterwarnings("ignore")


class Qwen2AudioSpeechEncoder:
    """
    Qwen2-Audio Speech Encoder wrapper for speech-to-embedding
    """
    def __init__(self, model_name="Qwen/Qwen2-Audio-7B-Instruct", device="cuda"):
        self.device = device
        print(f"[INFO] Loading Qwen2-Audio model: {model_name}")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        self.model.eval()
        
        # Get the audio encoder from the model
        self.audio_encoder = self.model.audio_tower
        
        print(f"[INFO] Qwen2-Audio model loaded successfully on {device}")
    
    def predict(self, audio_paths: List[str], max_length: int = None) -> np.ndarray:
        """
        Extract audio embeddings from audio files
        
        Args:
            audio_paths: List of audio file paths
            max_length: Maximum audio length in samples (optional)
            
        Returns:
            numpy array of shape [batch_size, embedding_dim]
        """
        embeddings = []
        
        with torch.no_grad():
            for audio_path in audio_paths:
                try:
                    # Load and preprocess audio
                    audio, sr = librosa.load(audio_path, sr=16000)  # Qwen2-Audio expects 16kHz
                    
                    # Limit audio length if specified
                    if max_length and len(audio) > max_length:
                        audio = audio[:max_length]
                    
                    # Process audio through the processor
                    inputs = self.processor(
                        audios=audio,
                        sampling_rate=sr,
                        return_tensors="pt"
                    )
                    
                    # Move inputs to device
                    for key in inputs:
                        if isinstance(inputs[key], torch.Tensor):
                            inputs[key] = inputs[key].to(self.model.device)
                    
                    # Extract audio features through the audio encoder
                    audio_features = self.audio_encoder(inputs["input_features"])
                    
                    # Pool the features to get a fixed-size embedding
                    # Use mean pooling over the sequence dimension
                    if len(audio_features.shape) == 3:  # [batch, seq_len, hidden_dim]
                        pooled_features = audio_features.mean(dim=1)  # [batch, hidden_dim]
                    else:
                        pooled_features = audio_features
                    
                    # Convert to CPU and numpy
                    embedding = pooled_features.cpu().float().numpy()
                    embeddings.append(embedding.squeeze())
                    
                except Exception as e:
                    print(f"[ERROR] Failed to process audio {audio_path}: {e}")
                    # Return zero embedding as fallback
                    dummy_embedding = np.zeros(4096)  # Qwen2-Audio typical hidden size
                    embeddings.append(dummy_embedding)
        
        return np.stack(embeddings)


class Qwen2AudioTextEncoder:
    """
    Qwen2-Audio Text Encoder wrapper for text-to-embedding
    """
    def __init__(self, model_name="Qwen/Qwen2-Audio-7B-Instruct", device="cuda"):
        self.device = device
        print(f"[INFO] Loading Qwen2-Audio text encoder: {model_name}")
        
        # Load processor and model (reuse the same model for text encoding)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        self.model.eval()
        
        print(f"[INFO] Qwen2-Audio text encoder loaded successfully on {device}")
    
    def predict(self, texts: List[str], source_lang: str = "eng_Latn") -> np.ndarray:
        """
        Extract text embeddings from text strings
        
        Args:
            texts: List of text strings
            source_lang: Source language (kept for compatibility, not used in Qwen2-Audio)
            
        Returns:
            numpy array of shape [batch_size, embedding_dim]
        """
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                try:
                    # Tokenize text
                    inputs = self.processor.tokenizer(
                        text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                    
                    # Move inputs to device
                    for key in inputs:
                        inputs[key] = inputs[key].to(self.model.device)
                    
                    # Get text embeddings from the language model
                    outputs = self.model.language_model(**inputs, output_hidden_states=True)
                    
                    # Use the last hidden state and pool it
                    last_hidden_state = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
                    
                    # Mean pooling over sequence dimension, excluding padding tokens
                    attention_mask = inputs["attention_mask"]
                    masked_embeddings = last_hidden_state * attention_mask.unsqueeze(-1)
                    pooled_embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                    
                    # Convert to CPU and numpy
                    embedding = pooled_embedding.cpu().float().numpy()
                    embeddings.append(embedding.squeeze())
                    
                except Exception as e:
                    print(f"[ERROR] Failed to process text '{text[:50]}...': {e}")
                    # Return zero embedding as fallback
                    dummy_embedding = np.zeros(4096)  # Qwen2-Audio typical hidden size
                    embeddings.append(dummy_embedding)
        
        return np.stack(embeddings)


class ContrastiveQwen2AudioModel(nn.Module):
    """
    Contrastive Speech-Text Model using Qwen2-Audio encoders
    """
    def __init__(self, speech_encoder, text_encoder, hidden_dim=4096, proj_dim=512, unfreeze_layers=0):
        super().__init__()
        self.speech_encoder = speech_encoder
        self.text_encoder = text_encoder
        
        # Projection layers
        self.proj_speech = nn.Linear(hidden_dim, proj_dim)
        self.proj_text = nn.Linear(hidden_dim, proj_dim)
        
        # Freeze most of the model parameters, only unfreeze last few layers if specified
        if unfreeze_layers > 0:
            self._unfreeze_last_layers(unfreeze_layers)
        else:
            # Freeze all encoder parameters
            for param in self.speech_encoder.model.parameters():
                param.requires_grad = False
            for param in self.text_encoder.model.parameters():
                param.requires_grad = False
        
        print(f"[INFO] ContrastiveQwen2AudioModel initialized with proj_dim={proj_dim}, unfreeze_layers={unfreeze_layers}")
    
    def _unfreeze_last_layers(self, num_layers):
        """Unfreeze the last num_layers of both encoders"""
        print(f"[INFO] Unfreezing last {num_layers} layers of Qwen2-Audio encoders")
        
        # First freeze all parameters
        for param in self.speech_encoder.model.parameters():
            param.requires_grad = False
        for param in self.text_encoder.model.parameters():
            param.requires_grad = False
        
        # Then unfreeze the last few layers
        # For Qwen2-Audio, we need to access the transformer layers
        try:
            # Unfreeze audio encoder layers
            if hasattr(self.speech_encoder.model, 'audio_tower'):
                audio_layers = self.speech_encoder.model.audio_tower
                if hasattr(audio_layers, 'layers') or hasattr(audio_layers, 'encoder'):
                    layers = getattr(audio_layers, 'layers', getattr(audio_layers, 'encoder', None))
                    if layers and hasattr(layers, 'layers'):
                        layers = layers.layers
                    if layers:
                        for layer in layers[-num_layers:]:
                            for param in layer.parameters():
                                param.requires_grad = True
            
            # Unfreeze language model layers
            if hasattr(self.text_encoder.model, 'language_model'):
                lm_layers = self.text_encoder.model.language_model
                if hasattr(lm_layers, 'layers'):
                    for layer in lm_layers.layers[-num_layers:]:
                        for param in layer.parameters():
                            param.requires_grad = True
                            
        except Exception as e:
            print(f"[WARN] Could not unfreeze layers automatically: {e}")
            print(f"[INFO] All encoder parameters remain frozen")
    
    def encode_audio(self, audio_paths: List[str]) -> torch.Tensor:
        """Encode audio files to embeddings"""
        speech_embeddings = self.speech_encoder.predict(audio_paths)  # [B, hidden_dim]
        if isinstance(speech_embeddings, np.ndarray):
            speech_embeddings = torch.from_numpy(speech_embeddings)
        
        speech_embeddings = speech_embeddings.to(self.proj_speech.weight.device)
        if not speech_embeddings.requires_grad:
            speech_embeddings = speech_embeddings.clone().detach().requires_grad_(True)
        
        return F.normalize(self.proj_speech(speech_embeddings), dim=-1)
    
    def encode_text(self, texts: List[str], source_lang: str = "eng_Latn") -> torch.Tensor:
        """Encode text strings to embeddings"""
        with torch.no_grad():
            text_embeddings = self.text_encoder.predict(texts, source_lang=source_lang)
        
        if isinstance(text_embeddings, np.ndarray):
            text_embeddings = torch.from_numpy(text_embeddings)
        
        text_embeddings = text_embeddings.to(self.proj_text.weight.device)
        if not text_embeddings.requires_grad:
            text_embeddings = text_embeddings.clone().detach().requires_grad_(True)
        
        return F.normalize(self.proj_text(text_embeddings), dim=-1)


def load_glossary_terms(path):
    """Load glossary terms from JSON file"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [item["term"] if isinstance(item, dict) else str(item) for item in data]
        elif isinstance(data, dict):
            return list(data.keys())
        else:
            return []
    except Exception as e:
        print(f"[ERROR] Failed to load glossary from {path}: {e}")
        return []


def encode_texts_in_batches(model, texts, batch_size=64, device="cuda"):
    """Encode texts in batches using the model's text encoder"""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            embeddings = model.encode_text(batch_texts)
            all_embeddings.append(embeddings.detach())
        except Exception as e:
            print(f"[ERROR] Failed to encode text batch {i//batch_size}: {e}")
            # Create dummy embeddings
            dummy_emb = torch.zeros(len(batch_texts), 512).to(device)
            all_embeddings.append(dummy_emb)
    
    return torch.cat(all_embeddings, dim=0)


def encode_audios_in_batches(model, audio_paths, batch_size=32, device="cuda"):
    """Encode audios in batches using the model's audio encoder"""
    all_embeddings = []
    
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i:i + batch_size]
        try:
            embeddings = model.encode_audio(batch_paths)
            all_embeddings.append(embeddings.detach())
        except Exception as e:
            print(f"[ERROR] Failed to encode audio batch {i//batch_size}: {e}")
            # Create dummy embeddings
            dummy_emb = torch.zeros(len(batch_paths), 512).to(device)
            all_embeddings.append(dummy_emb)
    
    return torch.cat(all_embeddings, dim=0)
