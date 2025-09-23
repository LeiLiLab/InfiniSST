import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
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
        
        print(f"[INFO] Qwen2-Audio model loaded successfully on {device}")
    
    def get_shared_model(self):
        """Return the model and processor for sharing with text encoder"""
        return {
            'model': self.model,
            'processor': self.processor
        }
    
    def predict(self, audio_paths: List[str], max_length: int = None) -> torch.Tensor:
        """
        Extract audio embeddings from audio files
        
        Args:
            audio_paths: List of audio file paths
            max_length: Maximum audio length in samples (optional)
            
        Returns:
            torch tensor of shape [batch_size, embedding_dim]
        """
        embeddings = []
        
        # 移除 torch.no_grad() 以保持梯度链
        for audio_path in audio_paths:
                try:
                    # Load and preprocess audio
                    audio, sr = librosa.load(audio_path, sr=16000)  # Qwen2-Audio expects 16kHz
                    
                    # Ensure audio is not empty
                    if len(audio) == 0:
                        print(f"[WARN] Empty audio file: {audio_path}")
                        dummy_embedding = torch.zeros(4096, device=self.model.device)
                        embeddings.append(dummy_embedding)
                        continue
                    
                    # Limit audio length if specified (default to 30 seconds max)
                    max_samples = max_length if max_length else 16000 * 30  # 30 seconds at 16kHz
                    if len(audio) > max_samples:
                        audio = audio[:max_samples]
                    
                    # Ensure minimum length (pad if too short)
                    min_samples = 1600  # 0.1 seconds
                    if len(audio) < min_samples:
                        audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')
                    
                    # Convert to float32 and ensure it's a numpy array
                    audio = np.array(audio, dtype=np.float32)
                    
                    # Process each audio file individually to avoid batch dimension issues
                    inputs = self.processor(
                        text="<|audio_bos|><|AUDIO|><|audio_eos|>",
                        audios=audio,  # Single audio, not wrapped in list
                        sampling_rate=16000,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    )
                    
                    # Move inputs to device
                    for key in inputs:
                        if isinstance(inputs[key], torch.Tensor):
                            inputs[key] = inputs[key].to(self.model.device)
                    
                    # Extract audio features through the model
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1,  # We don't need generation, just embeddings
                        output_hidden_states=True,
                        return_dict_in_generate=True
                    )
                    
                    # Get embeddings from the encoder hidden states
                    if hasattr(outputs, 'encoder_hidden_states') and outputs.encoder_hidden_states:
                        # Use encoder hidden states if available
                        last_hidden_state = outputs.encoder_hidden_states[-1]
                    else:
                        # Fallback: use decoder hidden states
                        last_hidden_state = outputs.hidden_states[0][-1]  # First generated token's hidden state
                    
                    # Mean pooling over sequence dimension
                    if "attention_mask" in inputs:
                        attention_mask = inputs["attention_mask"]
                        
                        # Ensure attention_mask has the same dimensions as last_hidden_state
                        if attention_mask.dim() == 2 and last_hidden_state.dim() == 3:
                            # Check if sequence lengths match before unsqueezing
                            if attention_mask.shape[1] == last_hidden_state.shape[1]:
                                attention_mask = attention_mask.unsqueeze(-1)  # [B, L] -> [B, L, 1]
                                # Apply attention masking
                                masked_embeddings = last_hidden_state * attention_mask
                                pooled_features = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                            else:
                                # For Qwen2-Audio, attention_mask length often doesn't match audio feature length
                                # This is normal - use mean pooling as fallback
                                pooled_features = last_hidden_state.mean(dim=1)
                        elif attention_mask.dim() != last_hidden_state.dim():
                            # If dimensions still don't match, use mean pooling without masking
                            print(f"[WARN] Attention mask dim {attention_mask.dim()} != hidden state dim {last_hidden_state.dim()}, using mean pooling")
                            pooled_features = last_hidden_state.mean(dim=1)
                        else:
                            # Apply attention masking (dimensions already match)
                            masked_embeddings = last_hidden_state * attention_mask
                            pooled_features = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                    else:
                        pooled_features = last_hidden_state.mean(dim=1)
                    
                    # 保持为torch tensor，不转换为numpy
                    embedding = pooled_features.squeeze()
                    embeddings.append(embedding)
                    
                except Exception as e:
                    print(f"[ERROR] Failed to process audio {audio_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Return zero embedding as fallback
                    dummy_embedding = torch.zeros(4096, device=self.model.device)  # Qwen2-Audio typical hidden size
                    embeddings.append(dummy_embedding)
        
        return torch.stack(embeddings)


class Qwen2AudioTextEncoder:
    """
    Qwen2-Audio Text Encoder wrapper for text-to-embedding
    """
    def __init__(self, model_name="Qwen/Qwen2-Audio-7B-Instruct", device="cuda", shared_model=None):
        self.device = device
        
        if shared_model is not None:
            # Reuse the shared model from speech encoder
            print(f"[INFO] Reusing shared Qwen2-Audio model for text encoder")
            self.processor = shared_model['processor']
            self.model = shared_model['model']
        else:
            # Load new model (fallback for backward compatibility)
            print(f"[INFO] Loading Qwen2-Audio text encoder: {model_name}")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            self.model.eval()
            print(f"[INFO] Qwen2-Audio text encoder loaded successfully on {device}")
    
    def predict(self, texts: List[str], source_lang: str = "eng_Latn") -> torch.Tensor:
        """
        Extract text embeddings from text strings
        
        Args:
            texts: List of text strings
            source_lang: Source language (kept for compatibility, not used in Qwen2-Audio)
            
        Returns:
            torch tensor of shape [batch_size, embedding_dim]
        """
        embeddings = []
        
        # 移除 torch.no_grad() 以保持梯度链
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
                    
                    # 保持为torch tensor，不转换为numpy
                    embedding = pooled_embedding.squeeze()
                    embeddings.append(embedding)
                    
                except Exception as e:
                    print(f"[ERROR] Failed to process text '{text[:50]}...': {e}")
                    # Return zero embedding as fallback
                    dummy_embedding = torch.zeros(4096, device=self.model.device)  # Qwen2-Audio typical hidden size
                    embeddings.append(dummy_embedding)
        
        return torch.stack(embeddings)


class ContrastiveQwen2AudioModel(nn.Module):
    """
    Contrastive Speech-Text Model using Qwen2-Audio encoders with LoRA fine-tuning
    """
    def __init__(self, speech_encoder, text_encoder, hidden_dim=4096, proj_dim=512, 
                 lora_r=16, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        self.speech_encoder = speech_encoder
        self.text_encoder = text_encoder
        
        # Projection layers (always trainable)
        self.proj_speech = nn.Linear(hidden_dim, proj_dim)
        self.proj_text = nn.Linear(hidden_dim, proj_dim)
        
        # 应用LoRA到共享的language model
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,  # rank
            lora_alpha=lora_alpha,  # scaling parameter
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Qwen2典型的attention和MLP层
            bias="none",
        )
        
        # 冻结所有原始参数
        for param in self.speech_encoder.model.parameters():
            param.requires_grad = False
        for param in self.text_encoder.model.parameters():
            param.requires_grad = False
        
        # 应用LoRA（只应用一次，因为模型是共享的）
        if self.speech_encoder.model is self.text_encoder.model:
            print(f"[INFO] Applying LoRA to shared Qwen2-Audio model")
            self.speech_encoder.model = get_peft_model(self.speech_encoder.model, self.lora_config)
            self.text_encoder.model = self.speech_encoder.model  # 保持共享
        else:
            print(f"[INFO] Applying LoRA to separate speech and text models")
            self.speech_encoder.model = get_peft_model(self.speech_encoder.model, self.lora_config)
            self.text_encoder.model = get_peft_model(self.text_encoder.model, self.lora_config)
        
        # 计算LoRA参数数量（需要在应用LoRA之后）
        self.actual_lora_params = sum(p.numel() for p in self.speech_encoder.model.parameters() if p.requires_grad)
        if self.speech_encoder.model is not self.text_encoder.model:
            self.actual_lora_params += sum(p.numel() for p in self.text_encoder.model.parameters() if p.requires_grad)
        
        print(f"[INFO] ContrastiveQwen2AudioModel initialized with LoRA (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})")
        print(f"[INFO] LoRA trainable parameters: {self.actual_lora_params:,}")
    
    def get_trainable_parameters(self):
        """获取可训练参数数量和详情"""
        lora_params = 0
        proj_params = 0
        
        # 计算LoRA参数
        for name, param in self.speech_encoder.model.named_parameters():
            if param.requires_grad:
                lora_params += param.numel()
        
        # 如果不是共享模型，还要计算text encoder的LoRA参数
        if self.speech_encoder.model is not self.text_encoder.model:
            for name, param in self.text_encoder.model.named_parameters():
                if param.requires_grad:
                    lora_params += param.numel()
        
        # 计算投影层参数
        proj_params = sum(p.numel() for p in self.proj_speech.parameters()) + \
                     sum(p.numel() for p in self.proj_text.parameters())
        
        return {
            'lora_params': lora_params,
            'proj_params': proj_params,
            'total_trainable': lora_params + proj_params
        }
    
    def encode_audio(self, audio_paths: List[str]) -> torch.Tensor:
        """Encode audio files to embeddings"""
        # 在训练模式下保持梯度，评估模式下断链
        if self.training:
            speech_embeddings = self.speech_encoder.predict(audio_paths)  # [B, hidden_dim]
        else:
            with torch.no_grad():
                speech_embeddings = self.speech_encoder.predict(audio_paths)  # [B, hidden_dim]
        
        # 确保数据类型为float32并移动到正确设备
        if not isinstance(speech_embeddings, torch.Tensor):
            speech_embeddings = torch.from_numpy(speech_embeddings)
        speech_embeddings = speech_embeddings.float().to(self.proj_speech.weight.device)
        
        return F.normalize(self.proj_speech(speech_embeddings), dim=-1)
    
    def encode_text(self, texts: List[str], source_lang: str = "eng_Latn") -> torch.Tensor:
        """Encode text strings to embeddings"""
        # 在训练模式下保持梯度，评估模式下断链
        if self.training:
            text_embeddings = self.text_encoder.predict(texts, source_lang=source_lang)
        else:
            with torch.no_grad():
                text_embeddings = self.text_encoder.predict(texts, source_lang=source_lang)
        
        # 确保数据类型为float32并移动到正确设备
        if not isinstance(text_embeddings, torch.Tensor):
            text_embeddings = torch.from_numpy(text_embeddings)
        text_embeddings = text_embeddings.float().to(self.proj_text.weight.device)
        
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
            # 确保数据类型为float32，但保持梯度（训练时）或断链（评估时）
            embeddings = embeddings.float()
            if not model.training:
                embeddings = embeddings.detach()
            all_embeddings.append(embeddings)
        except Exception as e:
            print(f"[ERROR] Failed to encode text batch {i//batch_size}: {e}")
            # Create dummy embeddings
            dummy_emb = torch.zeros(len(batch_texts), 512, dtype=torch.float32, device=device)
            all_embeddings.append(dummy_emb)
    
    return torch.cat(all_embeddings, dim=0)


def encode_audios_in_batches(model, audio_paths, batch_size=32, device="cuda"):
    """Encode audios in batches using the model's audio encoder"""
    all_embeddings = []
    
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i:i + batch_size]
        try:
            embeddings = model.encode_audio(batch_paths)
            # 确保数据类型为float32，但保持梯度（训练时）或断链（评估时）
            embeddings = embeddings.float()
            if not model.training:
                embeddings = embeddings.detach()
            all_embeddings.append(embeddings)
        except Exception as e:
            print(f"[ERROR] Failed to encode audio batch {i//batch_size}: {e}")
            # Create dummy embeddings
            dummy_emb = torch.zeros(len(batch_paths), 512, dtype=torch.float32, device=device)
            all_embeddings.append(dummy_emb)
    
    return torch.cat(all_embeddings, dim=0)
