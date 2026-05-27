"""
Streaming Term RAG Retriever for Online Speech Translation

This module implements a streaming RAG retriever that:
1. Accumulates audio incrementally (every ~120ms from SimulEval)
2. Uses sliding window with configurable chunk_size and hop_size
3. Aggregates terms across windows using max score pooling
4. Filters results by top-N (based on audio duration) and score threshold

Decouples RAG processing from vLLM calls:
- RAG: called every SimulEval step (~120ms)
- vLLM: called only when accumulated audio reaches segment size (e.g., 960ms)
"""

import os
import json
import math
import pickle
import logging
from typing import Optional, List, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    faiss = None
    logger.warning("FAISS not available; RAG retriever will be disabled")


class ProcessorAudioAlias:
    """Wrapper to handle 'audios' vs 'audio' parameter naming."""
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, *args, **kwargs):
        if "audios" in kwargs and "audio" not in kwargs:
            kwargs["audio"] = kwargs.pop("audios")
        return self.processor(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self.processor, item)


class TokenizerKwCleaner:
    """Wrapper to remove audio-related kwargs from tokenizer calls."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, *args, **kwargs):
        kwargs.pop("audio", None)
        kwargs.pop("audios", None)
        return self.tokenizer(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self.tokenizer, item)


def l2_distance_to_score(distance: float) -> float:
    """
    Convert FAISS L2 distance to a similarity score.
    
    Uses score = 1 / (1 + distance) so that:
    - score is in (0, 1]
    - higher is better
    - max score corresponds to min distance
    
    This is consistent with eval_local_sliding_window_acl6060_v2.py
    """
    d = float(distance)
    if not np.isfinite(d) or d < 0:
        return 0.0
    return 1.0 / (1.0 + d)


class StreamingTermRAGRetriever:
    """
    Streaming RAG retriever with sliding window support for online SST.
    
    Key features:
    - Incremental audio accumulation
    - Sliding window chunking (chunk_size, hop_size)
    - Max score pooling across windows
    - Top-N filtering based on audio duration
    - Score threshold filtering
    
    Decoupling strategy:
    - SimulEval calls `accumulate_audio()` every ~120ms
    - `get_current_references()` returns accumulated RAG results
    - vLLM calls only use the accumulated references when needed
    """
    
    def __init__(
        self,
        index_path: Optional[str],
        model_path: Optional[str],
        base_model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
        device: str = "cuda:0",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        # Retrieval parameters
        top_k: int = 5,
        target_lang: str = "zh",
        score_threshold: float = 0.5,
        # Sliding window parameters (consistent with eval_local)
        chunk_size: float = 1.92,  # seconds
        hop_size: float = 0.96,    # seconds (sliding window step)
        # Top-N filtering parameters
        terms_per_second: float = 2.5,
        enable_top_n_filter: bool = True,
        # Audio parameters
        sample_rate: int = 16000,
        # Batch processing
        batch_size: int = 32,
    ):
        """
        Initialize the streaming RAG retriever.
        
        Args:
            index_path: Path to prebuilt FAISS index (.pkl)
            model_path: Path to trained contrastive model checkpoint
            base_model_name: Base model name for speech encoder
            device: Device for model inference
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout (disabled in practice for FP16 stability)
            top_k: Number of terms to retrieve per chunk
            target_lang: Target language for translations (e.g., 'zh')
            score_threshold: Minimum score to include a term in results
            chunk_size: Sliding window size in seconds
            hop_size: Sliding window step in seconds
            terms_per_second: Terms to keep per second of audio (for top-N)
            enable_top_n_filter: Whether to enable top-N filtering
            sample_rate: Audio sample rate
            batch_size: Batch size for encoding
        """
        self.enabled = False
        self.index = None
        self.term_list: List[Dict[str, object]] = []
        self.embedding_dim = 512
        self.device_str = device
        
        # Set device
        if device and device.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")
            if device and device.startswith("cuda"):
                logger.warning("CUDA unavailable, falling back to CPU for RAG retriever")
        
        # Retrieval parameters
        self.top_k = top_k
        self.target_lang = target_lang.lower() if target_lang else "zh"
        self.score_threshold = float(max(0.0, min(1.0, score_threshold)))
        
        # Sliding window parameters
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_size * sample_rate)
        self.hop_samples = int(hop_size * sample_rate)
        
        # Top-N filtering
        self.terms_per_second = terms_per_second
        self.enable_top_n_filter = enable_top_n_filter
        
        # Batch processing
        self.batch_size = batch_size
        
        # Model
        self.model = None
        self.speech_encoder = None
        
        # === Streaming state ===
        # Audio buffer: accumulates samples between vLLM calls
        self._audio_buffer: np.ndarray = np.array([], dtype=np.float32)
        # Window position: tracks which windows have been processed
        self._processed_up_to: int = 0
        # Accumulated term scores: term (lowercase) -> max score across all windows
        self._term_scores: Dict[str, float] = {}
        # Canonical term surface form (preserve original case/spaces from glossary)
        # term_lc -> canonical_term
        self._term_canonical: Dict[str, str] = {}
        # Last processed chunk index for incremental processing
        self._last_chunk_idx: int = -1
        # Track if this is the first retrieval (for cold start handling)
        self._is_first_retrieval: bool = True
        
        # Check dependencies
        if faiss is None:
            logger.warning("FAISS not available; disabling RAG retriever")
            return
            
        if not index_path or not os.path.exists(index_path):
            logger.warning("RAG index path missing; disabling RAG retriever")
            return
            
        # Load index
        try:
            self._load_index(index_path)
        except Exception as exc:
            logger.exception("Failed to load RAG index from %s: %s", index_path, exc)
            return
        
        # Load model
        if not model_path or not os.path.exists(model_path):
            logger.warning("RAG model checkpoint not found at %s; disabling RAG retriever", model_path)
            return
            
        try:
            self._load_model(
                model_path=model_path,
                base_model_name=base_model_name,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        except Exception as exc:
            logger.exception("Failed to load RAG model: %s", exc)
            self.index = None
            self.term_list = []
            return
        
        self.enabled = self.index is not None and self.model is not None
        if self.enabled:
            logger.info(
                "StreamingTermRAGRetriever initialized: "
                "%d terms, embedding_dim=%d, chunk_size=%.1fs, hop_size=%.1fs, "
                "top_k=%d, terms_per_second=%.1f, score_threshold=%.2f",
                len(self.term_list),
                self.embedding_dim,
                self.chunk_size,
                self.hop_size,
                self.top_k,
                self.terms_per_second,
                self.score_threshold,
            )
    
    def _load_index(self, index_path: str):
        """Load FAISS index and term list from pickle file."""
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        
        serialized_index = data.get("faiss_index")
        if isinstance(serialized_index, bytes):
            serialized_index = np.frombuffer(serialized_index, dtype=np.uint8)
        elif isinstance(serialized_index, bytearray):
            serialized_index = np.frombuffer(bytes(serialized_index), dtype=np.uint8)
        elif isinstance(serialized_index, np.ndarray):
            if serialized_index.dtype != np.uint8:
                serialized_index = serialized_index.astype(np.uint8)
        else:
            raise ValueError(f"Unsupported faiss_index type: {type(serialized_index)}")
        
        self.index = faiss.deserialize_index(serialized_index)
        self.term_list = data.get("term_list", [])
        self.embedding_dim = int(data.get("embedding_dim", 512))

        # Strict mode: require new index format with explicit 'key' and original-cased 'term'.
        # We do NOT attempt to support legacy indices; fail fast to avoid silent recall/prompt issues.
        if not isinstance(self.term_list, list) or not self.term_list:
            raise ValueError("Invalid index: term_list must be a non-empty list")
        for i, entry in enumerate(self.term_list):
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid index: term_list[{i}] must be a dict, got {type(entry)}")
            key = entry.get("key")
            term = entry.get("term")
            if not isinstance(key, str) or not key.strip() or key != key.strip().lower():
                raise ValueError(
                    f"Invalid index: term_list[{i}]['key'] must be non-empty lowercase string, got {key!r}"
                )
            if not isinstance(term, str) or not term.strip():
                raise ValueError(
                    f"Invalid index: term_list[{i}]['term'] must be non-empty string, got {term!r}"
                )
        
        logger.info("Loaded index with %d vectors, %d terms", 
                   self.index.ntotal, len(self.term_list))
    
    def _load_model(
        self,
        model_path: str,
        base_model_name: str,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
    ):
        """Load the contrastive model for audio encoding."""
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
        from peft import LoraConfig, get_peft_model, TaskType
        from retriever.gigaspeech.modal.Qwen2_Audio_train import Qwen2AudioSpeechEncoder
        
        # Load processor
        processor = AutoProcessor.from_pretrained(base_model_name)
        processor.tokenizer = TokenizerKwCleaner(processor.tokenizer)
        processor = ProcessorAudioAlias(processor)
        
        # Load base model
        base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
        ).to(self.device)
        base_model.eval()
        
        # Configure LoRA (consistent with training)
        target_modules = ["q_proj", "k_proj", "v_proj"]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,  # Disabled for FP16 stability
            target_modules=target_modules,
            bias="none",
        )
        base_model = get_peft_model(base_model, lora_config)
        base_model.eval()
        
        # Initialize speech encoder (bypass __init__ for custom setup)
        speech_encoder = Qwen2AudioSpeechEncoder.__new__(Qwen2AudioSpeechEncoder)
        speech_encoder.device = self.device
        speech_encoder.model_name = base_model_name
        speech_encoder.processor = processor
        speech_encoder.model = base_model
        
        # Set encoding strategy (audio_tower for Qwen2-Audio)
        speech_encoder.has_audio_tower = True
        speech_encoder.audio_tower_name = 'audio_tower'
        speech_encoder.encoding_strategy = 'audio_tower'
        speech_encoder.audio_tower_output_type = 'BaseModelOutput'
        
        # Get audio hidden dimension from config
        if hasattr(base_model, 'base_model'):
            underlying_model = base_model.base_model.model
        else:
            underlying_model = base_model
        
        if hasattr(underlying_model.config, 'audio_config'):
            speech_encoder.audio_hidden_dim = underlying_model.config.audio_config.d_model
        else:
            speech_encoder.audio_hidden_dim = 1280
            logger.warning("Could not find audio_config, using default audio_hidden_dim=1280")
        
        speech_encoder.hidden_size = speech_encoder.audio_hidden_dim
        
        if hasattr(underlying_model, 'language_model'):
            speech_encoder.language_model_hidden_dim = underlying_model.language_model.config.hidden_size
        else:
            speech_encoder.language_model_hidden_dim = underlying_model.config.hidden_size
        
        # Suppress debug logging
        speech_encoder._logged_input_grad = True
        speech_encoder._logged_lora_layer_status = True
        speech_encoder._logged_peft_access = True
        speech_encoder._logged_gradient_debug = True
        speech_encoder._logged_pooled_debug = True
        
        speech_hidden = speech_encoder.get_hidden_size()
        logger.info("RAG speech encoder hidden_size: %d", speech_hidden)
        
        # Create simple contrastive model wrapper
        class SimpleContrastiveModel(nn.Module):
            def __init__(self, speech_encoder, speech_hidden_dim, proj_dim, device):
                super().__init__()
                self.speech_encoder = speech_encoder
                self.proj_speech = nn.Linear(speech_hidden_dim, proj_dim).to(device)
            
            def encode_audio(self, audio_inputs):
                with torch.no_grad():
                    emb = self.speech_encoder.predict(audio_inputs)
                if not isinstance(emb, torch.Tensor):
                    emb = torch.as_tensor(emb)
                emb = emb.float().to(self.proj_speech.weight.device)
                if emb.dim() == 3:
                    emb = emb.mean(dim=1)
                return F.normalize(self.proj_speech(emb), dim=-1)
        
        model = SimpleContrastiveModel(
            speech_encoder,
            speech_hidden_dim=speech_hidden,
            proj_dim=self.embedding_dim,
            device=self.device,
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        if state_dict:
            first_key = next(iter(state_dict))
            if first_key.startswith("module."):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        # Separate projection and LoRA weights
        proj_state = {}
        lora_state = {}
        for key, value in state_dict.items():
            if key.startswith("proj_speech"):
                proj_state[key] = value
            elif key.startswith("proj_text"):
                continue
            elif "lora_" in key or "base_model" in key:
                if key.startswith("speech_qwen2_model.") or key.startswith("text_qwen2_model."):
                    new_key = key.split(".", 1)[1] if "." in key else key
                    lora_state[new_key] = value
                else:
                    lora_state[key] = value
        
        # Load projection weights
        if proj_state:
            filtered_proj = {k: v for k, v in proj_state.items() if k.startswith("proj_speech")}
            if "proj_speech.weight" in filtered_proj:
                loaded_in_dim = filtered_proj["proj_speech.weight"].shape[1]
                if loaded_in_dim != speech_hidden:
                    logger.error(
                        "proj_speech dimension mismatch! Loaded: %d, Expected: %d",
                        loaded_in_dim, speech_hidden
                    )
                else:
                    logger.info("proj_speech dimension check passed: %d", loaded_in_dim)
            model.load_state_dict(filtered_proj, strict=False)
        
        # Load LoRA weights
        if lora_state:
            lora_a_count = sum(1 for k in lora_state if 'lora_A' in k)
            lora_b_count = sum(1 for k in lora_state if 'lora_B' in k)
            logger.info("Loading %d LoRA_A and %d LoRA_B parameters", lora_a_count, lora_b_count)
            base_model.load_state_dict(lora_state, strict=False)
        
        self.model = model.eval()
        self.speech_encoder = speech_encoder
    
    def reset(self):
        """
        Reset streaming state for a new utterance/session.
        
        Call this when:
        - Starting a new source audio
        - SimulEval resets states
        """
        self._audio_buffer = np.array([], dtype=np.float32)
        self._processed_up_to = 0
        self._term_scores = {}
        self._term_canonical = {}
        self._last_chunk_idx = -1
        self._is_first_retrieval = True  # Track if this is the first RAG retrieval
        logger.debug("StreamingRAG state reset")
    
    def accumulate_audio(
        self,
        audio_chunk: Union[np.ndarray, torch.Tensor],
        force_process: bool = False,
    ) -> List[Dict[str, str]]:
        """
        Accumulate audio and process new sliding windows.
        
        This is the main entry point called by SimulEval every ~120ms.
        
        Args:
            audio_chunk: New audio samples to add (numpy array or torch tensor)
            force_process: If True, process even if not enough for a full window
                          (used at source_finished)
        
        Returns:
            Current list of retrieved references (empty by design)
        """
        if not self.enabled:
            return []
        
        # Convert to numpy if needed
        if isinstance(audio_chunk, torch.Tensor):
            audio_chunk = audio_chunk.detach().cpu().numpy()
        
        audio_chunk = np.asarray(audio_chunk, dtype=np.float32).flatten()
        
        if audio_chunk.size == 0:
            return self.get_current_references()
        
        # Append to buffer
        self._audio_buffer = np.concatenate([self._audio_buffer, audio_chunk])
        
        # Process new windows
        self._process_new_windows(force_process=force_process)
        
        # Note: caller should use get_current_references() with increment_samples for proper top-N
        return []  # Return empty here; caller should call get_current_references() explicitly
    
    def _process_new_windows(self, force_process: bool = False):
        """
        Process any new sliding windows that can be formed.
        
        Uses incremental processing: only processes windows that haven't been
        processed yet.
        
        Args:
            force_process: If True, process partial windows at the end (for source_finished)
        """
        if self.model is None or self.index is None:
            return
        
        buffer_len = len(self._audio_buffer)
        
        # Calculate which windows we can process
        # Window i starts at i * hop_samples and ends at i * hop_samples + chunk_samples
        chunks_to_process = []
        chunk_idx = self._last_chunk_idx + 1
        
        while True:
            start = chunk_idx * self.hop_samples
            end = start + self.chunk_samples
            
            if end <= buffer_len:
                # Full window available
                chunk = self._audio_buffer[start:end]
                chunks_to_process.append((chunk_idx, chunk))
                chunk_idx += 1
            elif force_process and start < buffer_len:
                # Partial window at the end (pad to chunk_size)
                chunk = self._audio_buffer[start:]
                if len(chunk) < self.chunk_samples:
                    chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)), mode='constant')
                chunks_to_process.append((chunk_idx, chunk))
                chunk_idx += 1
                break
            else:
                # Not enough samples for next window
                break
        
        if not chunks_to_process:
            return
        
        # Process chunks in batches
        all_chunks = [c[1] for c in chunks_to_process]
        self._retrieve_and_aggregate(all_chunks)
        
        # Update last processed index
        self._last_chunk_idx = chunks_to_process[-1][0]
    
    def _retrieve_and_aggregate(self, chunks: List[np.ndarray]):
        """
        Encode audio chunks and retrieve terms, aggregating with max pooling.
        
        Args:
            chunks: List of audio chunks (numpy arrays)
        """
        if not chunks:
            return
        
        # Encode in batches
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]
            
            try:
                # Convert to list of numpy arrays for encoder
                audio_inputs = [c.astype(np.float32) for c in batch_chunks]
                
                with torch.no_grad():
                    # Encode batch
                    embeddings = self.model.encode_audio(audio_inputs)
                    if isinstance(embeddings, torch.Tensor):
                        embeddings = embeddings.detach().cpu().float().numpy()
                
                # Search for each embedding
                for emb in embeddings:
                    emb = emb.reshape(1, -1)
                    D, I = self.index.search(emb, self.top_k)
                    
                    for idx, dist in zip(I[0], D[0]):
                        if 0 <= idx < len(self.term_list):
                            term_entry = self.term_list[idx]
                            if not isinstance(term_entry, dict):
                                raise ValueError(f"Invalid term entry type: {type(term_entry)}")

                            # Strict: require explicit key + term.
                            term_lc = term_entry.get("key")
                            term_surface = term_entry.get("term")
                            if not isinstance(term_lc, str) or not term_lc.strip():
                                raise ValueError(f"Missing/invalid term key at term_list[{idx}]: {term_lc!r}")
                            term_lc = term_lc.strip().lower()
                            if not isinstance(term_surface, str) or not term_surface.strip():
                                raise ValueError(f"Missing/invalid term surface at term_list[{idx}]: {term_surface!r}")
                            term_surface = term_surface.strip()

                            score = l2_distance_to_score(dist)
                            
                            # Max pooling: keep highest score for each term
                            if term_lc not in self._term_scores or score > self._term_scores[term_lc]:
                                self._term_scores[term_lc] = score
                                # Keep the surface form aligned with the best score
                                self._term_canonical[term_lc] = term_surface
            
            except Exception as e:
                logger.warning("Failed to process audio batch: %s", e)
                continue
    
    def get_current_references(
        self,
        min_terms: int = 5,
    ) -> List[Dict[str, str]]:
        """
        Get current accumulated references with filtering applied.
        
        CRITICAL: This method returns Top-N terms for the current vLLM call,
        then CLEARS the internal state (_term_scores) to avoid accumulation.
        
        Args:
            min_terms: Minimum number of terms to keep (default: 5)
        
        Returns:
            List of dicts with:
            - 'key': canonical lowercase term key (used internally for matching)
            - 'term': original-cased surface form (for LLM prompting)
            - 'translation': target language translation
        """
        if not self.enabled or not self._term_scores:
            return []
        
        # Mark that we've done at least one retrieval
        self._is_first_retrieval = False
        
        # Use top_k for filtering
        n = max(min_terms, self.top_k)
        
        # Sort by score and take top N
        sorted_terms = sorted(self._term_scores.items(), key=lambda x: x[1], reverse=True)
        filtered_scores = dict(sorted_terms[:n])
        
        # 3. Apply score threshold and build results
        results: List[Dict[str, str]] = []
        candidate_logs: List[Dict[str, object]] = []
        
        # Sort by score descending
        sorted_items = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
        
        for term_lc, score in sorted_items:
            candidate_logs.append({
                "key": term_lc,
                "term": self._term_canonical.get(term_lc) or term_lc,
                "score": round(score, 4),
            })
            
            if score < self.score_threshold:
                continue
            
            # Find translation
            translation = self._get_translation(term_lc)
            term_surface = self._term_canonical.get(term_lc) or term_lc
            results.append({"key": term_lc, "term": term_surface, "translation": translation})
        
        # Log candidates
        if candidate_logs:
            logger.debug(
                "RAG candidates (threshold=%.2f, top_n=%d): %s",
                self.score_threshold,
                len(filtered_scores),
                json.dumps(candidate_logs[:10], ensure_ascii=False),  # Log first 10
            )
        
        # 4. [CRITICAL] Reset term scores after returning results
        self._term_scores = {}
        self._term_canonical = {}
        
        return results
    
    def _get_translation(self, term_lc: str) -> str:
        """Get translation for a term from term_list."""
        for entry in self.term_list:
            if not isinstance(entry, dict):
                continue
            entry_key = entry.get("key")
            if isinstance(entry_key, str) and entry_key.strip().lower() == term_lc:
                translations = entry.get("target_translations") or {}
                if isinstance(translations, dict):
                    return (
                        translations.get(self.target_lang) or
                        translations.get(self.target_lang.upper()) or
                        ""
                    )
        raise KeyError(f"Translation not found for term key: {term_lc}")
    
    def get_audio_duration(self) -> float:
        """Get current accumulated audio duration in seconds."""
        return len(self._audio_buffer) / self.sample_rate
    
    def has_enough_audio(self) -> bool:
        """Check if we have enough audio for at least one full window."""
        return len(self._audio_buffer) >= self.chunk_samples
    
    
    def _create_sliding_windows(self, audio: np.ndarray) -> List[np.ndarray]:
        """Create sliding window chunks from audio (consistent with eval_local)."""
        chunks = []
        start = 0
        
        while start < len(audio):
            end = min(start + self.chunk_samples, len(audio))
            chunk = audio[start:end]
            
            # Pad if necessary
            if len(chunk) < self.chunk_samples:
                chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)), mode='constant')
            
            chunks.append(chunk)
            start += self.hop_samples
            
            if start >= len(audio):
                break
        
        return chunks
    
    def retrieve_direct(
        self,
        audio: Union[np.ndarray, List[np.ndarray]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Direct retrieval without sliding window (for batch processing).
        
        This method bypasses the streaming/sliding window logic and directly
        retrieves terms for the given audio chunk(s).
        
        Args:
            audio: Single audio array or list of audio arrays
            top_k: Number of terms to retrieve (default: use self.top_k)
        
        Returns:
            List of dicts with:
            - 'key': canonical lowercase term key
            - 'term': original-cased surface form
            - 'translation': target language translation
            - 'score': retrieval score (0-1)
        """
        if not self.enabled:
            return []
        
        # Normalize input to list
        if isinstance(audio, np.ndarray):
            audio_list = [audio]
        else:
            audio_list = audio
        
        if not audio_list:
            return []
        
        k = top_k if top_k is not None else self.top_k
        
        # Encode audio
        try:
            audio_inputs = [a.astype(np.float32) for a in audio_list]
            
            with torch.no_grad():
                embeddings = self.model.encode_audio(audio_inputs)
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.detach().cpu().float().numpy()
            
            # Search index
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            D, I = self.index.search(embeddings, k)
            
            # Aggregate results across all audio chunks (max pooling by term)
            term_scores = {}  # term_key -> (score, term_surface, translation)
            
            for distances, indices in zip(D, I):
                for dist, idx in zip(distances, indices):
                    if idx < 0 or idx >= len(self.term_list):
                        continue
                    
                    term_entry = self.term_list[idx]
                    if not isinstance(term_entry, dict):
                        continue
                    
                    # Get term key and surface form
                    term_key = term_entry.get("key", "").strip().lower()
                    term_surface = term_entry.get("term", "").strip()
                    
                    if not term_key or not term_surface:
                        continue
                    
                    # Get translation
                    translations = term_entry.get("translations", {})
                    translation = translations.get(self.target_lang, "")
                    
                    if not translation:
                        continue
                    
                    # Convert L2 distance to score
                    score = l2_distance_to_score(dist)
                    
                    # Max pooling: keep highest score for each term
                    if term_key not in term_scores or score > term_scores[term_key][0]:
                        term_scores[term_key] = (score, term_surface, translation)
            
            # Sort by score and apply score threshold
            results = []
            for term_key, (score, term_surface, translation) in term_scores.items():
                if score >= self.score_threshold:
                    results.append({
                        'key': term_key,
                        'term': term_surface,
                        'translation': translation,
                        'score': score
                    })
            
            # Sort by score descending
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # Return top-k
            return results[:k]
        
        except Exception as e:
            logger.warning("Failed to retrieve terms: %s", e)
            return []

