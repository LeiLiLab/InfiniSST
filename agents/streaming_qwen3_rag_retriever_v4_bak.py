import os
import json
import math
import pickle
import logging
from time import perf_counter
from typing import Optional, List, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperFeatureExtractor, AutoTokenizer, AutoModel
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    faiss = None

def l2_distance_to_score(similarity: float) -> float:
    # BGE-M3 index uses Inner Product (IP), and we normalized L2.
    # For L2 normalized vectors: L2_dist^2 = 2 - 2 * Cosine_Similarity
    l2_dist = math.sqrt(max(0, 2 - 2 * float(similarity)))
    return 1.0 / (1.0 + l2_dist)

class AttentivePooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x, mask=None):
        scores = self.attention(x) # [B, T, 1]
        if mask is not None:
            mask = mask.unsqueeze(-1)
            scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=1)
        pooled = torch.sum(x * weights, dim=1)
        return pooled

class BgeM3TextEncoder(nn.Module):
    def __init__(self, model_id="BAAI/bge-m3", lora_rank=16):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            add_pooling_layer=False
        )
        
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=["query", "key", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type=None
        )
        self.encoder = get_peft_model(self.encoder, lora_config)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return F.normalize(embeddings, p=2, dim=-1)

class Qwen3OmniRetriever(nn.Module):
    def __init__(self, model_id="Atotti/Qwen3-Omni-AudioTransformer", target_dim=1024, 
                 use_lora=True, lora_rank=32, lora_alpha=64, lora_target_modules=None):
        super().__init__()
        
        self.audio_encoder = Qwen3OmniMoeAudioEncoder.from_pretrained(
            model_id, 
            dtype=torch.bfloat16
        )
        
        if hasattr(self.audio_encoder, "conv2d1"):
            self.audio_encoder.get_input_embeddings = lambda: self.audio_encoder.conv2d1
            
        self.audio_encoder.gradient_checkpointing_enable()
        
        if use_lora:
            if lora_target_modules is None:
                lora_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "proj1", "proj2"]
            
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type=None
            )
            self.audio_encoder = get_peft_model(self.audio_encoder, lora_config)
            
        self.pooler = AttentivePooling(2048) 
        self.projector = nn.Linear(2048, target_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, input_features, feature_lens):
        # The Qwen3 audio encoder expects "packed" 2D features: [C, sum(T_i)] and uses
        # `feature_lens` to split along the time axis internally.
        #
        # In some callers (e.g. offline ACL simulation), we receive 3D features:
        #   input_features: [B, C, T]
        # We must pack them to avoid shape/length mismatch in `split_with_sizes`.
        if input_features.ndim == 3:
            # [B, C, T] -> [C, B*T]
            input_features = input_features.transpose(0, 1).reshape(input_features.shape[1], -1)

        outputs = self.audio_encoder(input_features, feature_lens)
        hidden_states = outputs.last_hidden_state
        
        if hidden_states.ndim == 2:
            output_lens = []
            for l in feature_lens.tolist():
                curr_l = l
                for _ in range(3):
                    curr_l = (curr_l + 1) // 2
                output_lens.append(curr_l)
            
            if sum(output_lens) != hidden_states.shape[0]:
                ratio = input_features.shape[1] / hidden_states.shape[0]
                output_lens = [max(1, round(l / ratio)) for l in feature_lens.tolist()]
                output_lens[-1] = hidden_states.shape[0] - sum(output_lens[:-1])

            hidden_states_list = torch.split(hidden_states, output_lens, dim=0)
            from torch.nn.utils.rnn import pad_sequence
            hidden_states = pad_sequence(hidden_states_list, batch_first=True)
            feature_lens = torch.tensor(output_lens, device=hidden_states.device)
            
        batch_size, max_len, _ = hidden_states.shape
        mask = torch.arange(max_len, device=hidden_states.device).expand(batch_size, max_len) < feature_lens.unsqueeze(1)
        
        pooled_audio = self.pooler(hidden_states, mask)
        projected = self.projector(pooled_audio)
        return F.normalize(projected, p=2, dim=-1)

class StreamingQwen3RAGRetrieverV4:
    def __init__(
        self,
        index_path: str,
        model_path: str,
        base_model_name: str = "Atotti/Qwen3-Omni-AudioTransformer",
        device: str = "cuda:1",
        lora_r: int = 32,
        lora_alpha: int = 64,
        text_lora_r: int = 16,
        top_k: int = 5,
        voting_k: int = 20,
        voting_min_votes: int = 2,
        target_lang: str = "zh",
        score_threshold: float = 0.5,
        chunk_size: float = 1.92,
        hop_size: float = 0.96,
        aggregation_strategy: str = "max_pool", 
        sample_rate: int = 16000,
        debug_audio_dir: Optional[str] = None,
        verbose: bool = True,
        faiss_use_gpu: bool = False,
    ):
        self.device = torch.device(device)
        self.top_k = top_k
        self.voting_k = voting_k
        self.voting_min_votes = voting_min_votes
        self.target_lang = target_lang
        self.score_threshold = score_threshold
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.aggregation_strategy = aggregation_strategy.lower()
        self.sample_rate = sample_rate
        self.verbose = verbose
        self.chunk_samples = int(chunk_size * sample_rate)
        self.hop_samples = int(hop_size * sample_rate)
        
        self.buffer_max_size = max(1, round(chunk_size / hop_size))
        self._window_results_buffer = []
        
        self.debug_audio_dir = debug_audio_dir
        if self.debug_audio_dir:
            os.makedirs(self.debug_audio_dir, exist_ok=True)
        self._rag_call_count = 0

        self._audio_buffer = np.array([], dtype=np.float32)
        self._last_chunk_idx = -1
        self._term_scores = {}
        self._term_canonical = {}
        self._is_first_retrieval = True

        # Timing (for RTF / profiling; seconds)
        self.last_retrieve_time_sec: float = 0.0
        self.total_retrieve_time_sec: float = 0.0
        self.total_retrieve_calls: int = 0
        # Detailed timing (seconds) for last retrieval call
        self.last_feat_time_sec: float = 0.0
        self.last_audio_enc_time_sec: float = 0.0
        self.last_faiss_time_sec: float = 0.0
        self.faiss_use_gpu: bool = bool(faiss_use_gpu)
        
        # Load index
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        self.index = faiss.deserialize_index(data["faiss_index"])
        self.term_list = data["term_list"]
        self.term_map = {item["key"]: item for item in self.term_list}

        # Optional: move FAISS index to GPU if faiss-gpu is available.
        # This is a best-effort optimization; we fall back to CPU if unavailable.
        if self.faiss_use_gpu:
            try:
                if faiss is None:
                    raise RuntimeError("faiss is not available")
                if not hasattr(faiss, "StandardGpuResources"):
                    raise RuntimeError("faiss-gpu is not available in this environment")
                res = faiss.StandardGpuResources()
                # Use cuda device index within current process (after CUDA_VISIBLE_DEVICES remapping).
                cuda_id = int(self.device.index or 0)
                self.index = faiss.index_cpu_to_gpu(res, cuda_id, self.index)
                logger.info("FAISS index moved to GPU (cuda:%s).", cuda_id)
            except Exception as e:
                logger.warning("FAISS GPU requested but unavailable; falling back to CPU. Reason: %s", e)
                self.faiss_use_gpu = False
        
        # Load Audio Retriever Model
        self.model = Qwen3OmniRetriever(
            model_id=base_model_name,
            target_dim=1024,
            use_lora=True,
            lora_rank=lora_r,
            lora_alpha=lora_alpha
        ).to(self.device).to(torch.bfloat16)
        
        # Load Text Encoder (v4 specific: tuned text encoder)
        self.text_encoder = BgeM3TextEncoder(
            model_id="BAAI/bge-m3",
            lora_rank=text_lora_r
        ).to(self.device).to(torch.bfloat16)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load Audio State
        state_dict = checkpoint["model_state_dict"]
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict, strict=False)
        
        # Load Text State (v4 specific)
        if "text_model_state_dict" in checkpoint:
            text_state_dict = checkpoint["text_model_state_dict"]
            new_text_state_dict = {k.replace("module.", ""): v for k, v in text_state_dict.items()}
            self.text_encoder.load_state_dict(new_text_state_dict, strict=False)
            logger.info("Successfully loaded tuned Text Encoder from checkpoint.")
        else:
            logger.warning("No text_model_state_dict found in checkpoint, using base BGE-M3.")

        self.model.eval()
        self.text_encoder.eval()
        
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
        self.enabled = True

    def reset(self):
        if self.verbose:
            print("\n[RAG] 🔄 Resetting RAG Retriever buffer and scores...")
        self._audio_buffer = np.array([], dtype=np.float32)
        self._last_chunk_idx = -1
        self._term_scores = {}
        self._term_canonical = {}
        self._window_results_buffer = []
        self._is_first_retrieval = True
        self._rag_call_count = 0

    def accumulate_audio(self, audio_chunk, force_process: bool = False, process_windows: bool = True):
        if not self.enabled: return []
        
        if isinstance(audio_chunk, torch.Tensor):
            audio_chunk = audio_chunk.detach().cpu().numpy()
        
        if audio_chunk is not None and audio_chunk.size > 0:
            audio_chunk = np.asarray(audio_chunk, dtype=np.float32).flatten()
            self._audio_buffer = np.concatenate([self._audio_buffer, audio_chunk])
        
        if process_windows:
            self._process_new_windows(force_process=force_process)
        return []

    def _process_new_windows(self, force_process=False):
        buffer_len = len(self._audio_buffer)
        
        chunk_idx = self._last_chunk_idx + 1
        chunks_to_process = []
        start_seconds = []
        
        while True:
            start = int(chunk_idx * self.hop_samples)
            end = start + self.chunk_samples
            if end <= buffer_len:
                chunks_to_process.append(self._audio_buffer[start:end])
                start_seconds.append(start / self.sample_rate)
                chunk_idx += 1
            else:
                break
        
        if chunks_to_process:
            self._retrieve_and_aggregate(chunks_to_process, tag="normal", start_seconds=start_seconds)
            self._last_chunk_idx = chunk_idx - 1
            return

        if force_process and buffer_len > 0:
            start = int((self._last_chunk_idx + 1) * self.hop_samples)
            if start < buffer_len:
                chunk = self._audio_buffer[start:]
                chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)), mode='constant')
                self._retrieve_and_aggregate([chunk], tag="force_end", start_seconds=[start / self.sample_rate])
                self._last_chunk_idx += 1

    def _retrieve_and_aggregate(self, chunks, tag="", start_seconds=None):
        t0 = perf_counter()
        target_len = 30720 
        audios = []
        for c in chunks:
            max_val = np.max(np.abs(c))
            if max_val > 0:
                c = c / max_val

            if len(c) < target_len:
                c = np.pad(c, (0, target_len - len(c)), mode='constant')
            elif len(c) > target_len:
                c = c[:target_len]
            audios.append(c)
            
        t_feat0 = perf_counter()
        inputs = self.feature_extractor(audios, sampling_rate=16000, return_tensors="pt", padding=False)
        self.last_feat_time_sec = float(perf_counter() - t_feat0)
        features = inputs.input_features 
        B, C, T_mel = features.shape
        
        input_features = features.transpose(0, 1).reshape(C, -1).to(self.device).to(torch.bfloat16)
        feature_lens = torch.full((B,), T_mel, dtype=torch.long, device=self.device)
        
        t_enc0 = perf_counter()
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                audio_embs = self.model(input_features, feature_lens)
            audio_embs = audio_embs.cpu().float().numpy()
        self.last_audio_enc_time_sec = float(perf_counter() - t_enc0)

        # Batch FAISS search for all windows at once (much faster than per-window Python loop).
        t_faiss0 = perf_counter()
        faiss.normalize_L2(audio_embs)
        D, I = self.index.search(audio_embs, self.voting_k)
        self.last_faiss_time_sec = float(perf_counter() - t_faiss0)
        
        for i in range(audio_embs.shape[0]):
            
            t_start = start_seconds[i] if (start_seconds and i < len(start_seconds)) else 0.0
            t_end = t_start + self.chunk_size
            
            if self.verbose:
                print(f"\n[RAG Window {t_start:.2f}s-{t_end:.2f}s] ({tag}) Recall Top-{self.voting_k}:")

            window_res = {}
            for rank, (dist, idx) in enumerate(zip(D[i], I[i])):
                if 0 <= idx < len(self.term_list):
                    term_info = self.term_list[idx]
                    term_lc = term_info["key"]
                    score = l2_distance_to_score(float(dist))
                    
                    window_res[term_lc] = score
                    if term_lc not in self._term_canonical:
                        self._term_canonical[term_lc] = term_info["term"]
                    
                    if self.verbose and rank < 5: 
                        print(
                            f"  - {term_info['term']} ({term_info['target_translations'].get(self.target_lang, '')}) "
                            f"[score: {score:.4f}]"
                        )

            if self.aggregation_strategy == "voting":
                self._window_results_buffer.append(window_res)
                if len(self._window_results_buffer) > self.buffer_max_size:
                    self._window_results_buffer.pop(0)
                
                term_agg_scores = {}
                term_counts = {}
                for res in self._window_results_buffer:
                    for term, score in res.items():
                        term_agg_scores[term] = term_agg_scores.get(term, 0.0) + score
                        term_counts[term] = term_counts.get(term, 0) + 1
                
                # Use parameter for minimum votes
                min_count = min(self.voting_min_votes, len(self._window_results_buffer))
                new_step_scores = {t: s for t, s in term_agg_scores.items() if term_counts[t] >= min_count}
                
            else: # "max_pool"
                new_step_scores = {t: s for i, (t, s) in enumerate(window_res.items()) if i < self.top_k}
            
            for term_lc, score in new_step_scores.items():
                if term_lc not in self._term_scores or score > self._term_scores[term_lc]:
                    self._term_scores[term_lc] = score

        dt = perf_counter() - t0
        self.last_retrieve_time_sec = float(dt)
        self.total_retrieve_time_sec += float(dt)
        self.total_retrieve_calls += 1

    def retrieve_windows(self, chunks: List[np.ndarray]) -> List[Dict[str, float]]:
        """
        Stateless / side-effect-free retrieval.

        Given a list of raw audio windows (each 1-D float32 array), returns a list of dicts:
          window_res[i] = {term_key_lower: score}

        Notes:
        - Uses the same feature extraction + audio encoder + FAISS index as streaming mode.
        - Does NOT touch internal buffers like `_audio_buffer` / `_term_scores`.
        - Uses batch FAISS search for speed.
        """
        if not self.enabled:
            return []
        if not chunks:
            return []
        if faiss is None:
            raise RuntimeError("faiss is not available but is required for retrieval.")

        target_len = 30720
        audios: List[np.ndarray] = []
        for c in chunks:
            c = np.asarray(c, dtype=np.float32).flatten()
            max_val = float(np.max(np.abs(c))) if c.size else 0.0
            if max_val > 0:
                c = c / max_val
            if c.shape[0] < target_len:
                c = np.pad(c, (0, target_len - c.shape[0]), mode="constant")
            elif c.shape[0] > target_len:
                c = c[:target_len]
            audios.append(c)

        inputs = self.feature_extractor(audios, sampling_rate=16000, return_tensors="pt", padding=False)
        features = inputs.input_features
        B, C, T_mel = features.shape

        input_features = features.transpose(0, 1).reshape(C, -1).to(self.device).to(torch.bfloat16)
        feature_lens = torch.full((B,), T_mel, dtype=torch.long, device=self.device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                audio_embs = self.model(input_features, feature_lens)
            audio_embs = audio_embs.cpu().float().numpy()

        faiss.normalize_L2(audio_embs)
        D, I = self.index.search(audio_embs, self.voting_k)

        results: List[Dict[str, float]] = []
        for i in range(audio_embs.shape[0]):
            window_res: Dict[str, float] = {}
            for dist, idx in zip(D[i], I[i]):
                if 0 <= idx < len(self.term_list):
                    term_info = self.term_list[idx]
                    term_lc = term_info["key"]
                    window_res[term_lc] = l2_distance_to_score(float(dist))
            results.append(window_res)

        return results

    def get_current_references(self, min_terms=0):
        if not self.enabled or not self._term_scores:
            return []
        
        self._is_first_retrieval = False
        
        n = max(min_terms, self.top_k) 
        sorted_terms = sorted(self._term_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for term_lc, score in sorted_terms[:n]:
            if score < self.score_threshold:
                continue

            term_info = self.term_map.get(term_lc)
            if not term_info:
                continue

            term_surface = term_info.get("term", term_lc)
            translation = term_info["target_translations"].get(self.target_lang, "")
            results.append({"key": term_lc, "term": term_surface, "translation": translation, "score": score})

        self._term_scores = {}
        self._term_canonical = {}
        
        return results

    def get_audio_duration(self):
        return len(self._audio_buffer) / self.sample_rate

