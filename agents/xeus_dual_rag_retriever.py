# ======Configuration=====
AUDIO_TARGET_LEN = 30720
DEFAULT_FALLBACK_LANG_CODE = "zh"
DEFAULT_TTS_EMBEDDING_BATCH_SIZE = 32
DEFAULT_TTS_MAX_PROTOTYPES_PER_TERM = 8
DEFAULT_TTS_SIMILARITY_TOP_K = 20
XEUS_DEFAULT_HIDDEN_DIM = 1024
XEUS_DEFAULT_TARGET_DIM = 1024
XEUS_DEFAULT_LORA_RANK = 32
XEUS_DEFAULT_LORA_ALPHA = 64
XEUS_DEFAULT_LORA_TARGET_MODULES = [
    "linear_q", "linear_k", "linear_v", "linear_out",
    "w_1", "w_2", "merge_proj",
]
XEUS_DEFAULT_LORA_DROPOUT = 0.05
RETRIEVE_STATS_VERSION = "xeus_dual_v1"
# ======Configuration=====

import os
import json
import math
import pickle
import logging
from time import perf_counter
from typing import Any, Optional, List, Dict, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperFeatureExtractor
from peft import LoraConfig, get_peft_model
from math import gcd
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    faiss = None

from agents.streaming_qwen3_rag_retriever_v4 import (
    Qwen3OmniRetriever,
    BgeM3TextEncoder,
    AttentivePooling as Qwen3AttentivePooling,
    l2_distance_to_score,
)


# ---------------------------------------------------------------------------
# XEUS model components (must match training architecture)
# ---------------------------------------------------------------------------

class XeusAttentivePooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = self.attention(x)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        weights = F.softmax(scores, dim=1)
        return torch.sum(x * weights, dim=1)


class XeusRetrieverInference(nn.Module):
    def __init__(self, xeus, hidden_dim: int, target_dim: int):
        super().__init__()
        self.xeus = xeus
        self.pooler = XeusAttentivePooling(hidden_dim)
        self.projector = nn.Linear(hidden_dim, target_dim)
        self.register_buffer("logit_scale", torch.tensor(0.0))

    def forward(self, wavs: torch.Tensor, wav_lengths: torch.Tensor) -> torch.Tensor:
        feats = self.xeus.encode(wavs, wav_lengths, use_mask=False, use_final_output=False)[0][-1]
        B, T_out, _ = feats.shape
        T_in = wavs.shape[1]
        assert T_in > 0
        ratio = T_out / T_in
        output_lens = (wav_lengths.float() * ratio).long().clamp(min=1, max=T_out)
        mask = torch.arange(T_out, device=feats.device).unsqueeze(0).expand(B, -1) < output_lens.unsqueeze(1)
        pooled = self.pooler(feats, mask)
        projected = self.projector(pooled)
        return F.normalize(projected, p=2, dim=-1)


def _build_xeus_model(
    xeus_checkpoint_path: str,
    trained_model_path: str,
    device: torch.device,
    hidden_dim: int = XEUS_DEFAULT_HIDDEN_DIM,
    target_dim: int = XEUS_DEFAULT_TARGET_DIM,
    lora_rank: int = XEUS_DEFAULT_LORA_RANK,
    lora_alpha: int = XEUS_DEFAULT_LORA_ALPHA,
    lora_target_modules: Optional[List[str]] = None,
) -> XeusRetrieverInference:
    """Build and load XeusRetrieverInference from base + trained checkpoint."""
    from espnet2.tasks.ssl import SSLTask

    logger.info("Loading XEUS base model from: %s", xeus_checkpoint_path)
    xeus_model, _ = SSLTask.build_model_from_file(None, xeus_checkpoint_path, "cpu")
    xeus_model = xeus_model.to(dtype=torch.bfloat16)

    if lora_target_modules is None:
        lora_target_modules = list(XEUS_DEFAULT_LORA_TARGET_MODULES)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=XEUS_DEFAULT_LORA_DROPOUT,
        bias="none",
        task_type=None,
    )
    xeus_model = get_peft_model(xeus_model, lora_config)

    model = XeusRetrieverInference(xeus_model, hidden_dim, target_dim)

    logger.info("Loading trained XEUS weights from: %s", trained_model_path)
    ckpt = torch.load(trained_model_path, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(sd, strict=True)
    logger.info("XeusRetriever loaded (strict=True).")

    model = model.to(device).to(torch.bfloat16)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# XeusDualModelRAGRetriever
# ---------------------------------------------------------------------------

class XeusDualModelRAGRetriever:
    """
    Dual-encoder RAG retriever:
        Text pathway:  Qwen3-Omni AuT (mel spectrogram) on text_device
        TTS pathway:   XEUS E-Branchformer (raw waveform) on tts_device

    GPU layout (4-GPU example):
        GPU 0-1 : Omni speech LLM (vLLM TP=2)
        GPU 2   : Qwen3-Omni text model + FAISS
        GPU 3   : XEUS TTS model + TTS prototype bank
    """

    def __init__(
        self,
        index_path: str,
        text_model_path: str,
        tts_model_path: str,
        xeus_checkpoint_path: str,
        base_model_name: str = "Atotti/Qwen3-Omni-AudioTransformer",
        text_device: str = "cuda:2",
        tts_device: str = "cuda:3",
        lora_r: int = 32,
        lora_alpha: int = 64,
        text_lora_r: int = 16,
        xeus_hidden_dim: int = XEUS_DEFAULT_HIDDEN_DIM,
        xeus_lora_rank: int = XEUS_DEFAULT_LORA_RANK,
        xeus_lora_alpha: int = XEUS_DEFAULT_LORA_ALPHA,
        top_k: int = 5,
        voting_k: int = 20,
        voting_min_votes: int = 2,
        target_lang: str = "zh",
        score_threshold: float = 0.5,
        score_threshold_mode: str = "absolute",
        chunk_size: float = 1.92,
        hop_size: float = 0.96,
        aggregation_strategy: str = "max_pool",
        sample_rate: int = 16000,
        tts_terms_npy_path: Optional[str] = None,
        tts_wav_dir: Optional[str] = None,
        tts_embedding_batch_size: int = DEFAULT_TTS_EMBEDDING_BATCH_SIZE,
        tts_max_prototypes_per_term: int = DEFAULT_TTS_MAX_PROTOTYPES_PER_TERM,
        tts_similarity_top_k: int = DEFAULT_TTS_SIMILARITY_TOP_K,
        additional_tts_mapping_path: Optional[str] = None,
        debug_audio_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        self.text_device = torch.device(text_device)
        self.tts_device = torch.device(tts_device)
        self.top_k = top_k
        self.voting_k = voting_k
        self.voting_min_votes = voting_min_votes
        self.target_lang = (target_lang or DEFAULT_FALLBACK_LANG_CODE).strip().lower()
        self.score_threshold = score_threshold
        self.score_threshold_mode = (score_threshold_mode or "absolute").strip().lower()
        assert self.score_threshold_mode in ("absolute", "relative"), (
            f"Unknown score_threshold_mode={self.score_threshold_mode}"
        )
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.aggregation_strategy = aggregation_strategy.lower()
        self.sample_rate = sample_rate
        self.verbose = verbose

        self.tts_terms_npy_path = (tts_terms_npy_path or "").strip()
        self.tts_wav_dir = (tts_wav_dir or "").strip()
        self.additional_tts_mapping_path = (additional_tts_mapping_path or "").strip()
        self.tts_embedding_batch_size = max(1, int(tts_embedding_batch_size))
        self.tts_max_prototypes_per_term = max(1, int(tts_max_prototypes_per_term))
        self.tts_similarity_top_k = max(1, int(tts_similarity_top_k))
        self.chunk_samples = int(chunk_size * sample_rate)
        self.hop_samples = int(hop_size * sample_rate)
        self.buffer_max_size = max(1, round(chunk_size / hop_size))

        self.debug_audio_dir = debug_audio_dir
        if self.debug_audio_dir:
            os.makedirs(self.debug_audio_dir, exist_ok=True)
        self._rag_call_count = 0

        # Streaming state
        self._audio_buffer = np.array([], dtype=np.float32)
        self._last_chunk_idx = -1
        self._text_term_scores: Dict[str, float] = {}
        self._tts_term_scores: Dict[str, float] = {}
        self._term_scores: Dict[str, float] = {}
        self._term_canonical: Dict[str, str] = {}
        self._is_first_retrieval = True

        self.last_processed_windows = 0
        self.last_windows_total_sec = 0.0
        self.last_avg_sec_per_window = 0.0
        self.last_retrieve_stats: Dict[str, Union[str, int, float, bool]] = {}

        # TTS bank state
        self._tts_enabled = False
        self._tts_proto_embs: Optional[np.ndarray] = None
        self._tts_term_keys: List[str] = []
        self._tts_term_proto_ranges: List[Tuple[int, int]] = []

        # ── Load FAISS index + term list ──
        assert faiss is not None, "faiss-cpu or faiss-gpu must be installed"
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        self.index = faiss.deserialize_index(data["faiss_index"])
        self.term_list = data["term_list"]
        self.term_map = {item["key"]: item for item in self.term_list}

        # ── Text pathway: Qwen3-Omni on text_device (unchanged) ──
        logger.info("Loading TEXT audio encoder (Qwen3-Omni) on %s ...", text_device)
        self.text_model = Qwen3OmniRetriever(
            model_id=base_model_name,
            target_dim=1024,
            use_lora=True,
            lora_rank=lora_r,
            lora_alpha=lora_alpha,
        ).to(self.text_device).to(torch.bfloat16)

        self.text_encoder = BgeM3TextEncoder(
            model_id="BAAI/bge-m3",
            lora_rank=text_lora_r,
        ).to(self.text_device).to(torch.bfloat16)

        text_ckpt = torch.load(text_model_path, map_location=self.text_device)
        text_sd = {k.replace("module.", ""): v for k, v in text_ckpt["model_state_dict"].items()}
        self.text_model.load_state_dict(text_sd, strict=False)
        if "text_model_state_dict" in text_ckpt:
            text_enc_sd = {k.replace("module.", ""): v for k, v in text_ckpt["text_model_state_dict"].items()}
            self.text_encoder.load_state_dict(text_enc_sd, strict=False)
            logger.info("Text encoder loaded from checkpoint.")
        else:
            logger.warning("No text_model_state_dict in text checkpoint, using base BGE-M3.")
        self.text_model.eval()
        self.text_encoder.eval()

        # Whisper feature extractor for TEXT pathway only
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

        # ── TTS pathway: XEUS on tts_device ──
        logger.info("Loading TTS audio encoder (XEUS) on %s ...", tts_device)
        self.tts_model = _build_xeus_model(
            xeus_checkpoint_path=xeus_checkpoint_path,
            trained_model_path=tts_model_path,
            device=self.tts_device,
            hidden_dim=xeus_hidden_dim,
            target_dim=XEUS_DEFAULT_TARGET_DIM,
            lora_rank=xeus_lora_rank,
            lora_alpha=xeus_lora_alpha,
        )

        self._initialize_tts_term_bank()
        self.enabled = True
        logger.info(
            "XeusDualModelRAGRetriever ready: text=%s(Qwen3) tts=%s(XEUS) tts_bank=%s",
            text_device, tts_device, self._tts_enabled,
        )

    # ────────────────── TTS bank construction (XEUS raw waveform) ──────────────────

    @staticmethod
    def _normalize_term_key(text: str) -> str:
        return str(text or "").strip().lower()

    def _load_additional_tts_mapping(self, term_to_paths: Dict[str, List[str]]) -> None:
        """Merge additional TTS mapping JSONL (term -> wav path) into term_to_paths."""
        if not self.additional_tts_mapping_path or not os.path.isfile(self.additional_tts_mapping_path):
            return
        import json as _json
        added = 0
        with open(self.additional_tts_mapping_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = _json.loads(line)
                term_key = self._normalize_term_key(entry.get("term", ""))
                wav_path = entry.get("tts_audio_path", "")
                if not term_key or term_key not in self.term_map:
                    continue
                if not wav_path or not os.path.isfile(wav_path):
                    continue
                paths = term_to_paths.setdefault(term_key, [])
                if len(paths) < self.tts_max_prototypes_per_term and wav_path not in paths:
                    paths.append(wav_path)
                    added += 1
        logger.info("Additional TTS mapping: loaded %d extra prototypes from %s",
                     added, self.additional_tts_mapping_path)

    def _initialize_tts_term_bank(self) -> None:
        has_npy = self.tts_terms_npy_path and os.path.isfile(self.tts_terms_npy_path)
        has_wav_dir = self.tts_wav_dir and os.path.isdir(self.tts_wav_dir)
        has_additional = self.additional_tts_mapping_path and os.path.isfile(self.additional_tts_mapping_path)

        if not has_npy and not has_additional:
            logger.warning("No TTS source (npy or additional mapping). TTS bank disabled.")
            return

        term_to_paths: Dict[str, List[str]] = {}

        if has_npy and has_wav_dir:
            try:
                terms_array = np.load(self.tts_terms_npy_path, allow_pickle=True)
            except Exception as exc:
                logger.warning("Failed to load TTS terms npy: %s", exc)
                terms_array = None

            if terms_array is not None:
                max_index = int(terms_array.shape[0]) if hasattr(terms_array, "shape") else len(terms_array)
                for idx in range(max_index):
                    term_key = self._normalize_term_key(str(terms_array[idx]))
                    if not term_key or term_key not in self.term_map:
                        continue
                    wav_path = os.path.join(self.tts_wav_dir, f"{idx + 1}.wav")
                    if not os.path.isfile(wav_path):
                        continue
                    paths = term_to_paths.setdefault(term_key, [])
                    if len(paths) < self.tts_max_prototypes_per_term:
                        paths.append(wav_path)

        self._load_additional_tts_mapping(term_to_paths)

        if not term_to_paths:
            logger.warning("No valid TTS term audio mapped to glossary. TTS bank disabled.")
            return

        prototype_paths: List[str] = []
        prototype_owner_term: List[str] = []
        for term_key, path_list in term_to_paths.items():
            for p in path_list:
                prototype_paths.append(p)
                prototype_owner_term.append(term_key)

        logger.info("Encoding %d TTS prototypes with XEUS model ...", len(prototype_paths))
        prototype_emb_batches: List[np.ndarray] = []
        for start in range(0, len(prototype_paths), self.tts_embedding_batch_size):
            path_batch = prototype_paths[start: start + self.tts_embedding_batch_size]
            audios = []
            for p in path_batch:
                wav, file_sr = sf.read(p, dtype='float32')
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                wav = np.asarray(wav, dtype=np.float32).flatten()
                if file_sr != self.sample_rate:
                    g = gcd(self.sample_rate, file_sr)
                    wav = resample_poly(wav, self.sample_rate // g, file_sr // g).astype(np.float32)
                max_val = float(np.max(np.abs(wav))) if wav.size > 0 else 0.0
                if max_val > 0:
                    wav = wav / max_val
                if len(wav) < AUDIO_TARGET_LEN:
                    wav = np.pad(wav, (0, AUDIO_TARGET_LEN - len(wav)), mode="constant")
                elif len(wav) > AUDIO_TARGET_LEN:
                    wav = wav[:AUDIO_TARGET_LEN]
                audios.append(wav)

            emb = self._encode_raw_waveforms(audios, self.tts_model, self.tts_device)
            prototype_emb_batches.append(emb)

        if not prototype_emb_batches:
            logger.warning("Failed to build TTS prototype embeddings.")
            return

        all_proto_embs = np.concatenate(prototype_emb_batches, axis=0)
        unique_keys = sorted(set(prototype_owner_term))
        self._tts_term_keys = unique_keys
        term_to_pos: Dict[str, int] = {k: i for i, k in enumerate(unique_keys)}
        owner_indices = np.array([term_to_pos[k] for k in prototype_owner_term], dtype=np.int32)

        sorted_order = np.argsort(owner_indices, kind="stable")
        self._tts_proto_embs = all_proto_embs[sorted_order].astype(np.float32, copy=False)
        faiss.normalize_L2(self._tts_proto_embs)
        sorted_owner = owner_indices[sorted_order]

        self._tts_term_proto_ranges = []
        cursor = 0
        for term_pos in range(len(unique_keys)):
            start_pos = cursor
            while cursor < len(sorted_owner) and sorted_owner[cursor] == term_pos:
                cursor += 1
            self._tts_term_proto_ranges.append((start_pos, cursor))

        self._tts_enabled = True
        logger.info(
            "XEUS TTS bank ready: terms=%d prototypes=%d",
            len(self._tts_term_keys), self._tts_proto_embs.shape[0],
        )

    # ────────────────── Encoding helpers ──────────────────

    @staticmethod
    def _encode_raw_waveforms(
        audios: List[np.ndarray], model: nn.Module, device: torch.device,
    ) -> np.ndarray:
        """Encode raw waveforms via XEUS → L2-normalized float32 (B, D)."""
        wavs = torch.from_numpy(np.stack(audios, axis=0)).float().to(device)
        wav_lens = torch.tensor([a.shape[0] for a in audios], dtype=torch.long, device=device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                embs = model(wavs, wav_lens)
            embs = embs.detach().cpu().float().numpy()
        faiss.normalize_L2(embs)
        return embs.astype(np.float32, copy=False)

    def _prepare_mel_features(self, audios: List[np.ndarray]):
        """Extract mel features (CPU) for Qwen3-Omni TEXT pathway."""
        inputs = self.feature_extractor(audios, sampling_rate=self.sample_rate, return_tensors="pt", padding=False)
        features = inputs.input_features
        batch_size, channels, mel_len = features.shape
        packed = features.transpose(0, 1).reshape(channels, -1)
        return packed, batch_size, mel_len

    def _encode_text_pathway(self, packed_features: torch.Tensor, batch_size: int, mel_len: int) -> np.ndarray:
        """Qwen3-Omni text audio encoder → L2-normalized embeddings."""
        input_features = packed_features.to(self.text_device).to(torch.bfloat16)
        feature_lens = torch.full((batch_size,), mel_len, dtype=torch.long, device=self.text_device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                embs = self.text_model(input_features, feature_lens)
            embs = embs.detach().cpu().float().numpy()
        faiss.normalize_L2(embs)
        return embs

    def _encode_tts_pathway(self, audios: List[np.ndarray]) -> np.ndarray:
        """XEUS raw waveform encoder → L2-normalized embeddings."""
        return self._encode_raw_waveforms(audios, self.tts_model, self.tts_device)

    # ────────────────── TTS scoring ──────────────────

    def _compute_tts_window_scores(self, tts_emb_2d: np.ndarray, top_k_override: int = 0) -> Dict[str, float]:
        if not self._tts_enabled or self._tts_proto_embs is None:
            return {}
        speech_vec = tts_emb_2d.reshape(-1).astype(np.float32, copy=False)
        proto_scores = self._tts_proto_embs @ speech_vec
        if proto_scores.size == 0:
            return {}

        term_scores = np.full(len(self._tts_term_keys), -np.inf, dtype=np.float32)
        for term_pos, (start, end) in enumerate(self._tts_term_proto_ranges):
            if start < end:
                term_scores[term_pos] = float(proto_scores[start:end].max())

        valid_mask = np.isfinite(term_scores)
        if not np.any(valid_mask):
            return {}

        valid_indices = np.where(valid_mask)[0]
        valid_scores = term_scores[valid_indices]
        effective_k = top_k_override if top_k_override > 0 else self.tts_similarity_top_k
        local_top_k = min(effective_k, len(valid_scores))
        if local_top_k <= 0:
            return {}
        top_idx = np.argpartition(valid_scores, -local_top_k)[-local_top_k:]
        top_idx = top_idx[np.argsort(valid_scores[top_idx])[::-1]]

        out: Dict[str, float] = {}
        for idx in top_idx.tolist():
            term_key = self._tts_term_keys[int(valid_indices[idx])]
            out[term_key] = float(valid_scores[idx])
        return out

    # ────────────────── Translation helper ──────────────────

    def _select_translation(self, term_info: Dict[str, object]) -> str:
        try:
            trans = term_info.get("target_translations", {})
            if not isinstance(trans, dict):
                return ""
            v = trans.get(self.target_lang, "")
            if v:
                return str(v)
            v = trans.get(DEFAULT_FALLBACK_LANG_CODE, "")
            if v:
                return str(v)
            for _, vv in trans.items():
                if vv:
                    return str(vv)
            return ""
        except Exception:
            return ""

    # ────────────────── Streaming API ──────────────────

    def reset(self):
        if self.verbose:
            print("\n[XeusDualRAG] Resetting buffer and scores...")
        self._audio_buffer = np.array([], dtype=np.float32)
        self._last_chunk_idx = -1
        self._text_term_scores = {}
        self._tts_term_scores = {}
        self._term_scores = {}
        self._term_canonical = {}
        self._is_first_retrieval = True
        self._rag_call_count = 0
        self.last_processed_windows = 0
        self.last_windows_total_sec = 0.0
        self.last_avg_sec_per_window = 0.0

    def get_audio_duration(self) -> float:
        return len(self._audio_buffer) / self.sample_rate

    def accumulate_audio(self, audio_chunk, force_process=False):
        if not self.enabled:
            return []

        if isinstance(audio_chunk, torch.Tensor):
            audio_chunk = audio_chunk.detach().cpu().numpy()

        if audio_chunk is not None and audio_chunk.size > 0:
            audio_chunk = np.asarray(audio_chunk, dtype=np.float32).flatten()
            self._audio_buffer = np.concatenate([self._audio_buffer, audio_chunk])

        self.last_processed_windows = 0
        self.last_windows_total_sec = 0.0
        self.last_avg_sec_per_window = 0.0

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
            self.last_processed_windows = len(chunks_to_process)
            t0 = perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._retrieve_and_aggregate(chunks_to_process, tag="normal", start_seconds=start_seconds)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = float(perf_counter() - t0)
            self.last_windows_total_sec = dt
            self.last_avg_sec_per_window = dt / float(self.last_processed_windows)
            self._last_chunk_idx = chunk_idx - 1
            return

        if force_process and buffer_len > 0:
            start = int((self._last_chunk_idx + 1) * self.hop_samples)
            if start < buffer_len:
                chunk = self._audio_buffer[start:]
                chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)), mode="constant")
                self.last_processed_windows = 1
                t0 = perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self._retrieve_and_aggregate([chunk], tag="force_end", start_seconds=[start / self.sample_rate])
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt = float(perf_counter() - t0)
                self.last_windows_total_sec = dt
                self.last_avg_sec_per_window = dt
                self._last_chunk_idx += 1

    def _retrieve_and_aggregate(self, chunks, tag="", start_seconds=None):
        import soundfile as sf

        target_len = AUDIO_TARGET_LEN
        audios = []
        for c in chunks:
            max_val = np.max(np.abs(c))
            if max_val > 0:
                c = c / max_val
            if self.debug_audio_dir:
                wav_path = os.path.join(self.debug_audio_dir, f"rag_{tag}_call{self._rag_call_count:03d}.wav")
                sf.write(wav_path, c, self.sample_rate)
                self._rag_call_count += 1
            if len(c) < target_len:
                c = np.pad(c, (0, target_len - len(c)), mode="constant")
            elif len(c) > target_len:
                c = c[:target_len]
            audios.append(c)

        # ── Text pathway: mel features → Qwen3-Omni ──
        packed_features, batch_size, mel_len = self._prepare_mel_features(audios)
        text_embs = self._encode_text_pathway(packed_features, batch_size, mel_len)

        # ── TTS pathway: raw waveform → XEUS (parallel on different GPU) ──
        tts_embs = self._encode_tts_pathway(audios) if self._tts_enabled else None

        # ── Per-window scoring + pool_then_intersect ──
        for i in range(len(chunks)):
            t_start = start_seconds[i] if (start_seconds and i < len(start_seconds)) else 0.0
            t_end = t_start + self.chunk_size

            # Text: FAISS search
            text_emb = text_embs[i].reshape(1, -1).copy()
            faiss.normalize_L2(text_emb)
            D, I = self.index.search(text_emb, self.voting_k)

            if self.verbose:
                print(f"\n[XeusDualRAG Window {t_start:.2f}s-{t_end:.2f}s] ({tag}) Text Top-{self.voting_k}:")

            for rank, (dist, idx) in enumerate(zip(D[0], I[0])):
                if 0 <= idx < len(self.term_list):
                    term_info = self.term_list[idx]
                    term_lc = term_info["key"]
                    score = l2_distance_to_score(float(dist))
                    if term_lc not in self._text_term_scores or score > self._text_term_scores[term_lc]:
                        self._text_term_scores[term_lc] = score
                    if term_lc not in self._term_canonical:
                        self._term_canonical[term_lc] = term_info["term"]
                    if self.verbose and rank < 5:
                        tr = self._select_translation(term_info)
                        print(f"  - {term_info['term']} ({tr}) [score: {score:.4f}]")

            # TTS: prototype bank search
            if tts_embs is not None:
                tts_emb = tts_embs[i].reshape(1, -1)
                tts_window_res = self._compute_tts_window_scores(tts_emb, top_k_override=self.voting_k)
                for term_lc, score in tts_window_res.items():
                    if term_lc not in self._tts_term_scores or score > self._tts_term_scores[term_lc]:
                        self._tts_term_scores[term_lc] = score

    def get_current_references(self, min_terms=0):
        if self._text_term_scores and self._tts_term_scores:
            overlap_keys = set(self._text_term_scores.keys()) & set(self._tts_term_scores.keys())
            self._term_scores = {
                k: min(self._text_term_scores[k], self._tts_term_scores[k])
                for k in overlap_keys
            }
        elif self._text_term_scores:
            self._term_scores = dict(self._text_term_scores)
        else:
            self._term_scores = {}

        self._text_term_scores = {}
        self._tts_term_scores = {}

        if not self.enabled or not self._term_scores:
            return []

        self._is_first_retrieval = False
        n = max(min_terms, self.top_k)
        sorted_terms = sorted(self._term_scores.items(), key=lambda x: x[1], reverse=True)

        filtered_terms = sorted_terms
        if self.score_threshold_mode == "relative":
            reference_k = self.voting_k if self.voting_k > 0 else n
            reference_index = min(reference_k, len(sorted_terms)) - 1
            if reference_index < 0:
                reference_index = 0
            threshold_base_score = sorted_terms[reference_index][1] if sorted_terms else None
            if threshold_base_score is not None:
                filtered_terms = [
                    (term_lc, score) for term_lc, score in sorted_terms
                    if (score - threshold_base_score) >= self.score_threshold
                ]
            else:
                filtered_terms = []

        results = []
        for term_lc, score in filtered_terms[:n]:
            if self.score_threshold_mode != "relative" and score < self.score_threshold:
                continue
            term_info = self.term_map.get(term_lc)
            if not term_info:
                continue
            term_surface = term_info.get("term", term_lc)
            translation = self._select_translation(term_info)
            results.append({"key": term_lc, "term": term_surface, "translation": translation, "score": score})

        self._term_scores = {}
        self._term_canonical = {}
        return results
