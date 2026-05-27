"""
Streaming MaxSim retriever for SimulEval inference.

Replaces the FAISS-based sliding-window retriever with direct MaxSim scoring,
matching the retrieval mechanism used during Speech LLM training data generation.

The audio chunk (whatever length) is passed directly to the MaxSim model, which
handles multi-scale windowing internally via _multiscale_pool.

Reuses model definitions from qwen3_glossary_neg_train.py:
  - Qwen3OmniRetriever (use_maxsim=True)
  - BgeM3TextEncoder
  - _maxsim_score
"""

from __future__ import annotations

# ======Configuration=====
AUDIO_MODEL_ID = "Atotti/Qwen3-Omni-AudioTransformer"
TEXT_MODEL_ID = "BAAI/bge-m3"
RAG_FEATURE_EXTRACTOR_MODEL_ID = "openai/whisper-large-v3"
EXPECTED_SAMPLE_RATE = 16000
ENCODER_FPS = 12.5
FRAME_SEC = 1.0 / ENCODER_FPS

LORA_RANK = 128
LORA_ALPHA = 256
POOLING_TYPE = "transformer"
TEMPERATURE = 0.03
USE_MAXSIM = True
# Match the actual retriever checkpoint used for TCM final v3 training.
# The training script default is older; the launched run overrode it explicitly.
MAXSIM_WINDOWS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24]
MAXSIM_STRIDE = 2
TARGET_DIM = 1024

TEXT_LORA_RANK = 128
TEXT_LORA_ALPHA = 256
TEXT_POOLING = "cls"
SPARSE_WEIGHT = 0.7

LORA_TARGET_MODULES = "q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2".split()
TEXT_LORA_TARGET_MODULES = "query key value dense".split()
# ======Configuration=====

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_model_classes():
    """Import model classes from the training code."""
    train_dir = _REPO_ROOT / "documents" / "code" / "train" / "term_train"
    if str(train_dir) not in sys.path:
        sys.path.insert(0, str(train_dir))
    from qwen3_glossary_neg_train import (
        BgeM3TextEncoder,
        Qwen3OmniRetriever,
        _maxsim_score,
    )
    return Qwen3OmniRetriever, BgeM3TextEncoder, _maxsim_score


def _build_retriever_model(
    device: torch.device,
    lora_rank: int,
    text_lora_rank: int,
    maxsim_windows: Optional[List[int]] = None,
    maxsim_stride: int = MAXSIM_STRIDE,
):
    """Build MaxSim retriever and text encoder with given LoRA ranks."""
    Qwen3OmniRetriever, BgeM3TextEncoder, _maxsim_score = _import_model_classes()
    from transformers import WhisperFeatureExtractor

    retriever = Qwen3OmniRetriever(
        model_id=AUDIO_MODEL_ID,
        target_dim=TARGET_DIM,
        use_lora=True,
        lora_rank=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_target_modules=LORA_TARGET_MODULES,
        temperature=TEMPERATURE,
        learn_temp=False,
        pooling_type=POOLING_TYPE,
        use_maxsim=USE_MAXSIM,
        maxsim_windows=maxsim_windows or MAXSIM_WINDOWS,
        maxsim_stride=maxsim_stride,
    ).to(device)

    feat_ext = WhisperFeatureExtractor.from_pretrained(RAG_FEATURE_EXTRACTOR_MODEL_ID)

    return retriever, feat_ext, _maxsim_score


def _load_checkpoint(
    retriever, model_path: str, device: torch.device
) -> None:
    """Load checkpoint weights into audio retriever (text encoder already pre-encoded in index)."""
    ckpt = torch.load(model_path, map_location=device)

    def _strip(sd):
        return {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in sd.items()
        }

    retriever.load_state_dict(
        _strip(ckpt.get("model_state_dict", {})), strict=False
    )
    retriever.eval()
    logger.info("Loaded MaxSim retriever checkpoint from %s", model_path)


@torch.no_grad()
def _encode_audio(
    audio_array: np.ndarray,
    retriever,
    feat_ext,
    device: torch.device,
) -> torch.Tensor:
    """Encode a single variable-length audio array into MaxSim multi-window embeddings.

    Returns: [1, W, D] tensor of L2-normalized window embeddings.
    """
    inp = feat_ext(
        [audio_array],
        sampling_rate=EXPECTED_SAMPLE_RATE,
        return_tensors="pt",
        padding=False,
    )
    mel = inp.input_features.squeeze(0)  # [C, T_mel]
    mel_len = mel.shape[-1]
    input_features = mel.to(device, dtype=torch.bfloat16)
    feature_lens = torch.tensor([mel_len], dtype=torch.long, device=device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        embs = retriever(input_features, feature_lens)
    return embs.float()  # [1, W, D]


def _compute_sim(
    speech_emb: torch.Tensor,
    text_embs: torch.Tensor,
    maxsim_score_fn,
) -> torch.Tensor:
    """Compute raw cos-sim / max-sim logits [1, G] for one audio chunk."""
    if speech_emb.ndim == 3:
        return maxsim_score_fn(speech_emb, text_embs)
    return speech_emb @ text_embs.T


def _build_window_time_ranges(
    maxsim_windows: List[int], maxsim_stride: int, t_frames: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build window start/end times in seconds, matching _multiscale_pool order."""
    starts = []
    ends = []
    for w in maxsim_windows:
        if w >= t_frames:
            starts.append(0.0)
            ends.append(t_frames * FRAME_SEC)
        else:
            n_out = (t_frames - w) // maxsim_stride + 1
            for p in range(n_out):
                frame_start = p * maxsim_stride
                frame_end = frame_start + w
                starts.append(frame_start * FRAME_SEC)
                ends.append(frame_end * FRAME_SEC)
    return (
        torch.tensor(starts, dtype=torch.float32),
        torch.tensor(ends, dtype=torch.float32),
    )


@torch.no_grad()
def _encode_audio_projected_seq(
    audio_array: np.ndarray,
    retriever,
    feat_ext,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode audio to projected encoder frames before MaxSim pooling.

    This mirrors Qwen3OmniRetriever.forward up to `projected_seq`, so inference
    can pool over a continuous previous+current frame sequence and attach
    per-window timeline metadata.
    """
    inp = feat_ext(
        [audio_array],
        sampling_rate=EXPECTED_SAMPLE_RATE,
        return_tensors="pt",
        padding=False,
    )
    mel = inp.input_features.squeeze(0)  # [C, T_mel]
    mel_len = mel.shape[-1]
    input_features = mel.to(device, dtype=torch.bfloat16)
    feature_lens = torch.tensor([mel_len], dtype=torch.long, device=device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = retriever.audio_encoder(input_features, feature_lens)
        hidden_states = outputs.last_hidden_state

        if hidden_states.ndim == 2:
            output_lens: List[int] = []
            for cur in feature_lens.tolist():
                reduced = cur
                for _ in range(3):
                    reduced = (reduced + 1) // 2
                output_lens.append(reduced)
            if sum(output_lens) != hidden_states.shape[0]:
                ratio = input_features.shape[1] / hidden_states.shape[0]
                output_lens = [max(1, round(x / ratio)) for x in feature_lens.tolist()]
                output_lens[-1] = hidden_states.shape[0] - sum(output_lens[:-1])

            from torch.nn.utils.rnn import pad_sequence

            parts = torch.split(hidden_states, output_lens, dim=0)
            hidden_states = pad_sequence(parts, batch_first=True)
            feature_lens = torch.tensor(output_lens, device=hidden_states.device)

        batch_size, max_len, _ = hidden_states.shape
        mask = (
            torch.arange(max_len, device=hidden_states.device).expand(batch_size, max_len)
            < feature_lens.unsqueeze(1)
        )
        projected_seq = retriever.projector(hidden_states)
        projected_seq = projected_seq * mask.unsqueeze(-1).float()

    return projected_seq.float(), mask


@torch.no_grad()
def _encode_audio_projected_seq_batch(
    audio_arrays: Sequence[np.ndarray],
    retriever,
    feat_ext,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode variable-length audio to projected encoder frames.

    The Whisper feature extraction is batched for same-length arrays, but the
    Qwen3-Omni audio encoder is intentionally kept per sample.  The Qwen audio
    encoder accepts concatenated mel streams with ``feature_lens``, but that
    path is not numerically equivalent to the serial SimulEval retriever near
    tau thresholds because convolution/position handling changes at sequence
    boundaries.  Keeping the encoder call per sample preserves exact term_map
    semantics while still removing the CPU feature-extractor loop for the
    common same-lm case.
    """
    if not audio_arrays:
        raise ValueError("_encode_audio_projected_seq_batch requires non-empty audio_arrays")

    audio_lens = [int(np.asarray(audio).shape[0]) for audio in audio_arrays]
    if any(length <= 0 for length in audio_lens):
        raise ValueError("audio_arrays contains an empty audio segment")

    if len(set(audio_lens)) == 1:
        inp = feat_ext(
            list(audio_arrays),
            sampling_rate=EXPECTED_SAMPLE_RATE,
            return_tensors="pt",
            padding=False,
        )
        mel_list = [inp.input_features[i] for i in range(len(audio_arrays))]
    else:
        mel_list: List[torch.Tensor] = []
        for audio in audio_arrays:
            inp = feat_ext(
                [audio],
                sampling_rate=EXPECTED_SAMPLE_RATE,
                return_tensors="pt",
                padding=False,
            )
            mel = inp.input_features.squeeze(0)  # [C, T_mel_i]
            mel_list.append(mel)

    projected_parts: List[torch.Tensor] = []
    mask_parts: List[torch.Tensor] = []
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for mel in mel_list:
            mel_len = int(mel.shape[-1])
            input_features = mel.to(device, dtype=torch.bfloat16)
            feature_lens = torch.tensor([mel_len], dtype=torch.long, device=device)
            outputs = retriever.audio_encoder(input_features, feature_lens)
            hidden_states = outputs.last_hidden_state

            if hidden_states.ndim == 2:
                reduced = mel_len
                for _ in range(3):
                    reduced = (reduced + 1) // 2
                output_lens = [int(reduced)]
                if output_lens[0] != hidden_states.shape[0]:
                    ratio = input_features.shape[1] / hidden_states.shape[0]
                    output_lens = [max(1, round(mel_len / ratio))]
                    output_lens[-1] = hidden_states.shape[0]

                hidden_states = hidden_states.unsqueeze(0)
                feature_lens = torch.tensor(output_lens, device=hidden_states.device)

            batch_size, max_len, _ = hidden_states.shape
            mask = (
                torch.arange(max_len, device=hidden_states.device).expand(batch_size, max_len)
                < feature_lens.unsqueeze(1)
            )
            projected_seq = retriever.projector(hidden_states)
            projected_seq = projected_seq * mask.unsqueeze(-1).float()
            projected_parts.append(projected_seq.squeeze(0).float())
            mask_parts.append(mask.squeeze(0))

    from torch.nn.utils.rnn import pad_sequence

    projected_seq = pad_sequence(projected_parts, batch_first=True)
    mask = pad_sequence(mask_parts, batch_first=True, padding_value=False)
    return projected_seq.float(), mask


def _retrieve_topk(
    speech_emb: torch.Tensor,
    text_embs: torch.Tensor,
    maxsim_score_fn,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) of top-k glossary terms for one audio chunk."""
    sim = _compute_sim(speech_emb, text_embs, maxsim_score_fn)  # [1, G]
    scores = sim.squeeze(0)
    n = min(k, scores.numel())
    top_sco, top_idx = torch.topk(scores, k=n, largest=True, sorted=True)
    return top_idx.cpu().numpy(), top_sco.cpu().numpy()


class StreamingMaxSimRetriever:
    """MaxSim-based retriever for streaming simultaneous translation.

    Replaces FAISS-based retrieval with direct MaxSim scoring against
    pre-encoded text embeddings from a .pt index file.
    """

    def __init__(
        self,
        model_path: str,
        index_path: str,
        device: str = "cuda:1",
        top_k: int = 10,
        lora_rank: int = LORA_RANK,
        text_lora_rank: int = TEXT_LORA_RANK,
        target_lang: str = "zh",
        window_sec: float = 0.0,
        score_threshold: float = 0.0,
        maxsim_windows: Optional[List[int]] = None,
        maxsim_stride: int = MAXSIM_STRIDE,
    ):
        self.device = torch.device(device)
        self.top_k = top_k
        self.target_lang = target_lang
        self.window_sec = window_sec
        self.score_threshold = float(score_threshold)
        self.enabled = False

        assert model_path and Path(model_path).is_file(), (
            f"MaxSim model checkpoint not found: {model_path}"
        )
        assert index_path and Path(index_path).is_file(), (
            f"MaxSim text index not found: {index_path}"
        )

        logger.info(
            "Loading MaxSim retriever: model=%s index=%s device=%s lora_r=%d "
            "window_sec=%.2f maxsim_windows=%s maxsim_stride=%d",
            model_path,
            index_path,
            device,
            lora_rank,
            window_sec,
            maxsim_windows or MAXSIM_WINDOWS,
            maxsim_stride,
        )

        self.retriever, self.feat_ext, self._maxsim_score = _build_retriever_model(
            self.device,
            lora_rank,
            text_lora_rank,
            maxsim_windows=maxsim_windows,
            maxsim_stride=maxsim_stride,
        )
        _load_checkpoint(self.retriever, model_path, self.device)

        index_data = torch.load(index_path, map_location=self.device)
        self.text_embs = index_data["text_embs"].to(self.device)
        self.term_list = index_data["term_list"]
        assert self.text_embs.shape[0] == len(self.term_list), (
            f"Index mismatch: text_embs has {self.text_embs.shape[0]} entries "
            f"but term_list has {len(self.term_list)}"
        )

        self._audio_buffer: Optional[np.ndarray] = None
        self._retrieve_offset: int = 0
        self._last_results: List[Dict] = []

        self.enabled = True
        logger.info(
            "MaxSim retriever ready: %d glossary terms, text_embs shape=%s",
            len(self.term_list), list(self.text_embs.shape),
        )

    def reset(self) -> None:
        """Reset state for a new utterance."""
        self._audio_buffer = None
        self._retrieve_offset = 0
        self._last_results = []

    def get_audio_duration(self) -> float:
        """Return duration of accumulated audio in seconds."""
        if self._audio_buffer is None:
            return 0.0
        return float(len(self._audio_buffer)) / EXPECTED_SAMPLE_RATE

    def accumulate_audio(self, new_audio: Optional[np.ndarray]) -> None:
        """Append new audio samples to the internal buffer."""
        if new_audio is None or len(new_audio) == 0:
            return
        audio = np.asarray(new_audio, dtype=np.float32).flatten()
        if self._audio_buffer is None:
            self._audio_buffer = audio
        else:
            self._audio_buffer = np.concatenate([self._audio_buffer, audio])

    def retrieve(self, top_k: Optional[int] = None) -> List[Dict]:
        """Run MaxSim retrieval using a sliding window over accumulated audio.

        When ``window_sec > 0``, encodes the most recent ``window_sec`` of audio
        (reaching back into previously processed regions).  This gives 50%
        overlap between consecutive calls when the caller invokes retrieve()
        every ``window_sec / 2`` seconds.

        When ``window_sec <= 0`` (legacy mode), encodes only audio since the
        last retrieve() call (no overlap).

        Returns list of dicts with keys: term, translation, key, score.
        """
        if self._audio_buffer is None or len(self._audio_buffer) == 0:
            return self._last_results

        buffer_end = len(self._audio_buffer)
        new_since_last = buffer_end - self._retrieve_offset
        min_samples = int(0.48 * EXPECTED_SAMPLE_RATE)
        if new_since_last < min_samples:
            return self._last_results

        if self.window_sec > 0:
            window_samples = int(self.window_sec * EXPECTED_SAMPLE_RATE)
            start = max(0, buffer_end - window_samples)
        else:
            start = self._retrieve_offset

        chunk = self._audio_buffer[start:buffer_end]

        k = top_k if top_k is not None else self.top_k

        embs = _encode_audio(
            chunk, self.retriever, self.feat_ext, self.device
        )

        idx_arr, sco_arr = _retrieve_topk(
            embs, self.text_embs, self._maxsim_score, k
        )

        self._retrieve_offset = buffer_end

        results = self._build_results(idx_arr, sco_arr)
        self._last_results = results
        return results

    def retrieve_timeline(
        self,
        top_k: Optional[int],
        current_start_sec: float,
        current_end_sec: float,
        lookback_sec: Optional[float] = None,
    ) -> List[Dict]:
        """Run timeline-aware MaxSim over previous+current audio frames.

        The encoded audio spans roughly ``[current_start_sec - lookback_sec,
        current_end_sec]``.  MaxSim windows whose best evidence ends before the
        current vLLM chunk starts are masked out before top-k selection.  This
        preserves boundary-crossing terms while preventing stale terms from the
        previous chunk from leaking into the current prompt.
        """
        if self._audio_buffer is None or len(self._audio_buffer) == 0:
            return []

        k = top_k if top_k is not None else self.top_k
        current_start_sec = max(0.0, float(current_start_sec))
        current_end_sec = max(current_start_sec, float(current_end_sec))
        lookback = self.window_sec if lookback_sec is None else float(lookback_sec)
        lookback = max(0.0, lookback)

        buffer_end = min(len(self._audio_buffer), int(round(current_end_sec * EXPECTED_SAMPLE_RATE)))
        if buffer_end <= 0:
            return []
        encode_start_sec = max(0.0, current_start_sec - lookback)
        buffer_start = min(buffer_end, int(round(encode_start_sec * EXPECTED_SAMPLE_RATE)))
        chunk = self._audio_buffer[buffer_start:buffer_end]
        if len(chunk) == 0:
            return []

        actual_start_sec = float(buffer_start) / EXPECTED_SAMPLE_RATE
        actual_end_sec = float(buffer_end) / EXPECTED_SAMPLE_RATE
        actual_duration = max(1e-6, actual_end_sec - actual_start_sec)

        projected_seq, mask = _encode_audio_projected_seq(
            chunk, self.retriever, self.feat_ext, self.device
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            window_embs = self.retriever._multiscale_pool(projected_seq, mask)
            window_embs = F.normalize(window_embs, p=2, dim=-1).float()

        window_embs_2d = window_embs.squeeze(0)  # [W, D]
        if window_embs_2d.numel() == 0:
            return []
        t_frames = int(projected_seq.shape[1])
        rel_starts, rel_ends = _build_window_time_ranges(
            self.retriever.maxsim_windows, self.retriever.maxsim_stride, t_frames
        )
        if rel_starts.numel() != window_embs_2d.shape[0]:
            logger.warning(
                "Timeline retrieval window count mismatch: ranges=%d embs=%d",
                rel_starts.numel(), window_embs_2d.shape[0],
            )
            return []

        # Map training-time frame seconds to the actual encoded audio span.  This
        # avoids boundary drift from feature-extractor rounding on short chunks.
        nominal_duration = max(float(rel_ends.max().item()), 1e-6)
        scale = actual_duration / nominal_duration
        abs_starts = actual_start_sec + rel_starts.to(self.device) * scale
        abs_ends = actual_start_sec + rel_ends.to(self.device) * scale

        # Keep lookback evidence only when the MaxSim evidence window overlaps
        # the current vLLM chunk timeline.  Windows fully before chunk start are
        # stale; windows crossing the boundary are valid.
        valid_windows = (abs_ends > current_start_sec) & (abs_starts < current_end_sec)
        if int(valid_windows.sum().item()) == 0:
            self._last_results = []
            return []

        sim_by_window = window_embs_2d @ self.text_embs.float().T  # [W, G]
        sim_by_window = sim_by_window.masked_fill(~valid_windows.unsqueeze(1), -float("inf"))
        scores, best_window_idx = sim_by_window.max(dim=0)  # [G], [G]
        finite = torch.isfinite(scores)
        if int(finite.sum().item()) == 0:
            self._last_results = []
            return []

        n = min(int(k), int(finite.sum().item()))
        masked_scores = scores.masked_fill(~finite, -float("inf"))
        top_sco, top_idx = torch.topk(masked_scores, k=n, largest=True, sorted=True)
        top_win = best_window_idx.gather(0, top_idx)
        top_start = abs_starts.gather(0, top_win)
        top_end = abs_ends.gather(0, top_win)

        results = self._build_results(
            top_idx.detach().cpu().numpy(),
            top_sco.detach().cpu().numpy(),
            time_starts=top_start.detach().cpu().numpy(),
            time_ends=top_end.detach().cpu().numpy(),
            retrieval_mode="timeline",
        )
        self._retrieve_offset = buffer_end
        self._last_results = results
        return results

    def retrieve_timeline_batch(
        self,
        requests: Sequence[Dict[str, Any]],
        top_k: Optional[int] = None,
        lookback_sec: Optional[float] = None,
    ) -> List[List[Dict]]:
        """Batch timeline-aware retrieval for independent streaming states.

        Each request must contain:
          - ``audio_buffer``: accumulated audio up to current_end_sec
          - ``current_start_sec``
          - ``current_end_sec``

        The method batches feature extraction/audio encoding, then applies the
        same per-sample timeline window mask and score-threshold filtering as
        ``retrieve_timeline``.  It intentionally does not mutate the streaming
        retriever's internal audio buffer.
        """
        outputs: List[List[Dict]] = [[] for _ in requests]
        if not requests:
            return outputs

        k = top_k if top_k is not None else self.top_k
        default_lookback = self.window_sec if lookback_sec is None else float(lookback_sec)
        default_lookback = max(0.0, default_lookback)

        chunks: List[np.ndarray] = []
        metas: List[Dict[str, Any]] = []
        for req_idx, req in enumerate(requests):
            audio_buffer = np.asarray(req.get("audio_buffer"), dtype=np.float32).flatten()
            if audio_buffer.size == 0:
                continue
            current_start_sec = max(0.0, float(req["current_start_sec"]))
            current_end_sec = max(current_start_sec, float(req["current_end_sec"]))
            cur_lookback = max(0.0, float(req.get("lookback_sec", default_lookback)))

            buffer_end = min(
                len(audio_buffer),
                int(round(current_end_sec * EXPECTED_SAMPLE_RATE)),
            )
            if buffer_end <= 0:
                continue
            encode_start_sec = max(0.0, current_start_sec - cur_lookback)
            buffer_start = min(
                buffer_end,
                int(round(encode_start_sec * EXPECTED_SAMPLE_RATE)),
            )
            chunk = audio_buffer[buffer_start:buffer_end]
            if len(chunk) == 0:
                continue

            actual_start_sec = float(buffer_start) / EXPECTED_SAMPLE_RATE
            actual_end_sec = float(buffer_end) / EXPECTED_SAMPLE_RATE
            chunks.append(chunk)
            metas.append(
                {
                    "request_idx": req_idx,
                    "current_start_sec": current_start_sec,
                    "current_end_sec": current_end_sec,
                    "actual_start_sec": actual_start_sec,
                    "actual_end_sec": actual_end_sec,
                    "actual_duration": max(1e-6, actual_end_sec - actual_start_sec),
                }
            )

        if not chunks:
            return outputs

        projected_seq, mask = _encode_audio_projected_seq_batch(
            chunks, self.retriever, self.feat_ext, self.device
        )

        for batch_idx, meta in enumerate(metas):
            valid_frames = int(mask[batch_idx].sum().item())
            if valid_frames <= 0:
                continue
            seq_i = projected_seq[batch_idx : batch_idx + 1, :valid_frames, :]
            mask_i = mask[batch_idx : batch_idx + 1, :valid_frames]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                window_embs = self.retriever._multiscale_pool(seq_i, mask_i)
                window_embs = F.normalize(window_embs, p=2, dim=-1).float()

            window_embs_2d = window_embs.squeeze(0)
            if window_embs_2d.numel() == 0:
                continue
            rel_starts, rel_ends = _build_window_time_ranges(
                self.retriever.maxsim_windows,
                self.retriever.maxsim_stride,
                valid_frames,
            )
            if rel_starts.numel() != window_embs_2d.shape[0]:
                logger.warning(
                    "Batch timeline retrieval window count mismatch: ranges=%d embs=%d",
                    rel_starts.numel(),
                    window_embs_2d.shape[0],
                )
                continue

            nominal_duration = max(float(rel_ends.max().item()), 1e-6)
            scale = float(meta["actual_duration"]) / nominal_duration
            abs_starts = float(meta["actual_start_sec"]) + rel_starts.to(self.device) * scale
            abs_ends = float(meta["actual_start_sec"]) + rel_ends.to(self.device) * scale
            valid_windows = (
                (abs_ends > float(meta["current_start_sec"]))
                & (abs_starts < float(meta["current_end_sec"]))
            )
            if int(valid_windows.sum().item()) == 0:
                continue

            sim_by_window = window_embs_2d @ self.text_embs.float().T
            sim_by_window = sim_by_window.masked_fill(
                ~valid_windows.unsqueeze(1),
                -float("inf"),
            )
            scores, best_window_idx = sim_by_window.max(dim=0)
            finite = torch.isfinite(scores)
            if int(finite.sum().item()) == 0:
                continue

            n = min(int(k), int(finite.sum().item()))
            masked_scores = scores.masked_fill(~finite, -float("inf"))
            top_sco, top_idx = torch.topk(masked_scores, k=n, largest=True, sorted=True)
            top_win = best_window_idx.gather(0, top_idx)
            top_start = abs_starts.gather(0, top_win)
            top_end = abs_ends.gather(0, top_win)

            outputs[int(meta["request_idx"])] = self._build_results(
                top_idx.detach().cpu().numpy(),
                top_sco.detach().cpu().numpy(),
                time_starts=top_start.detach().cpu().numpy(),
                time_ends=top_end.detach().cpu().numpy(),
                retrieval_mode="timeline_batch",
            )

        return outputs

    def retrieve_window_with_times(self, top_k: Optional[int] = None) -> List[Dict]:
        """Run the legacy sliding-window retrieve, but attach best-window times.

        This is used by `stride_merge`: intermediate stride windows can still be
        computed ahead of the vLLM call, while final merging can drop candidates
        whose evidence window ended before the current vLLM chunk starts.
        """
        if self._audio_buffer is None or len(self._audio_buffer) == 0:
            return self._last_results

        buffer_end = len(self._audio_buffer)
        new_since_last = buffer_end - self._retrieve_offset
        min_samples = int(0.48 * EXPECTED_SAMPLE_RATE)
        if new_since_last < min_samples:
            return self._last_results

        if self.window_sec > 0:
            window_samples = int(self.window_sec * EXPECTED_SAMPLE_RATE)
            start = max(0, buffer_end - window_samples)
        else:
            start = self._retrieve_offset

        results = self.retrieve_timeline(
            top_k=top_k if top_k is not None else self.top_k,
            current_start_sec=float(start) / EXPECTED_SAMPLE_RATE,
            current_end_sec=float(buffer_end) / EXPECTED_SAMPLE_RATE,
            lookback_sec=0.0,
        )
        for item in results:
            item["retrieval_mode"] = "stride_window"
        self._last_results = results
        return results

    def _build_results(
        self,
        idx_arr: np.ndarray,
        sco_arr: np.ndarray,
        time_starts: Optional[np.ndarray] = None,
        time_ends: Optional[np.ndarray] = None,
        retrieval_mode: str = "window",
    ) -> List[Dict]:
        """Convert top-k index/score arrays into result dicts.

        Applies score_threshold filter: candidates with score < threshold are
        dropped.  Threshold is absolute cosine similarity (MaxSim output).
        """
        results = []
        for pos, (ti, sc) in enumerate(zip(idx_arr, sco_arr)):
            score_f = float(sc)
            if score_f < self.score_threshold:
                continue
            entry = self.term_list[int(ti)]
            translations = entry.get("target_translations", {})
            translation = translations.get(self.target_lang, "")
            if not translation:
                continue
            item = {
                "key": entry.get("key", entry.get("term", "")),
                "term": entry.get("term", ""),
                "translation": translation,
                "score": score_f,
                "retrieval_mode": retrieval_mode,
            }
            if time_starts is not None and time_ends is not None:
                item["time_start"] = float(time_starts[pos])
                item["time_end"] = float(time_ends[pos])
            results.append(item)
        return results

    @staticmethod
    def merge_results(
        result_groups: List[List[Dict]],
        top_k: int,
        min_time_end: Optional[float] = None,
    ) -> List[Dict]:
        """Merge multiple retrieve() results, keeping the highest score per term."""
        best: Dict[str, Dict] = {}
        for results in result_groups:
            for r in results:
                if min_time_end is not None:
                    time_end = r.get("time_end")
                    if time_end is None or float(time_end) <= float(min_time_end):
                        continue
                term = r["term"]
                if term not in best or r["score"] > best[term]["score"]:
                    best[term] = r
        merged = sorted(best.values(), key=lambda x: -x["score"])
        return merged[:top_k]

    def get_current_references(self, min_terms: int = 0) -> List[Dict]:
        """Return the latest retrieval results (for the SimulEval agent)."""
        return self._last_results
