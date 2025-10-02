import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Union, Optional

try:
    from transformers import AutoProcessor, AutoModel
except Exception as _e:
    AutoProcessor = None
    AutoModel = None


class Qwen3AuTSpeechEncoder:
    """
    AuT (Audio-understanding Transformer) speech encoder wrapper.

    Goals:
    - Drop-in replace Qwen2 audio encoder in the existing pipeline
    - Follow Qwen3 preprocessing: 16kHz, 128-mel, 25ms window, 10ms hop
    - AuT does 8x conv downsample before attention → token rate ~12.5 Hz
    - Provide masked mean pooling with feature_attention_mask downsampled to AuT length

    This wrapper exposes:
    - get_hidden_size(): int
    - predict(audio_inputs: List[Union[np.ndarray, torch.Tensor]], dynamic_padding=True) -> torch.Tensor [B, H]
    - attributes: model (nn.Module) so that optional LoRA can be attached by the training wrapper if desired
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Omni-Audio", device: str = "cuda"):
        self.device = device
        self.model_name = model_name

        if AutoProcessor is None or AutoModel is None:
            raise RuntimeError("transformers is required to load AuT model")

        print(f"[INFO] Loading AuT model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        # Use AutoModel to keep compatibility with Qwen3 Omni checkpoints
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=None,
        ).to(device)

        try:
            if hasattr(self.model, "config"):
                setattr(self.model.config, "use_cache", False)
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            print("[INFO] Enabled gradient checkpointing and disabled use_cache")
        except Exception as e:
            print(f"[WARN] Failed to enable gradient checkpointing: {e}")

        # Find internal AuT module for direct feature extraction
        self.aut_module = self._locate_aut_module()
        if self.aut_module is None:
            # Fall back to using the full model forward with output_hidden_states
            print("[WARN] AuT submodule not found explicitly; will use full forward outputs")

        # Determine hidden size by a small probe on dummy inputs
        self.hidden_size = self._infer_hidden_size()
        print(f"[INFO] AuT hidden size: {self.hidden_size}")

        # Expose encoding strategy for upstream usage
        # We mark it as 'audio_tower' to keep existing target_modules (q/k/v) logic usable
        self.encoding_strategy = 'audio_tower'

    def _locate_aut_module(self) -> Optional[nn.Module]:
        """
        Try best-effort to locate the AuT backbone inside the loaded model.
        Heuristics: prefer attributes likely to contain audio backbone.
        """
        candidate_attr_names = [
            "audio_tower", "audio_encoder", "audio_backbone", "audio_model", "aut", "auditory", "audio_transformer"
        ]

        for name in candidate_attr_names:
            try:
                mod = getattr(self.model, name, None)
                if isinstance(mod, nn.Module):
                    print(f"[INFO] Found AuT module by attribute: {name} -> {type(mod).__name__}")
                    return mod
            except Exception:
                pass

        # Fallback: scan named_modules for plausible candidates by name
        best = None
        for full_name, mod in self.model.named_modules():
            lname = full_name.lower()
            if any(k in lname for k in ["audio", "aut", "auditory"]) and isinstance(mod, nn.Module):
                # Prefer a module that has stacked layers
                if hasattr(mod, "layers") or hasattr(mod, "encoder") or hasattr(mod, "blocks"):
                    print(f"[INFO] Found AuT-like module by scan: {full_name} -> {type(mod).__name__}")
                    best = mod
                    break
        return best

    def _infer_hidden_size(self) -> int:
        # Construct a tiny dummy feature batch via processor
        dummy = np.zeros(16000, dtype=np.float32)
        proc = self.processor(
            audio=[dummy],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        # Move to device
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in proc.items()}

        with torch.no_grad():
            if self.aut_module is not None:
                # Prefer direct AuT path if accessible
                feats = inputs.get("input_features", None)
                if feats is None:
                    # Some processors may use different key; fall back to running full model
                    outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                    last = self._pick_last_hidden(outputs)
                else:
                    out = self.aut_module(feats)
                    last = getattr(out, "last_hidden_state", None)
                    if last is None:
                        # For plain tensor return
                        last = out[0] if isinstance(out, (tuple, list)) else out
            else:
                outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                last = self._pick_last_hidden(outputs)

        if last is None:
            raise RuntimeError("Failed to infer AuT hidden size: last hidden state is None")
        return int(last.shape[-1])

    @staticmethod
    def _pick_last_hidden(model_outputs) -> Optional[torch.Tensor]:
        if model_outputs is None:
            return None
        if hasattr(model_outputs, "hidden_states") and model_outputs.hidden_states is not None:
            return model_outputs.hidden_states[-1]
        if hasattr(model_outputs, "last_hidden_state"):
            return model_outputs.last_hidden_state
        if isinstance(model_outputs, (tuple, list)) and len(model_outputs) > 0:
            return model_outputs[0]
        return None

    def get_hidden_size(self) -> int:
        return self.hidden_size

    def _build_inputs(self, audio_list: List[np.ndarray]):
        # Follow Qwen3 preprocessing implicitly via processor; set sr=16000
        proc = self.processor(
            audio=audio_list,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        inputs = {}
        for k, v in proc.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
            else:
                inputs[k] = v
        # Normalize feature_attention_mask to [B, T]
        fam = inputs.get("feature_attention_mask")
        if isinstance(fam, torch.Tensor):
            if fam.dim() == 3 and fam.shape[1] == 1:
                fam = fam.squeeze(1)
            if fam.dtype != torch.bool:
                fam = fam != 0
            inputs["feature_attention_mask"] = fam
        return inputs

    @staticmethod
    def _to_numpy(audio_input: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(audio_input, torch.Tensor):
            arr = audio_input.detach().cpu().float().numpy()
        else:
            arr = np.asarray(audio_input, dtype=np.float32)
        # Force mono
        if arr.ndim > 1:
            arr = np.mean(arr, axis=0)
        # Safety checks
        if arr.size == 0:
            arr = np.zeros(16000, dtype=np.float32)
        if np.isnan(arr).any() or np.isinf(arr).any():
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        # Normalize to prevent clipping
        m = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
        if m > 0:
            arr = (arr / m) * 0.95
        return arr.astype(np.float32)

    @staticmethod
    def _downsample_mask_to_length(feature_mask: torch.Tensor, target_len: int) -> torch.Tensor:
        if not isinstance(feature_mask, torch.Tensor):
            raise ValueError("feature_mask must be a torch.Tensor")
        if feature_mask.dtype != torch.float32:
            feature_mask = feature_mask.float()
        x = feature_mask.unsqueeze(1)  # [B, 1, T]
        y = F.adaptive_max_pool1d(x, output_size=target_len)
        return (y.squeeze(1) > 0.5)

    def _extract_embeddings(self, inputs: dict) -> torch.Tensor:
        feats = inputs.get("input_features")
        fam = inputs.get("feature_attention_mask")

        with torch.set_grad_enabled(torch.is_grad_enabled()):
            if self.aut_module is not None:
                out = self.aut_module(feats)
                hidden = getattr(out, "last_hidden_state", None)
                if hidden is None:
                    hidden = out[0] if isinstance(out, (tuple, list)) else out
            else:
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = self._pick_last_hidden(outputs)

        if hidden is None:
            raise RuntimeError("AuT forward produced no hidden states")

        # Masked mean pooling along time
        if fam is not None:
            # Downsample mask to match hidden length (AuT is ~8x shorter)
            if fam.shape[-1] != hidden.shape[1]:
                fam_ds = self._downsample_mask_to_length(fam, hidden.shape[1])
            else:
                fam_ds = fam
            mask = fam_ds.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        else:
            pooled = hidden.mean(dim=1)

        # Ensure float32 for stable projection downstream
        if pooled.dtype != torch.float32:
            pooled = pooled.float()
        return pooled

    def predict(self, audio_inputs: List[Union[np.ndarray, torch.Tensor]], dynamic_padding: bool = True) -> torch.Tensor:
        # Convert to numpy, honor dynamic_padding implicitly through processor
        processed = [self._to_numpy(x) for x in audio_inputs]
        inputs = self._build_inputs(processed)
        embeddings = self._extract_embeddings(inputs)
        return embeddings



