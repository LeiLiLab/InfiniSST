import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Union, Optional
import os
import inspect

try:
    from transformers import (
        AutoConfig,
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeProcessor,
    )
except Exception as _e:
    AutoConfig = None
    Qwen3OmniMoeForConditionalGeneration = None
    Qwen3OmniMoeProcessor = None


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

    def __init__(self, model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct", device: str = "cuda"):
        self.device = device
        self.model_name = model_name

        if Qwen3OmniMoeProcessor is None or Qwen3OmniMoeForConditionalGeneration is None:
            raise RuntimeError("transformers is required to load AuT model")

        print(f"[INFO] Loading AuT model: {model_name}")
        
        # ============ Diagnostic Section: Check Environment ============
        print("\n" + "="*80)
        print("🔍 DIAGNOSTIC: Checking Environment & Dependencies")
        print("="*80)
        
        # Check transformers version
        try:
            import transformers
            print(f"✅ transformers version: {transformers.__version__}")
        except Exception as e:
            print(f"❌ Failed to get transformers version: {e}")
        
        # Check for HF tokens
        print("\n📋 Checking HuggingFace authentication tokens:")
        hf_token = (
            os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        )
        
        token_sources = {
            "HUGGINGFACE_HUB_TOKEN": os.environ.get("HUGGINGFACE_HUB_TOKEN"),
            "HF_TOKEN": os.environ.get("HF_TOKEN"),
            "HUGGINGFACEHUB_API_TOKEN": os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        }
        
        for key, val in token_sources.items():
            if val:
                print(f"  ✅ {key}: {'*' * min(10, len(val))}... (found, length={len(val)})")
            else:
                print(f"  ❌ {key}: Not set")
        
        if hf_token:
            print(f"✅ Using token from environment (length: {len(hf_token)})")
        else:
            print("⚠️  No HF token found - may fail on gated/private repos")
        
        # Check model accessibility
        print(f"\n🔍 Checking if model '{model_name}' is accessible...")
        try:
            from huggingface_hub import model_info
            info = model_info(model_name, token=hf_token) if hf_token else model_info(model_name)
            print(f"✅ Model found on HuggingFace Hub")
            print(f"   - Model ID: {info.modelId}")
            print(f"   - Private: {info.private if hasattr(info, 'private') else 'Unknown'}")
            print(f"   - Gated: {info.gated if hasattr(info, 'gated') else 'Unknown'}")
        except Exception as e:
            print(f"⚠️  Could not fetch model info: {e}")
            print(f"   This might be OK if model is public and accessible")
        
        print("="*80 + "\n")
        # ============ End Diagnostic Section ============
        
        # Load processor with remote code and token
        print("[STEP 1/3] Loading processor...")
        try:
            if hf_token:
                full_processor = Qwen3OmniMoeProcessor.from_pretrained(
                    model_name, trust_remote_code=True, token=hf_token
                )
            else:
                full_processor = Qwen3OmniMoeProcessor.from_pretrained(
                    model_name, trust_remote_code=True
                )
            self.processor = full_processor
            # Prefer dedicated audio processor to avoid text requirements
            self.audio_processor = getattr(full_processor, "audio_processor", None)
            if self.audio_processor is None:
                self.audio_processor = getattr(full_processor, "feature_extractor", None)
            if self.audio_processor is not None:
                print("✅ Audio processor is available (no text needed)")
            else:
                print("⚠️  No standalone audio processor; will fallback to full processor with dummy text if needed")
        except Exception as e:
            print(f"❌ Failed to load processor: {e}")
            print(f"   Trying alternative parameter names...")
            try:
                if hf_token:
                    full_processor = Qwen3OmniMoeProcessor.from_pretrained(
                        model_name, trust_remote_code=True, use_auth_token=hf_token
                    )
                else:
                    full_processor = Qwen3OmniMoeProcessor.from_pretrained(
                        model_name, trust_remote_code=True
                    )
                self.processor = full_processor
                self.audio_processor = getattr(full_processor, "audio_processor", None)
                if self.audio_processor is None:
                    self.audio_processor = getattr(full_processor, "feature_extractor", None)
                if self.audio_processor is not None:
                    print("✅ Audio processor available with use_auth_token")
                else:
                    print("✅ Processor loaded with use_auth_token (no standalone audio processor)")
            except Exception as e2:
                print(f"❌ Failed again: {e2}")
                raise

        # Try to load config (optional, for logging). If it fails, continue.
        print("[STEP 2/3] Loading model config (optional)...")
        config = None
        if AutoConfig is not None:
            try:
                if hf_token:
                    config = AutoConfig.from_pretrained(
                        model_name, trust_remote_code=True, token=hf_token
                    )
                else:
                    config = AutoConfig.from_pretrained(
                        model_name, trust_remote_code=True
                    )
                print(f"✅ Config loaded successfully")
                print(f"   - Model type: {getattr(config, 'model_type', 'Unknown')}")
                print(f"   - Architecture: {getattr(config, 'architectures', 'Unknown')}")
            except Exception as e:
                print(f"⚠️  Skipping config load due to error: {e}")

        # Load model using the fetched config
        # Qwen3OmniMoeForConditionalGeneration is a CausalLM model
        print("[STEP 3/3] Loading model weights...")
        
        # Try loading with Flash Attention 2 first, then fall back to eager
        load_kwargs = {
            "dtype": torch.float16,  # Use 'dtype' instead of deprecated 'torch_dtype'
            "device_map": None,
            "trust_remote_code": True,
        }
        if hf_token:
            load_kwargs["token"] = hf_token
        
        # Try Flash Attention 2 first
        try:
            print("   Attempting to load with Flash Attention 2...")
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                model_name,
                attn_implementation="flash_attention_2",
                **load_kwargs
            ).to(device)
            print(f"✅ Model loaded successfully with Flash Attention 2 on {device}")
        except Exception as e:
            print(f"⚠️  Flash Attention 2 not available: {str(e)[:100]}")
            print(f"   Falling back to eager attention (slower but compatible)...")
            try:
                self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                    model_name,
                    attn_implementation="eager",
                    **load_kwargs
                ).to(device)
                print(f"✅ Model loaded successfully with eager attention on {device}")
            except Exception as e2:
                print(f"❌ Failed to load model: {e2}")
                raise

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
        
        # Introspect forward signatures to avoid unexpected kwargs
        self.model_forward_params = set()
        try:
            self.model_forward_params = set(inspect.signature(self.model.forward).parameters.keys())
            print(f"[DEBUG] model.forward params: {sorted(self.model_forward_params)}")
        except Exception as _:
            pass
        self.aut_forward_params = set()
        try:
            if self.aut_module is not None and hasattr(self.aut_module, "forward"):
                self.aut_forward_params = set(inspect.signature(self.aut_module.forward).parameters.keys())
                print(f"[DEBUG] aut.forward params: {sorted(self.aut_forward_params)}")
        except Exception as _:
            pass

        # Print concise model structure before probing size
        try:
            self._print_model_structure()
        except Exception:
            pass

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
        if self.audio_processor is not None:
            proc = self.audio_processor(
                raw_speech=[dummy],
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )
        else:
            proc = self.processor(
                text=["<|audio_bos|>"],
                audio=[dummy],
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )
        # Move to device
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in proc.items()}

        with torch.no_grad():
            # Prefer direct AuT path with proper feature lengths
            feats = inputs.get("input_features")
            fam = inputs.get("feature_attention_mask") or inputs.get("attention_mask")
            if feats is None and "input_values" in inputs:
                feats = inputs["input_values"]
            if self.aut_module is not None and feats is not None:
                call_kwargs = {}
                # Always pass features by name
                if "input_features" in self.aut_forward_params:
                    call_kwargs["input_features"] = feats
                else:
                    call_kwargs["input_features"] = feats
                # Ensure dtype/device and shape match conv2d expectations
                try:
                    model_dtype = next(self.model.parameters()).dtype
                except Exception:
                    model_dtype = torch.float16
                feats_norm = self._normalize_input_features(call_kwargs["input_features"])  # -> [B*,1,mel,T]
                feats_norm = feats_norm.to(device=self.device, dtype=model_dtype)
                call_kwargs["input_features"] = feats_norm
                # Build feature lengths aligned with normalized features batch
                batch_n = feats_norm.shape[0]
                time_dim = feats_norm.shape[-1]
                if fam is not None and fam.dim() >= 2:
                    # Reduce fam to [B*] if possible; otherwise fallback to time_dim
                    fam_r = fam
                    if fam_r.dim() == 3 and fam_r.shape[1] == 1:
                        fam_r = fam_r.squeeze(1)
                    if fam_r.dim() == 2 and fam_r.shape[0] == batch_n:
                        feature_lens = (fam_r > 0).sum(dim=-1).to(dtype=torch.long, device=feats_norm.device)
                    else:
                        feature_lens = torch.full((batch_n,), time_dim, dtype=torch.long, device=feats_norm.device)
                else:
                    feature_lens = torch.full((batch_n,), time_dim, dtype=torch.long, device=feats_norm.device)
                if "feature_lens" in self.aut_forward_params:
                    call_kwargs["feature_lens"] = feature_lens
                elif "feature_lengths" in self.aut_forward_params:
                    call_kwargs["feature_lengths"] = feature_lens
                out = self.aut_module(**call_kwargs)
                last = getattr(out, "last_hidden_state", None)
                if last is None:
                    last = out[0] if isinstance(out, (tuple, list)) else out
            else:
                # Fallback: use full model forward with filtered kwargs
                call_kwargs = {k: v for k, v in inputs.items() if k in self.model_forward_params}
                # Enable hidden states via config to avoid extra kwargs
                try:
                    if hasattr(self.model, "config") and hasattr(self.model.config, "output_hidden_states"):
                        self.model.config.output_hidden_states = True
                except Exception:
                    pass
                outputs = self.model(**call_kwargs)
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
        # Prefer audio-only processor if available to avoid text dependency
        if self.audio_processor is not None:
            proc = self.audio_processor(
                raw_speech=audio_list,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )
        else:
            # Fallback to full processor with minimal dummy text
            batch_size = len(audio_list)
            dummy_texts = ["<|audio_bos|>"] * batch_size
            proc = self.processor(
                text=dummy_texts,
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
        return inputs

    def _print_model_structure(self) -> None:
        print("\n" + "-"*80)
        print(f"[STRUCT] Model class: {self.model.__class__.__name__}")
        # Top-level interesting submodules
        interesting = [
            "audio_tower", "audio_encoder", "audio_backbone", "audio_model",
            "vision_tower", "text_model", "lm", "backbone"
        ]
        for name in interesting:
            try:
                mod = getattr(self.model, name)
                if isinstance(mod, nn.Module):
                    print(f"[STRUCT] .{name}: {mod.__class__.__name__}")
            except Exception:
                pass
        if self.aut_module is not None:
            print(f"[STRUCT] AuT module: {self.aut_module.__class__.__name__}")
            # Try to report layer counts
            candidates = [
                ("layers", getattr(self.aut_module, "layers", None)),
                ("encoder", getattr(self.aut_module, "encoder", None)),
                ("blocks", getattr(self.aut_module, "blocks", None)),
            ]
            for label, obj in candidates:
                try:
                    if isinstance(obj, (nn.ModuleList, list, tuple)):
                        print(f"[STRUCT] AuT.{label}: {len(obj)} layers")
                    elif isinstance(obj, nn.Module) and hasattr(obj, "layers"):
                        l = getattr(obj, "layers")
                        if isinstance(l, (nn.ModuleList, list, tuple)):
                            print(f"[STRUCT] AuT.{label}.layers: {len(l)} layers")
                except Exception:
                    pass
        # Forward param recaps
        if self.model_forward_params:
            print(f"[STRUCT] model.forward params: {sorted(self.model_forward_params)}")
        if self.aut_forward_params:
            print(f"[STRUCT] aut.forward params: {sorted(self.aut_forward_params)}")
        print("-"*80 + "\n")

    @staticmethod
    def _normalize_input_features(feats: torch.Tensor) -> torch.Tensor:
        """
        Ensure input_features shape is 4D NCHW expected by conv2d: [B, 1, 128, T].
        Accepts common variants and fixes dtypes/squeezes safely.
        """
        if feats is None:
            return feats
        x = feats
        # Remove trivial trailing dim if present: [B, C, H, W, 1] -> [B, C, H, W]
        if x.dim() == 5 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        # If 3D [B, 128, T] -> add channel dim
        if x.dim() == 3 and x.shape[1] in (80, 96, 128):
            x = x.unsqueeze(1)
        # If 4D but channel dim not present, try to reshape to [B, 1, 128, T]
        if x.dim() == 4:
            b, a, c, d = x.shape
            # Accept [B, 1, 128, T] as-is
            if a == 1 and c in (80, 96, 128):
                return x
            # Accept [B, 128, T, 1] -> permute to [B, 1, 128, T]
            if a in (80, 96, 128) and d == 1:
                x = x.permute(0, 3, 1, 2)  # [B,1,128,T]
                return x
            # Fallback: if looks like [B, 1, T, 128] -> [B,1,128,T]
            if a == 1 and d in (80, 96, 128):
                x = x.permute(0, 1, 3, 2)
                return x
        return x

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
        with torch.set_grad_enabled(torch.is_grad_enabled()):
            # Always use full model forward to ensure proper preprocessing
            # (Qwen3-Omni audio encoder needs attention masks and length info)
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            
            # For Qwen3-Omni, we need to extract audio embeddings specifically
            # The model outputs combined text+audio hidden states
            # We'll use the audio tower's output directly if available
            if hasattr(outputs, 'audio_hidden_states') and outputs.audio_hidden_states is not None:
                # Direct audio hidden states
                hidden = outputs.audio_hidden_states[-1] if isinstance(outputs.audio_hidden_states, (list, tuple)) else outputs.audio_hidden_states
            elif self.aut_module is not None and hasattr(outputs, 'hidden_states'):
                # Try to locate audio embeddings in the hidden states
                # For multimodal models, audio usually comes before text in the sequence
                hidden = outputs.hidden_states[-1]
                
                # Get audio length from attention_mask if available
                attn_mask = inputs.get('attention_mask')
                if attn_mask is not None:
                    # Audio tokens are usually at the beginning of the sequence
                    # We'll take a reasonable portion assuming audio dominates
                    # (since our dummy text is minimal: "<|audio_bos|>")
                    audio_len = attn_mask.sum(dim=1, keepdim=True) - 5  # Subtract ~5 for text tokens
                    audio_len = torch.clamp(audio_len, min=1)
                    # Take audio portion (first N tokens)
                    max_audio_len = int(audio_len.max().item())
                    hidden = hidden[:, :max_audio_len, :]
            else:
                hidden = self._pick_last_hidden(outputs)

        if hidden is None:
            raise RuntimeError("AuT forward produced no hidden states")

        # Mean pooling along time dimension
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



