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
from contextlib import nullcontext

AUDIO_PROMPT = "<|audio_bos|><|AUDIO|><|audio_eos|>"

def downsample_mask_to(hidden_len: int, feature_mask: torch.Tensor) -> torch.Tensor:
    """Downsample a feature-level mask [B, L_feat] to [B, L_hidden] using max-pooling.
    Returns a boolean mask of shape [B, L_hidden]."""
    if not isinstance(feature_mask, torch.Tensor):
        raise ValueError("feature_mask must be a torch.Tensor")
    if feature_mask.dtype != torch.float32:
        feature_mask = feature_mask.float()
    # [B, L] -> [B, 1, L]
    m = feature_mask.unsqueeze(1)
    # Adaptive max pool to target length
    m_ds = F.adaptive_max_pool1d(m, output_size=hidden_len)  # [B, 1, L_hidden]
    return (m_ds.squeeze(1) > 0.5)

def build_qwen2_audio_inputs(processor, audio_np, device, max_mel_len=3000):
    """
    Build inputs for Qwen2-Audio strictly via its processor to keep
    the alignment between audio features and audio tokens.
    Do NOT manually resize/interpolate features; let the processor
    decide correct shapes and masks.
    
    Args:
        processor: Qwen2Audio的processor
        audio_np: numpy音频数据或列表
        device: 目标设备
        max_mel_len: 保留参数兼容性，但固定为3000
    
    Returns:
        处理好的inputs字典
    """
    # 确保audio_np是列表格式（即使只有一个音频）
    if isinstance(audio_np, np.ndarray):
        audio_list = [audio_np]
    else:
        audio_list = audio_np
    
    # 为每个音频创建对应的文本（都是AUDIO_PROMPT）
    text_list = [AUDIO_PROMPT] * len(audio_list)
    
    # 使用单次processor联合处理text+audio，避免分离调用
    try:
        proc = processor(
            text=text_list,
            audio=audio_list,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )
    except Exception as e:
        # 如果联合处理失败，回退到分离处理（兼容性）
        print(f"[WARN] Joint processing failed ({e}), falling back to separate processing")
        
        # 分别处理（原方法）
        text_inputs = processor.tokenizer(
            text_list[0] if len(text_list) == 1 else text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True
        )
        
        audio_inputs = processor.feature_extractor(
            audio_list[0] if len(audio_list) == 1 else audio_list,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )
        
        proc = {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "input_features": audio_inputs["input_features"],
            "feature_attention_mask": audio_inputs.get("attention_mask", None)
        }

    # 将 processor 产物移到目标设备，不做任何形状篡改
    inputs = {}
    for k, v in proc.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
        else:
            inputs[k] = v

    # 兜底：若processor未提供 feature_attention_mask，则基于 input_features 生成全1掩码
    if "feature_attention_mask" not in inputs:
        feats = inputs.get("input_features", None)
        if isinstance(feats, torch.Tensor):
            # 通常时间维是最后一维，生成 [B, T]
            time_len = feats.shape[-1]
            inputs["feature_attention_mask"] = torch.ones(
                (feats.shape[0], time_len), dtype=torch.bool, device=device
            )
    else:
        fam = inputs.get("feature_attention_mask")
        if isinstance(fam, torch.Tensor):
            # squeeze [B,1,T] -> [B,T] if needed
            if fam.dim() == 3 and fam.shape[1] == 1:
                fam = fam.squeeze(1)
            # 转为bool以兼容SDPA要求
            if fam.dtype != torch.bool:
                fam = fam != 0
            inputs["feature_attention_mask"] = fam
    return inputs


def calculate_feature_length(audio_duration_seconds, sampling_rate=16000, 
                           hop_length=160, frame_length=400):
    """
    计算音频对应的特征长度
    
    Args:
        audio_duration_seconds: 音频时长（秒）
        sampling_rate: 采样率，Qwen2-Audio 使用 16kHz
        hop_length: 帧移，通常是 160 samples (10ms at 16kHz)
        frame_length: 帧长，通常是 400 samples (25ms at 16kHz)
    
    Returns:
        特征序列长度
    """
    audio_samples = int(audio_duration_seconds * sampling_rate)
    # 计算特征帧数：(samples - frame_length) // hop_length + 1
    feature_frames = (audio_samples - frame_length) // hop_length + 1
    return max(1, feature_frames)  # 至少1帧


def get_dynamic_audio_chunk_length(audio_batch, default_max_seconds=5.0):
    """
    基于批次内音频长度动态计算音频波形的处理长度
    注意：这只影响波形层面的裁剪，mel特征长度固定为3000
    
    Args:
        audio_batch: 音频数据列表
        default_max_seconds: 默认最大秒数
    
    Returns:
        动态计算的音频波形长度（样本数）
    """
    if not audio_batch:
        return int(default_max_seconds * 16000)  # 默认长度
    
    # 计算批次内最长音频的长度
    max_samples = 0
    for audio in audio_batch:
        if isinstance(audio, (np.ndarray, torch.Tensor)):
            if isinstance(audio, torch.Tensor):
                samples = audio.numel()
            else:
                samples = len(audio.flatten())
            max_samples = max(max_samples, samples)
    
    # 限制在合理范围内
    max_samples = min(max_samples, int(default_max_seconds * 16000))
    
    return max_samples


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
            device_map=None  # 不使用自动设备分配，手动控制
        ).to(device)
        try:
            if hasattr(self.model, "config"):
                setattr(self.model.config, "use_cache", False)
            self.model.gradient_checkpointing_enable()
            print("[INFO] Enabled gradient checkpointing and disabled use_cache")
        except Exception as e:
            print(f"[WARN] Failed to enable gradient checkpointing: {e}")
        # 不要强制设置为 eval 模式，让上层模块控制 train/eval 状态
        # self.model.eval()  # REMOVED: 这会阻止梯度传播
        
        print(f"[INFO] Qwen2-Audio model loaded successfully on {device}")
    
        # 分析并确定模型结构
        self._analyze_model_structure()
    
    def _print_module_tree(self, module, prefix="", max_depth=3, current_depth=0):
        """递归打印模块结构树"""
        if current_depth >= max_depth:
            return
        
        # 打印子模块
        children = list(module.named_children())
        for i, (name, child) in enumerate(children):
            is_last = (i == len(children) - 1)
            connector = "└── " if is_last else "├── "
            
            # 获取模块信息
            module_type = type(child).__name__
            
            # 统计参数
            num_params = sum(p.numel() for p in child.parameters())
            trainable_params = sum(p.numel() for p in child.parameters() if p.requires_grad)
            
            param_info = f"params={num_params:,}"
            if trainable_params > 0:
                param_info += f" (trainable={trainable_params:,})"
            
            print(f"  {prefix}{connector}{name}: {module_type} [{param_info}]")
            
            # 递归打印子模块
            extension = "    " if is_last else "│   "
            self._print_module_tree(child, prefix + extension, max_depth, current_depth + 1)
    
    def _analyze_model_structure(self):
        """分析并缓存模型结构信息，避免运行时多分支判断"""
        print("\n" + "="*80)
        print("🔍 QWEN2-AUDIO MODEL STRUCTURE ANALYSIS")
        print("="*80)
        
        # ==================== 第1步：打印模型基本信息 ====================
        print(f"\n[STEP 1] Model Basic Information:")
        print(f"  Model class: {type(self.model).__name__}")
        print(f"  Model module: {type(self.model).__module__}")
        
        # ==================== 第2步：列出所有顶层属性 ====================
        print(f"\n[STEP 2] All Top-Level Attributes:")
        all_attrs = [attr for attr in dir(self.model) if not attr.startswith('_')]
        print(f"  Total attributes: {len(all_attrs)}")
        
        # 按类型分类
        module_attrs = []
        config_attrs = []
        other_attrs = []
        
        for attr in all_attrs:
            try:
                obj = getattr(self.model, attr)
                if isinstance(obj, nn.Module):
                    module_attrs.append((attr, type(obj).__name__))
                elif 'config' in attr.lower() or attr == 'config':
                    config_attrs.append((attr, type(obj).__name__))
                elif not callable(obj):  # 排除方法
                    other_attrs.append((attr, type(obj).__name__))
            except:
                pass
        
        print(f"\n  📦 Module attributes ({len(module_attrs)}):")
        for name, type_name in module_attrs[:20]:  # 只显示前20个
            print(f"    - {name}: {type_name}")
        if len(module_attrs) > 20:
            print(f"    ... and {len(module_attrs) - 20} more")
        
        print(f"\n  ⚙️  Config attributes ({len(config_attrs)}):")
        for name, type_name in config_attrs:
            print(f"    - {name}: {type_name}")
        
        # ==================== 第3步：检查audio_tower ====================
        print(f"\n[STEP 3] Inspecting audio_tower:")
        if hasattr(self.model, 'audio_tower'):
            audio_tower = self.model.audio_tower
            print(f"  ✅ Found 'audio_tower': {type(audio_tower).__name__}")
            print(f"     Sub-modules:")
            for sub_name, sub_module in audio_tower.named_children():
                print(f"       - {sub_name}: {type(sub_module).__name__}")
            
            # 检查layers
            if hasattr(audio_tower, 'layers'):
                num_layers = len(audio_tower.layers)
                print(f"     ✅ Has {num_layers} transformer layers")
                if num_layers > 0:
                    print(f"        First layer sub-modules:")
                    for name, mod in audio_tower.layers[0].named_children():
                        print(f"          - {name}: {type(mod).__name__}")
        else:
            print(f"  ❌ No 'audio_tower' - will use full_forward strategy")
        
        # ==================== 第4步：检查 language_model ====================
        print(f"\n[STEP 4] Inspecting language_model:")
        if hasattr(self.model, 'language_model'):
            lm = self.model.language_model
            print(f"  ✅ Found 'language_model': {type(lm).__name__}")
            if hasattr(lm, 'config'):
                hidden_size = lm.config.hidden_size
                num_layers = getattr(lm.config, 'num_hidden_layers', 'unknown')
                print(f"     Config: hidden_size={hidden_size}, num_layers={num_layers}")
        else:
            print(f"  ❌ No 'language_model'")
        
        # ==================== 第5步：检查 config 了解模型架构 ====================
        print(f"\n[STEP 5] Model Config Analysis:")
        if hasattr(self.model, 'config'):
            config = self.model.config
            print(f"  Config type: {type(config).__name__}")
            
            # 打印关键配置
            important_config_keys = ['hidden_size', 'num_hidden_layers', 'num_attention_heads',
                                    'audio_config', 'text_config', 'encoder_hidden_size']
            
            for key in important_config_keys:
                if hasattr(config, key):
                    value = getattr(config, key)
                    print(f"  - {key}: {value}")
            
            # 打印所有配置键
            all_config_keys = [k for k in dir(config) if not k.startswith('_')]
            print(f"\n  All config keys ({len(all_config_keys)}):")
            for key in all_config_keys[:30]:  # 只显示前30个
                try:
                    value = getattr(config, key)
                    if not callable(value):
                        print(f"    - {key}: {type(value).__name__}")
                except:
                    pass
        
        # ==================== 第6步：递归打印模型结构树 ====================
        print(f"\n[STEP 6] Model Structure Tree (first 3 levels):")
        self._print_module_tree(self.model, max_depth=3)
        
        # ==================== 第7步：查找可以应用 LoRA 的模块 ====================
        print(f"\n[STEP 7] Finding LoRA Target Modules:")
        
        # 常见的 LoRA target module 名称
        common_targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                         'gate_proj', 'up_proj', 'down_proj',
                         'query', 'key', 'value', 'dense']
        
        found_targets = {}
        for name, module in self.model.named_modules():
            # 获取模块名的最后一部分
            module_name = name.split('.')[-1]
            if module_name in common_targets:
                if module_name not in found_targets:
                    found_targets[module_name] = []
                found_targets[module_name].append(name)
        
        if found_targets:
            print(f"  Found potential LoRA target modules:")
            for target_name, locations in found_targets.items():
                print(f"    '{target_name}' appears {len(locations)} times:")
                # 显示前3个位置
                for loc in locations[:3]:
                    print(f"      - {loc}")
                if len(locations) > 3:
                    print(f"      ... and {len(locations) - 3} more locations")
        else:
            print(f"  ❌ No common LoRA target modules found!")
            print(f"  Available module names (first 50):")
            all_module_names = set()
            for name, _ in self.model.named_modules():
                module_name = name.split('.')[-1]
                if module_name and not module_name.startswith('_'):
                    all_module_names.add(module_name)
            for i, name in enumerate(sorted(all_module_names)[:50]):
                print(f"    - {name}")
        
        print("\n" + "="*80)
        
        # 现在基于实际检查结果进行配置
        print(f"\n[STEP 8] Determining Encoding Strategy Based on Analysis:")
        
        # 先打印模型的顶层属性，了解实际结构
        important_attrs = ['audio_tower', 'audio_encoder', 'language_model', 'lm_head', 'config']
        for attr in important_attrs:
            has_it = hasattr(self.model, attr)
            obj = getattr(self.model, attr, None) if has_it else None
            print(f"  - {attr}: {has_it} {'(type: ' + type(obj).__name__ + ')' if obj is not None else ''}")
        
        # 1. 检查audio tower
        # 根据实际模型结构，Qwen2-Audio 使用 'audio_tower'
        self.has_audio_tower = hasattr(self.model, 'audio_tower') and self.model.audio_tower is not None
        self.audio_tower_name = 'audio_tower' if self.has_audio_tower else None
        
        print(f"[STRUCT] Has audio_tower: {self.has_audio_tower}")
        if self.has_audio_tower:
            print(f"[STRUCT] Audio module name: '{self.audio_tower_name}'")
        
        if self.has_audio_tower:
            audio_tower = self.model.audio_tower
            print(f"[STRUCT] Audio tower type: {type(audio_tower).__name__}")
            
            # 根据模型config直接获取hidden dimension，不需要测试
            if hasattr(self.model.config, 'audio_config'):
                self.audio_hidden_dim = self.model.config.audio_config.d_model
                print(f"[STRUCT] Audio hidden dimension (from config): {self.audio_hidden_dim}")
            else:
                # Fallback: 测试获取
                print(f"[STRUCT] Testing audio tower output format...")
                dummy_input = torch.randn(1, 128, 80, device=self.device, dtype=torch.float16)
                with torch.no_grad():
                    try:
                        test_output = audio_tower(dummy_input)
                        if hasattr(test_output, 'last_hidden_state'):
                            self.audio_hidden_dim = test_output.last_hidden_state.shape[-1]
                        elif isinstance(test_output, tuple):
                            self.audio_hidden_dim = test_output[0].shape[-1]
                        elif isinstance(test_output, torch.Tensor):
                            self.audio_hidden_dim = test_output.shape[-1]
                        print(f"[STRUCT] Audio hidden dimension (from test): {self.audio_hidden_dim}")
                    except Exception as e:
                        print(f"[ERROR] Failed to determine audio hidden dim: {e}")
                        self.has_audio_tower = False
            
            # Qwen2-Audio 的 audio_tower 输出 BaseModelOutput
            self.audio_tower_output_type = 'BaseModelOutput'
            print(f"[STRUCT] Audio tower output type: {self.audio_tower_output_type}")
        
        # 2. 检查language model
        self.has_language_model = hasattr(self.model, 'language_model') and self.model.language_model is not None
        print(f"[STRUCT] Has language_model: {self.has_language_model}")
        
        if self.has_language_model:
            self.language_model_hidden_dim = self.model.language_model.config.hidden_size
            print(f"[STRUCT] Language model hidden dimension: {self.language_model_hidden_dim}")
        else:
            # 从顶层config获取
            self.language_model_hidden_dim = self.model.config.hidden_size
            print(f"[STRUCT] Hidden dimension (from top config): {self.language_model_hidden_dim}")
        
        # 3. 确定使用的编码路径
        if self.has_audio_tower:
            self.encoding_strategy = 'audio_tower'
            self.hidden_size = self.audio_hidden_dim
            print(f"[STRUCT] ✅ Will use AUDIO_TOWER for encoding (recommended)")
        else:
            self.encoding_strategy = 'full_forward'
            self.hidden_size = self.language_model_hidden_dim
            print(f"[STRUCT] ⚠️  Will use FULL_FORWARD for encoding (fallback)")
        
        print(f"[STRUCT] Final hidden size for projection: {self.hidden_size}")
        print("="*80 + "\n")
    
    def get_shared_model(self):
        """Return the model and processor for sharing with text encoder"""
        return {
            'model': self.model,
            'processor': self.processor
        }
    
    def get_hidden_size(self):
        """获取音频编码器的hidden size - 使用缓存的值"""
        return self.hidden_size
    
    def predict(self, audio_inputs: List[Union[str, np.ndarray, torch.Tensor]], max_length: int = None, dynamic_padding: bool = True) -> torch.Tensor:
        """
        Extract audio embeddings from audio files or tensors
        
        Args:
            audio_inputs: List of audio file paths, numpy arrays, or torch tensors
            max_length: Maximum audio length in samples (optional)
            dynamic_padding: 是否使用动态padding（推荐）
            
        Returns:
            torch tensor of shape [batch_size, embedding_dim]
        """
        # 预处理所有音频数据
        processed_audios = []
        
        for audio_input in audio_inputs:
            try:
                # Load and preprocess audio (support file path or in-memory waveform)
                if isinstance(audio_input, (np.ndarray, torch.Tensor)):
                    if isinstance(audio_input, torch.Tensor):
                        # 确保tensor在CPU上进行numpy转换，但保持原始精度
                        audio_np = audio_input.detach().cpu().float().numpy()
                    else:
                        audio_np = audio_input
                    
                    # 处理多声道音频：转为单声道
                    if audio_np.ndim > 1:
                        audio_np = np.mean(audio_np, axis=0)
                    
                    audio = np.array(audio_np, dtype=np.float32)
                    sr = 16000  # mmap音频数据默认已经是16kHz
                else:
                    # 文件路径输入（向后兼容）
                    audio, sr = librosa.load(audio_input, sr=16000)  # Qwen2-Audio expects 16kHz
                
                # Ensure audio is not empty
                if len(audio) == 0:
                    print(f"[WARN] Empty audio input: {type(audio_input)}")
                    # 创建1秒的静音作为fallback
                    audio = np.zeros(16000, dtype=np.float32)
                
                # Limit audio length if specified (default to 3 seconds max)
                max_samples = max_length if max_length else int(16000 * 3.0)
                if len(audio) > max_samples:
                    audio = audio[:max_samples]
                
                # Ensure minimum length (pad if too short)
                min_samples = 1600  # 0.1 seconds
                if len(audio) < min_samples:
                    audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')
                
                # Convert to float32 and ensure it's a numpy array
                audio = np.array(audio, dtype=np.float32)
                
                # Validate audio data
                if np.isnan(audio).any() or np.isinf(audio).any():
                    raise ValueError(f"Audio input contains NaN or Inf values")
                
                # Normalize audio to prevent extreme values
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio)) * 0.95  # Prevent clipping
                
                processed_audios.append(audio)
                
            except Exception as e:
                print(f"[ERROR] Failed to preprocess audio input {type(audio_input)}: {e}")
                # 使用1秒静音作为fallback
                processed_audios.append(np.zeros(16000, dtype=np.float32))
        
        # 批量处理所有音频（交由processor处理padding/truncation）
        embeddings = self._batch_extract_embeddings(processed_audios)
        
        # 直接返回embeddings，因为_batch_extract_embeddings现在返回正确格式的张量
        return embeddings
    
    def _batch_extract_embeddings(self, audio_batch):
        """批量提取音频embeddings - 使用初始化时确定的策略"""
        # 为每个音频创建对应的文本（都是AUDIO_PROMPT）
        text_list = [AUDIO_PROMPT] * len(audio_batch)
        
        # 使用processor处理text+audio
        inputs = self.processor(
            text=text_list,
            audio=audio_batch,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move inputs to device
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.model.device)
        
        # 根据初始化时确定的策略进行编码
        # CRITICAL: 使用 torch.is_grad_enabled() 而不是 self.model.training
        # 因为我们需要根据当前上下文（而不是模型状态）来决定是否计算梯度
        with torch.set_grad_enabled(torch.is_grad_enabled()):
            if self.encoding_strategy == 'audio_tower':
                embeddings = self._extract_from_audio_tower(inputs)
            else:  # 'full_forward'
                embeddings = self._extract_from_full_forward(inputs)
        
        # CRITICAL: 只在需要时转换数据类型，并保持梯度
        # .float() 不会断开梯度图，但为了确保，我们明确检查
        if embeddings.dtype != torch.float32:
            embeddings = embeddings.float()  # 这个操作保持梯度
        
        return embeddings
    
    def _extract_from_audio_tower(self, inputs):
        """从audio tower提取特征 - 确定性路径，基于Qwen2-Audio结构"""
        audio_features = inputs['input_features']
        feature_attention_mask = inputs.get('feature_attention_mask')
        
        # DEBUG: 检查输入的梯度状态
        if not hasattr(self, '_logged_input_grad'):
            print(f"[DEBUG INPUT] audio_features.requires_grad: {audio_features.requires_grad}")
            print(f"[DEBUG INPUT] audio_features.dtype: {audio_features.dtype}")
            print(f"[DEBUG INPUT] audio_features contains NaN: {torch.isnan(audio_features).any()}")
            print(f"[DEBUG INPUT] audio_features contains Inf: {torch.isinf(audio_features).any()}")
            print(f"[DEBUG INPUT] audio_features min/max: {audio_features.min():.4f}/{audio_features.max():.4f}")
            self._logged_input_grad = True
        
        # CRITICAL: 确保 audio_features 有梯度！
        # Processor 生成的 input_features 默认不需要梯度
        if not audio_features.requires_grad and torch.is_grad_enabled():
            audio_features = audio_features.requires_grad_(True)
            print(f"[DEBUG INPUT] Force enabled requires_grad for audio_features")
        
        # DEBUG: 检查 audio_tower 的第一层参数状态
        if hasattr(self.model, 'base_model') and not hasattr(self, '_logged_lora_layer_status'):
            audio_tower_test = self.model.base_model.model.audio_tower
            first_q_proj = audio_tower_test.layers[0].self_attn.q_proj
            print(f"[DEBUG LORA] First q_proj type: {type(first_q_proj).__name__}")
            if hasattr(first_q_proj, 'lora_A'):
                lora_A = first_q_proj.lora_A['default']
                lora_B = first_q_proj.lora_B['default']
                print(f"[DEBUG LORA] lora_A weight dtype: {lora_A.weight.dtype}, requires_grad: {lora_A.weight.requires_grad}")
                print(f"[DEBUG LORA] lora_B weight dtype: {lora_B.weight.dtype}, requires_grad: {lora_B.weight.requires_grad}")
                print(f"[DEBUG LORA] lora_A training mode: {lora_A.training}")
                print(f"[DEBUG LORA] lora_dropout: {first_q_proj.lora_dropout}")
            self._logged_lora_layer_status = True
        
        # CRITICAL: PEFT 包装后，需要通过 base_model 或 model 访问原始模块
        # 直接访问 self.model.audio_tower 会绕过 PEFT 包装！
        if hasattr(self.model, 'base_model'):
            # PEFT 包装后的模型
            audio_tower = self.model.base_model.model.audio_tower
            if not hasattr(self, '_logged_peft_access'):
                print(f"[DEBUG] Using PEFT-wrapped audio_tower")
                print(f"[DEBUG] First layer q_proj type: {type(audio_tower.layers[0].self_attn.q_proj)}")
                print(f"[DEBUG] Has lora_A: {hasattr(audio_tower.layers[0].self_attn.q_proj, 'lora_A')}")
                self._logged_peft_access = True
        else:
            # 未包装的原始模型
            audio_tower = self.model.audio_tower
            if not hasattr(self, '_logged_direct_access'):
                print(f"[DEBUG] Using direct audio_tower (no PEFT wrapping detected)")
                self._logged_direct_access = True
        
        # Qwen2-Audio: audio_tower 返回 BaseModelOutput
        audio_tower_output = audio_tower(audio_features)
        audio_hidden_states = audio_tower_output.last_hidden_state  # [B, T, 1280]
        
        # DEBUG: 检查 hidden states 的梯度状态
        if not hasattr(self, '_logged_gradient_debug'):
            print(f"[DEBUG] audio_hidden_states.requires_grad: {audio_hidden_states.requires_grad}")
            print(f"[DEBUG] audio_hidden_states.dtype: {audio_hidden_states.dtype}")
            print(f"[DEBUG] audio_hidden_states.shape: {audio_hidden_states.shape}")
            self._logged_gradient_debug = True
        
        # 使用attention mask进行masked pooling
        if feature_attention_mask is not None:
            # 如果长度不匹配，下采样attention mask
            if feature_attention_mask.shape[-1] != audio_hidden_states.shape[1]:
                target_len = audio_hidden_states.shape[1]
                feature_attention_mask = downsample_mask_to(target_len, feature_attention_mask)
            
            # Masked mean pooling
            mask_expanded = feature_attention_mask.unsqueeze(-1).float()  # [B, T, 1]
            masked_hidden = audio_hidden_states * mask_expanded  # [B, T, H]
            pooled_features = masked_hidden.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)  # [B, H]
        else:
            # 简单平均池化
            pooled_features = audio_hidden_states.mean(dim=1)  # [B, H]
        
        # DEBUG: 检查池化后的梯度状态
        if not hasattr(self, '_logged_pooled_debug'):
            print(f"[DEBUG] pooled_features.requires_grad: {pooled_features.requires_grad}")
            print(f"[DEBUG] pooled_features.dtype: {pooled_features.dtype}")
            self._logged_pooled_debug = True
        
        return pooled_features
    
    def _extract_from_full_forward(self, inputs):
        """从完整forward pass提取特征 - 确定性路径，无兜底逻辑"""
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        # 优先使用hidden_states
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            last_hidden_state = outputs.hidden_states[-1]
        else:
            last_hidden_state = outputs.last_hidden_state
        
        # 对序列维度进行平均池化
        pooled_features = last_hidden_state.mean(dim=1)  # [batch_size, hidden_dim]
        
        return pooled_features


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
                device_map=None  # 不使用自动设备分配，手动控制
            ).to(device)

            try:
                if hasattr(self.model, "config"):
                    setattr(self.model.config, "use_cache", False)
                self.model.gradient_checkpointing_enable()
                print("[INFO] Enabled gradient checkpointing and disabled use_cache")
            except Exception as e:
                print(f"[WARN] Failed to enable gradient checkpointing: {e}")
            
            # 不要强制设置为 eval 模式，让上层模块控制 train/eval 状态
            # self.model.eval()  # REMOVED: 这会阻止梯度传播
            print(f"[INFO] Qwen2-Audio text encoder loaded successfully on {device}")
        
        # 分析模型结构
        self._analyze_model_structure()
    
    def _analyze_model_structure(self):
        """分析文本编码器结构"""
        print("\n" + "="*80)
        print("🔍 TEXT ENCODER STRUCTURE ANALYSIS")
        print("="*80)
        
        # 检查language model
        if hasattr(self.model, "language_model") and self.model.language_model is not None:
            self.hidden_size = self.model.language_model.config.hidden_size
            print(f"[STRUCT] Using language_model.hidden_size: {self.hidden_size}")
        else:
            self.hidden_size = self.model.config.hidden_size
            print(f"[STRUCT] Using top-level config.hidden_size: {self.hidden_size}")
        
        print("="*80 + "\n")
    
    def get_hidden_size(self):
        """获取文本编码器的hidden size - 使用缓存的值"""
        return self.hidden_size
    
    def predict(self, texts: List[str], source_lang: str = "eng_Latn") -> torch.Tensor:
        """
        Extract text embeddings from text strings using text encoder hidden layers
        
        Args:
            texts: List of text strings
            source_lang: Source language (kept for compatibility, not used in Qwen2-Audio)
            
        Returns:
            torch tensor of shape [batch_size, embedding_dim]
        """
        # Tokenize all texts at once
        inputs = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move inputs to device
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        
        # 使用梯度上下文（训练时保持梯度，评估时断链）
        with torch.set_grad_enabled(self.model.training):
            # Get text embeddings from the language model
            outputs = self.model.language_model(**inputs, output_hidden_states=True)
            
            # Use the last hidden state and pool it
            last_hidden_state = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
            
            # Masked mean pooling over sequence dimension, excluding padding tokens
            attention_mask = inputs["attention_mask"]
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
            masked_embeddings = last_hidden_state * mask_expanded  # [batch, seq_len, hidden_dim]
            
            # Sum over sequence dimension and divide by actual length (excluding padding)
            pooled_embeddings = masked_embeddings.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)  # [batch, hidden_dim]
        
        # 确保输出为float32
        return pooled_embeddings.float()


class ContrastiveQwen2AudioModel(nn.Module):
    """
    Contrastive Speech-Text Model using Qwen2-Audio encoders with LoRA fine-tuning
    """
    def __init__(self, speech_encoder, text_encoder, proj_dim=512, 
                 lora_r=16, lora_alpha=32, lora_dropout=0.1,
                 speech_hidden_dim=None, text_hidden_dim=None):
        super().__init__()
        self.speech_encoder = speech_encoder
        self.text_encoder = text_encoder
        
        # 自动推断各自的hidden size
        if speech_hidden_dim is None:
            speech_hidden_dim = self.speech_encoder.get_hidden_size()
        if text_hidden_dim is None:
            text_hidden_dim = self.text_encoder.get_hidden_size()
        
        print(f"[INFO] Speech encoder hidden size: {speech_hidden_dim}")
        print(f"[INFO] Text encoder hidden size: {text_hidden_dim}")
        
        # Projection layers (always trainable) - 使用各自的输入维度
        self.proj_speech = nn.Linear(speech_hidden_dim, proj_dim)
        self.proj_text = nn.Linear(text_hidden_dim, proj_dim)
        
        # 根据编码策略决定 LoRA 应用位置
        # 基于实际模型结构：
        # - audio_tower 有: q_proj, k_proj, v_proj (没有 o_proj!)
        # - language_model 有: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        
        if self.speech_encoder.encoding_strategy == 'audio_tower':
            # 只在音频编码器的attention层加LoRA
            print(f"[INFO] LoRA strategy: Applying to AUDIO_TOWER only (audio_tower encoding)")
            target_modules = ["q_proj", "k_proj", "v_proj"]  # audio_tower 没有 o_proj
            print(f"[INFO] Note: audio_tower attention uses q/k/v_proj only (no o_proj)")
        else:
            # 在language model加LoRA（fallback策略，一般不会用到）
            print(f"[INFO] LoRA strategy: Applying to LANGUAGE_MODEL (full_forward encoding)")
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        # CRITICAL: 禁用 lora_dropout 以避免 FP16 下的梯度消失
        # lora_A 的输出在 FP16 下可能很小，经过 dropout 后梯度会下溢
        effective_lora_dropout = 0.0 if lora_dropout > 0 else 0.0
        if lora_dropout > 0:
            print(f"[WARN] LoRA dropout disabled (was {lora_dropout}) to prevent gradient underflow in FP16")
        
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,  # rank
            lora_alpha=lora_alpha,  # scaling parameter
            lora_dropout=effective_lora_dropout,  # 禁用 dropout
            target_modules=target_modules,
            bias="none",
        )
        
        print(f"[INFO] LoRA target modules: {target_modules}")

        # 检查模型结构
        print(f"[DEBUG] Model structure analysis:")
        if hasattr(self.speech_encoder.model, 'audio_tower'):
            print(f"[DEBUG] - Has audio_tower: Yes")
        if hasattr(self.speech_encoder.model, 'language_model'):
            print(f"[DEBUG] - Has language_model: Yes")

        # 用于控制只打印一次的调试信息
        self._logged_speech_shape = False
        
        # 冻结所有原始参数
        for param in self.speech_encoder.model.parameters():
            param.requires_grad = False
        for param in self.text_encoder.model.parameters():
            param.requires_grad = False
        
        # 应用LoRA（只应用一次，因为模型是共享的）
        if self.speech_encoder.model is self.text_encoder.model:
            print(f"[INFO] Applying LoRA to shared Qwen2-Audio model")
            
            # 检查模型结构并打印LoRA将应用到哪些模块
            print(f"[DEBUG] Checking modules before LoRA application:")
            module_count = 0
            target_modules_found = []
            for name, module in self.speech_encoder.model.named_modules():
                if any(target in name for target in self.lora_config.target_modules):
                    module_count += 1
                    target_modules_found.append(name)
                    if module_count <= 10:  # 打印前10个
                        print(f"[DEBUG] - Target module: {name} ({type(module).__name__})")
            print(f"[DEBUG] Total target modules found: {module_count}")
            
            # 确保模型处于训练模式以启用LoRA
            self.speech_encoder.model.train()
            
            # 应用LoRA
            try:
                self.speech_encoder.model = get_peft_model(self.speech_encoder.model, self.lora_config)
                self.text_encoder.model = self.speech_encoder.model  # 保持共享
                print(f"[INFO] LoRA applied successfully")
            except Exception as e:
                print(f"[ERROR] Failed to apply LoRA: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            # 验证LoRA参数的创建和状态
            lora_params_found = 0
            lora_params_trainable = 0
            all_lora_params = []
            lora_params_converted_to_fp32 = 0
            
            for name, param in self.speech_encoder.model.named_parameters():
                if 'lora' in name.lower():
                    lora_params_found += 1
                    all_lora_params.append((name, param))
                    if param.requires_grad:
                        lora_params_trainable += 1
                    else:
                        # 强制启用梯度
                        param.requires_grad = True
                        lora_params_trainable += 1
                        print(f"[DEBUG] Force-enabled gradient for: {name}")
                    
                    # CRITICAL: 将 LoRA 参数转换为 FP32 以避免梯度下溢
                    if param.dtype != torch.float32:
                        param.data = param.data.float()
                        lora_params_converted_to_fp32 += 1
            
            print(f"[DEBUG] LoRA parameters found: {lora_params_found}")
            print(f"[DEBUG] LoRA parameters trainable: {lora_params_trainable}")
            if lora_params_converted_to_fp32 > 0:
                print(f"[INFO] ✅ Converted {lora_params_converted_to_fp32} LoRA parameters to FP32 for stable gradients")
            
            # 详细检查前几个LoRA参数
            print(f"[DEBUG] First few LoRA parameters:")
            for i, (name, param) in enumerate(all_lora_params[:5]):
                print(f"[DEBUG]   {name}: shape={param.shape}, dtype={param.dtype}, requires_grad={param.requires_grad}, device={param.device}")
            
            # 验证LoRA适配器是否正确添加
            if hasattr(self.speech_encoder.model, 'peft_config'):
                print(f"[DEBUG] PEFT config found: {self.speech_encoder.model.peft_config}")
            else:
                print(f"[WARN] No PEFT config found - LoRA may not be applied correctly")
            
            # 检查是否有活跃的适配器
            if hasattr(self.speech_encoder.model, 'active_adapters'):
                print(f"[DEBUG] Active adapters: {self.speech_encoder.model.active_adapters}")
            
        else:
            print(f"[INFO] Applying LoRA to separate speech and text models")
            self.speech_encoder.model.train()
            self.text_encoder.model.train()
            self.speech_encoder.model = get_peft_model(self.speech_encoder.model, self.lora_config)
            self.text_encoder.model = get_peft_model(self.text_encoder.model, self.lora_config)

        # Register the Qwen2 backbone(s) as nn.Module submodules so that
        # downstream optimizers (and DDP) can discover the LoRA parameters.
        # Without an explicit registration, model.parameters() would only expose
        # the projection heads defined on this wrapper module.
        self.add_module("speech_qwen2_model", self.speech_encoder.model)
        if self.speech_encoder.model is not self.text_encoder.model:
            self.add_module("text_qwen2_model", self.text_encoder.model)

        # 计算LoRA参数数量（需要在应用LoRA之后）
        self.actual_lora_params = sum(p.numel() for p in self.speech_encoder.model.parameters() if p.requires_grad)
        if self.speech_encoder.model is not self.text_encoder.model:
            self.actual_lora_params += sum(p.numel() for p in self.text_encoder.model.parameters() if p.requires_grad)
        
        print(f"[INFO] ContrastiveQwen2AudioModel initialized with LoRA (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})")
        print(f"[INFO] LoRA trainable parameters: {self.actual_lora_params:,}")
        
        # 详细打印LoRA参数信息
        self._print_detailed_lora_info()
    
    def _print_detailed_lora_info(self):
        """打印详细的LoRA参数信息"""
        print("\n" + "="*60)
        print("🔍 DETAILED LORA PARAMETER ANALYSIS")
        print("="*60)
        
        # 分析speech encoder的LoRA参数
        print("\n📢 Speech Encoder LoRA Parameters:")
        speech_lora_params = 0
        speech_total_params = 0
        lora_modules_found = []
        
        for name, param in self.speech_encoder.model.named_parameters():
            speech_total_params += param.numel()
            if param.requires_grad:
                speech_lora_params += param.numel()
                # 检查是否是LoRA参数
                if 'lora' in name.lower():
                    lora_modules_found.append(name)
                    print(f"  ✅ {name}: {param.numel():,} params, shape={param.shape}, requires_grad={param.requires_grad}")
                else:
                    print(f"  ⚠️  Non-LoRA trainable param: {name}: {param.numel():,} params")
        
        print(f"\n📊 Speech Encoder Summary:")
        print(f"  - Total parameters: {speech_total_params:,}")
        print(f"  - Trainable (LoRA) parameters: {speech_lora_params:,}")
        print(f"  - LoRA modules found: {len(lora_modules_found)}")
        print(f"  - LoRA ratio: {speech_lora_params/speech_total_params*100:.4f}%")
        
        # 如果是分离模型，也分析text encoder
        if self.speech_encoder.model is not self.text_encoder.model:
            print("\n📝 Text Encoder LoRA Parameters:")
            text_lora_params = 0
            text_total_params = 0
            
            for name, param in self.text_encoder.model.named_parameters():
                text_total_params += param.numel()
                if param.requires_grad:
                    text_lora_params += param.numel()
                    if 'lora' in name.lower():
                        print(f"  ✅ {name}: {param.numel():,} params, shape={param.shape}")
                    else:
                        print(f"  ⚠️  Non-LoRA trainable param: {name}: {param.numel():,} params")
            
            print(f"\n📊 Text Encoder Summary:")
            print(f"  - Total parameters: {text_total_params:,}")
            print(f"  - Trainable (LoRA) parameters: {text_lora_params:,}")
            print(f"  - LoRA ratio: {text_lora_params/text_total_params*100:.4f}%")
        else:
            print("\n📝 Text Encoder: Sharing model with Speech Encoder")
        
        # 分析投影层参数
        print("\n🎯 Projection Layers:")
        proj_speech_params = sum(p.numel() for p in self.proj_speech.parameters())
        proj_text_params = sum(p.numel() for p in self.proj_text.parameters())
        total_proj_params = proj_speech_params + proj_text_params
        
        print(f"  - Speech projection: {proj_speech_params:,} params")
        print(f"  - Text projection: {proj_text_params:,} params")
        print(f"  - Total projection: {total_proj_params:,} params")
        
        # 总体统计
        total_trainable = self.actual_lora_params + total_proj_params
        print(f"\n🎯 OVERALL TRAINING SUMMARY:")
        print(f"  - LoRA parameters: {self.actual_lora_params:,}")
        print(f"  - Projection parameters: {total_proj_params:,}")
        print(f"  - Total trainable: {total_trainable:,}")
        
        # 验证LoRA是否正确应用
        if len(lora_modules_found) == 0:
            print("\n❌ WARNING: No LoRA modules found! LoRA may not be applied correctly!")
        else:
            print(f"\n✅ SUCCESS: Found {len(lora_modules_found)} LoRA modules")
        
        print("="*60 + "\n")
    
    def train(self, mode: bool = True):
        """确保训练/评估模式正确传播到底层模型"""
        super().train(mode)
        # 确保共享的 Qwen2 模型遵循此模式
        self.speech_encoder.model.train(mode)
        self.text_encoder.model.train(mode)
        if mode:
            print(f"[INFO] Set ContrastiveQwen2AudioModel to TRAINING mode")
        else:
            print(f"[INFO] Set ContrastiveQwen2AudioModel to EVAL mode")
        return self
    
    def eval(self):
        """设置为评估模式"""
        return self.train(False)
    
    def diagnose_lora_step_by_step(self):
        """
        逐步诊断LoRA为什么没有生效
        这是一个完整的诊断流程，会打印每一步的结果
        """
        print("\n" + "="*80)
        print("🔬 LORA TROUBLESHOOTING - STEP BY STEP DIAGNOSIS")
        print("="*80)
        
        # ============ 步骤 1: 检查 LoRA 是否被正确应用 ============
        print("\n【步骤 1/7】检查 LoRA 适配器是否正确应用")
        print("-" * 60)
        
        has_peft = hasattr(self.speech_encoder.model, 'peft_config')
        print(f"Has peft_config: {has_peft}")
        
        if has_peft:
            print(f"✅ PEFT config keys: {list(self.speech_encoder.model.peft_config.keys())}")
            for key, config in self.speech_encoder.model.peft_config.items():
                print(f"   - Adapter '{key}': {config}")
        else:
            print(f"❌ No peft_config found - LoRA was NOT applied!")
            print(f"   → 原因: get_peft_model() 调用失败或未执行")
            return
        
        has_active_adapters = hasattr(self.speech_encoder.model, 'active_adapters')
        if has_active_adapters:
            print(f"Active adapters: {self.speech_encoder.model.active_adapters}")
        
        # ============ 步骤 2: 检查 LoRA 参数是否存在 ============
        print("\n【步骤 2/7】检查 LoRA 参数是否被创建")
        print("-" * 60)
        
        lora_params = []
        for name, param in self.speech_encoder.model.named_parameters():
            if 'lora' in name.lower():
                lora_params.append((name, param))
        
        print(f"Found {len(lora_params)} LoRA parameters")
        if len(lora_params) == 0:
            print(f"❌ No LoRA parameters found!")
            print(f"   → 原因: LoRA 适配器未正确添加参数")
            return
        else:
            print(f"✅ LoRA parameters exist")
            print(f"   First 3 LoRA params:")
            for name, param in lora_params[:3]:
                print(f"   - {name}: shape={param.shape}, dtype={param.dtype}")
        
        # ============ 步骤 3: 检查 LoRA 参数的 requires_grad ============
        print("\n【步骤 3/7】检查 LoRA 参数的 requires_grad 标志")
        print("-" * 60)
        
        lora_trainable = sum(1 for name, param in lora_params if param.requires_grad)
        lora_frozen = sum(1 for name, param in lora_params if not param.requires_grad)
        
        print(f"Trainable LoRA params: {lora_trainable}/{len(lora_params)}")
        print(f"Frozen LoRA params: {lora_frozen}/{len(lora_params)}")
        
        if lora_trainable == 0:
            print(f"❌ All LoRA parameters are frozen (requires_grad=False)!")
            print(f"   → 原因: 参数被意外冻结")
            print(f"   → 解决: 调用 force_enable_lora_gradients()")
            # 自动修复
            self.force_enable_lora_gradients()
        else:
            print(f"✅ LoRA parameters have requires_grad=True")
        
        # ============ 步骤 4: 检查模型训练模式 ============
        print("\n【步骤 4/7】检查模型是否处于训练模式")
        print("-" * 60)
        
        print(f"ContrastiveQwen2AudioModel.training: {self.training}")
        print(f"speech_encoder.model.training: {self.speech_encoder.model.training}")
        print(f"text_encoder.model.training: {self.text_encoder.model.training}")
        
        if not self.training:
            print(f"❌ Model is in EVAL mode!")
            print(f"   → 原因: 模型被设置为评估模式")
            print(f"   → 解决: 调用 model.train()")
            return
        else:
            print(f"✅ Model is in TRAINING mode")
        
        # ============ 步骤 5: 测试前向传播是否触及 LoRA 层 ============
        print("\n【步骤 5/7】测试前向传播是否经过 LoRA 层")
        print("-" * 60)
        
        # 保存初始参数值
        initial_values = {}
        for name, param in lora_params[:5]:  # 只检查前5个
            initial_values[name] = param.data.clone()
        
        # 创建一个简单的测试输入
        try:
            print("Creating test audio input...")
            test_audio = [np.random.randn(16000).astype(np.float32)]  # 1秒音频
            
            print("Running forward pass...")
            with torch.set_grad_enabled(True):
                audio_emb = self.encode_audio(test_audio)
            
            print(f"✅ Forward pass successful")
            print(f"   Output shape: {audio_emb.shape}")
            print(f"   Output requires_grad: {audio_emb.requires_grad}")
            
            if not audio_emb.requires_grad:
                print(f"❌ Output does not require gradients!")
                print(f"   → 原因: 前向传播中梯度被断开")
                return
            
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            return
        
        # ============ 步骤 6: 测试反向传播 ============
        print("\n【步骤 6/7】测试反向传播是否更新 LoRA 参数")
        print("-" * 60)
        
        try:
            # 创建一个简单的损失
            loss = audio_emb.sum()
            print(f"Created dummy loss: {loss.item()}")
            
            # 清除之前的梯度
            self.zero_grad()
            
            # 反向传播
            print("Running backward pass...")
            loss.backward()
            
            print(f"✅ Backward pass successful")
            
            # 检查 LoRA 参数是否有梯度
            lora_with_grad = 0
            lora_without_grad = 0
            
            for name, param in lora_params:
                if param.grad is not None and param.grad.abs().sum() > 0:
                    lora_with_grad += 1
                else:
                    lora_without_grad += 1
            
            print(f"LoRA params with gradients: {lora_with_grad}/{len(lora_params)}")
            print(f"LoRA params without gradients: {lora_without_grad}/{len(lora_params)}")
            
            if lora_with_grad == 0:
                print(f"❌ No LoRA parameters received gradients!")
                print(f"   → 原因分析:")
                print(f"      1. 前向传播未经过 LoRA 层")
                print(f"      2. 梯度在某处被阻断（detach/no_grad）")
                print(f"      3. 使用了错误的编码路径（未使用 audio_tower）")
                
                # 详细检查哪些层有梯度
                print(f"\n   检查投影层的梯度:")
                for name, param in self.named_parameters():
                    if 'proj' in name and param.requires_grad:
                        has_grad = param.grad is not None and param.grad.abs().sum() > 0
                        status = "✅" if has_grad else "❌"
                        print(f"      {status} {name}: {'HAS GRAD' if has_grad else 'NO GRAD'}")
                
                return
            else:
                print(f"✅ LoRA parameters received gradients!")
                
                # 显示前几个有梯度的参数
                print(f"\n   Sample LoRA gradients:")
                shown = 0
                for name, param in lora_params:
                    if param.grad is not None and param.grad.abs().sum() > 0:
                        grad_norm = param.grad.norm().item()
                        print(f"      ✅ {name}: grad_norm={grad_norm:.6f}")
                        shown += 1
                        if shown >= 5:
                            break
            
        except Exception as e:
            print(f"❌ Backward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # ============ 步骤 7: 检查编码策略 ============
        print("\n【步骤 7/7】检查使用的编码策略")
        print("-" * 60)
        
        print(f"Encoding strategy: {self.speech_encoder.encoding_strategy}")
        print(f"Has audio_tower: {self.speech_encoder.has_audio_tower}")
        
        if self.speech_encoder.has_audio_tower:
            print(f"   Audio tower name: {self.speech_encoder.audio_tower_name}")
        
        if self.speech_encoder.encoding_strategy == 'audio_tower':
            print(f"✅ Using audio_tower (recommended)")
            print(f"   Audio tower output type: {self.speech_encoder.audio_tower_output_type}")
            print(f"   Audio hidden dim: {self.speech_encoder.audio_hidden_dim}")
            
            # 检查 audio_tower 中的 LoRA (使用动态名称)
            audio_module_name = self.speech_encoder.audio_tower_name
            audio_tower_lora = sum(1 for name, _ in lora_params if audio_module_name in name)
            print(f"   LoRA params in {audio_module_name}: {audio_tower_lora}")
            
            if audio_tower_lora == 0:
                print(f"   ⚠️  Warning: No LoRA in {audio_module_name}!")
                print(f"      LoRA 可能只应用在 language_model 上")
                print(f"      但使用 {audio_module_name} 编码时不会经过 language_model")
                print(f"   ❌ 这就是为什么 LoRA 没有梯度的原因！")
        else:
            print(f"⚠️  Using full_forward (fallback)")
            if not self.speech_encoder.has_audio_tower:
                print(f"   ❌ 原因: 模型没有找到 audio 模块")
                print(f"   → 检查的模块名: audio_tower, audio_encoder, audio_model, encoder")
                print(f"   → 实际模型类型: {type(self.speech_encoder.model).__name__}")
                print(f"   → 可能的问题: 模型版本不对或加载方式有误")
        
        # ============ 总结 ============
        print("\n" + "="*80)
        print("📊 DIAGNOSIS SUMMARY")
        print("="*80)
        print(f"✅ LoRA is properly configured and receiving gradients!")
        print(f"✅ Total LoRA params: {len(lora_params)}")
        print(f"✅ LoRA params with gradients: {lora_with_grad}")
        print(f"✅ Model is ready for training")
        print("="*80 + "\n")
    
    def force_enable_lora_gradients(self):
        """强制启用LoRA参数的梯度"""
        print("\n🔧 FORCING LORA GRADIENTS ENABLED")
        print("-" * 40)
        
        enabled_count = 0
        for name, param in self.speech_encoder.model.named_parameters():
            if 'lora' in name.lower() and not param.requires_grad:
                param.requires_grad = True
                enabled_count += 1
                print(f"  ✅ Enabled gradient for: {name}")
        
        if self.speech_encoder.model is not self.text_encoder.model:
            for name, param in self.text_encoder.model.named_parameters():
                if 'lora' in name.lower() and not param.requires_grad:
                    param.requires_grad = True
                    enabled_count += 1
                    print(f"  ✅ Enabled gradient for: {name}")
        
        print(f"\n📊 Force-enabled gradients for {enabled_count} LoRA parameters")
        print("-" * 40)
    
    def check_lora_gradients(self, step=None):
        """检查LoRA参数的梯度更新情况"""
        step_info = f" (Step {step})" if step is not None else ""
        print(f"\n🔍 LoRA Gradient Check{step_info}")
        print("-" * 60)
        
        lora_with_grad = 0
        lora_without_grad = 0
        non_lora_with_grad = 0
        lora_params_details = []
        
        # 首先检查模型是否处于训练模式
        print(f"[DEBUG] Model training mode: {self.speech_encoder.model.training}")
        
        # 检查PEFT状态
        if hasattr(self.speech_encoder.model, 'peft_config'):
            print(f"[DEBUG] PEFT config exists: {list(self.speech_encoder.model.peft_config.keys())}")
        else:
            print(f"[DEBUG] ❌ No PEFT config found!")
        
        if hasattr(self.speech_encoder.model, 'active_adapters'):
            print(f"[DEBUG] Active adapters: {self.speech_encoder.model.active_adapters}")
        
        # 统计所有参数
        total_params = 0
        trainable_params = 0
        
        for name, param in self.speech_encoder.model.named_parameters():
            total_params += 1
            if param.requires_grad:
                trainable_params += 1
                has_grad = param.grad is not None and param.grad.abs().sum() > 0
                
                if 'lora' in name.lower():
                    lora_params_details.append({
                        'name': name,
                        'shape': param.shape,
                        'has_grad': has_grad,
                        'grad_norm': param.grad.norm().item() if param.grad is not None else 0.0,
                        'param_norm': param.data.norm().item(),
                        'dtype': param.dtype,
                        'device': param.device
                    })
                    
                    if has_grad:
                        lora_with_grad += 1
                        grad_norm = param.grad.norm().item()
                        print(f"  ✅ LoRA {name}: grad_norm={grad_norm:.6f}")
                    else:
                        lora_without_grad += 1
                        grad_status = "None" if param.grad is None else "Zero"
                        print(f"  ❌ LoRA {name}: NO GRADIENT ({grad_status})")
                else:
                    if has_grad:
                        non_lora_with_grad += 1
                        grad_norm = param.grad.norm().item()
                        print(f"  ⚠️  Non-LoRA {name}: grad_norm={grad_norm:.6f}")
        
        print(f"\n📊 Parameter Summary{step_info}:")
        print(f"  - Total parameters: {total_params}")
        print(f"  - Trainable parameters: {trainable_params}")
        print(f"  - LoRA params with gradients: {lora_with_grad}")
        print(f"  - LoRA params without gradients: {lora_without_grad}")
        print(f"  - Non-LoRA params with gradients: {non_lora_with_grad}")
        
        # 详细分析LoRA参数
        if lora_params_details:
            print(f"\n🔍 Detailed LoRA Analysis:")
            for detail in lora_params_details[:10]:  # 只显示前10个
                print(f"  {detail['name']}: shape={detail['shape']}, "
                      f"grad_norm={detail['grad_norm']:.6f}, "
                      f"param_norm={detail['param_norm']:.6f}, "
                      f"dtype={detail['dtype']}")
        
        # 诊断建议
        if lora_with_grad == 0:
            print(f"\n❌ CRITICAL: NO LoRA parameters have gradients!")
            print(f"   Possible issues:")
            print(f"   1. LoRA not applied correctly")
            print(f"   2. Model not in training mode")
            print(f"   3. Forward pass not reaching LoRA layers")
            print(f"   4. Loss computation issues")
        else:
            print(f"\n✅ SUCCESS: {lora_with_grad} LoRA parameters are being updated")
        
        print("-" * 60)
    
    def print_parameter_stats_before_after(self, before_state=None):
        """比较训练前后的参数变化"""
        if before_state is None:
            # 保存当前状态
            state = {}
            for name, param in self.speech_encoder.model.named_parameters():
                if param.requires_grad and 'lora' in name.lower():
                    state[name] = param.data.clone().detach()
            return state
        else:
            # 比较变化
            print("\n🔄 LoRA Parameter Changes:")
            print("-" * 40)
            
            changes_found = 0
            for name, param in self.speech_encoder.model.named_parameters():
                if param.requires_grad and 'lora' in name.lower() and name in before_state:
                    old_param = before_state[name]
                    diff = (param.data - old_param).abs().sum().item()
                    if diff > 1e-8:
                        changes_found += 1
                        print(f"  ✅ {name}: changed by {diff:.8f}")
                    else:
                        print(f"  ❌ {name}: NO CHANGE")
            
            print(f"\n📊 Parameter Change Summary:")
            print(f"  - LoRA parameters changed: {changes_found}")
            
            if changes_found == 0:
                print("  ❌ WARNING: NO LoRA parameters changed during training!")
            else:
                print(f"  ✅ SUCCESS: {changes_found} LoRA parameters were updated")
            
            print("-" * 40)
    
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
    
    def get_optimizer_parameters(self):
        """获取优化器需要的参数列表（包括LoRA和投影层参数）"""
        # 收集LoRA参数
        lora_params = [p for n, p in self.speech_encoder.model.named_parameters() if p.requires_grad]
        
        # 如果不是共享模型，添加text encoder的LoRA参数
        if self.speech_encoder.model is not self.text_encoder.model:
            lora_params.extend([p for n, p in self.text_encoder.model.named_parameters() if p.requires_grad])
        
        # 添加投影层参数
        head_params = list(self.proj_speech.parameters()) + list(self.proj_text.parameters())
        
        return lora_params + head_params
    
    def encode_audio(self, audio_inputs: List[Union[str, torch.Tensor]], dynamic_padding: bool = True) -> torch.Tensor:
        """Encode audio files or tensors to embeddings"""
        # 在训练模式下保持梯度，评估模式下断链
        if self.training:
            speech_embeddings = self.speech_encoder.predict(audio_inputs, dynamic_padding=dynamic_padding)  # [B, hidden_dim]
        else:
            with torch.no_grad():
                speech_embeddings = self.speech_encoder.predict(audio_inputs, dynamic_padding=dynamic_padding)  # [B, hidden_dim]
        
        # 确保数据类型为float32并移动到正确设备
        if not isinstance(speech_embeddings, torch.Tensor):
            speech_embeddings = torch.from_numpy(speech_embeddings)
        speech_embeddings = speech_embeddings.float().to(self.proj_speech.weight.device)
        
        # 确保张量是2D的 [batch_size, hidden_dim]
        if speech_embeddings.dim() == 3:
            # 如果是3D张量，需要进行池化或者取平均
            if speech_embeddings.shape[1] == speech_embeddings.shape[0]:
                # 如果第二个维度等于batch_size，可能是错误的堆叠，取对角线
                print(f"[WARN] Detected 3D tensor with shape {speech_embeddings.shape}, extracting diagonal")
                speech_embeddings = torch.diagonal(speech_embeddings, dim1=0, dim2=1).T  # [batch_size, hidden_dim]
            else:
                # 否则对第二个维度进行平均池化
                print(f"[WARN] Detected 3D tensor with shape {speech_embeddings.shape}, applying mean pooling on dim=1")
                speech_embeddings = speech_embeddings.mean(dim=1)  # [batch_size, hidden_dim]
        elif speech_embeddings.dim() == 1:
            # 如果是1D张量，添加batch维度
            speech_embeddings = speech_embeddings.unsqueeze(0)
        
        # # 确保最终形状正确
        # if self.training and not getattr(self, "_logged_speech_shape", False):
        #     print(f"[DEBUG] Final speech_embeddings shape: {speech_embeddings.shape}")
        #     self._logged_speech_shape = True
        
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


def encode_texts_in_batches(model, texts, batch_size=1024, device="cuda"):
    """Encode texts in batches using the model's text encoder"""
    # 检查输入是否为空
    if not texts or len(texts) == 0:
        print("[WARN] Empty text list provided to encode_texts_in_batches, returning empty tensor")
        return torch.empty(0, 512, dtype=torch.float32, device=device)
    
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
    
    # 确保all_embeddings不为空
    if not all_embeddings:
        print("[WARN] No embeddings generated, returning empty tensor")
        return torch.empty(0, 512, dtype=torch.float32, device=device)
    
    return torch.cat(all_embeddings, dim=0)


def encode_audios_in_batches(model, audio_inputs, batch_size=64, device="cuda"):
    """
    Encode audios in batches using the model's audio encoder
    Optimized for both file paths and tensor inputs (mmap data)
    """
    all_embeddings = []
    
    for i in range(0, len(audio_inputs), batch_size):
        batch_inputs = audio_inputs[i:i + batch_size]
        try:
            embeddings = model.encode_audio(batch_inputs)
            # 确保数据类型为float32，但保持梯度（训练时）或断链（评估时）
            embeddings = embeddings.float()
            if not model.training:
                embeddings = embeddings.detach()
            all_embeddings.append(embeddings)
        except Exception as e:
            print(f"[ERROR] Failed to encode audio batch {i//batch_size}: {e}")
            print(f"[DEBUG] Batch input types: {[type(inp) for inp in batch_inputs[:3]]}")  # 只打印前3个的类型
            # Create dummy embeddings
            dummy_emb = torch.zeros(len(batch_inputs), 512, dtype=torch.float32, device=device)
            all_embeddings.append(dummy_emb)
    
    return torch.cat(all_embeddings, dim=0)


def encode_audio_tensors_in_batches_optimized(model, audio_tensors, batch_size=32, device="cuda"):
    """
    专门为mmap tensor数据优化的批量音频编码函数
    相比原版本，减少了不必要的类型检查和转换
    """
    if not audio_tensors:
        return torch.empty(0, 512, dtype=torch.float32, device=device)
    
    all_embeddings = []
    
    for i in range(0, len(audio_tensors), batch_size):
        batch_tensors = audio_tensors[i:i + batch_size]
        try:
            # 直接使用tensor输入，避免重复的类型检查
            if model.training:
                embeddings = model.encode_audio(batch_tensors)
            else:
                with torch.no_grad():
                    embeddings = model.encode_audio(batch_tensors)
            
            # 确保数据类型为float32
            embeddings = embeddings.float()
            if not model.training:
                embeddings = embeddings.detach()
            all_embeddings.append(embeddings)
            
        except Exception as e:
            print(f"[ERROR] Failed to encode audio tensor batch {i//batch_size}: {e}")
            print(f"[DEBUG] Batch tensor shapes: {[t.shape if isinstance(t, torch.Tensor) else 'Not tensor' for t in batch_tensors[:3]]}")
            # Create dummy embeddings with correct shape
            dummy_emb = torch.zeros(len(batch_tensors), 512, dtype=torch.float32, device=device)
            all_embeddings.append(dummy_emb)
    
    if not all_embeddings:
        return torch.empty(0, 512, dtype=torch.float32, device=device)
    
    return torch.cat(all_embeddings, dim=0)

class SimpleRetriever:
    """
    简单的检索器类，用于训练评估
    替代复杂的new_retrieve.Retriever类
    """
    def __init__(self, enable_fusion=True, device="cuda"):
        self.device = device
        self.enable_fusion = enable_fusion
        self.model = None  # 将由训练脚本设置
        self.index = None  # 将由训练脚本设置为FAISS索引
        self.term_list = []  # 将由训练脚本设置为术语列表
