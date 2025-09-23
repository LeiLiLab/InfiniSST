"""
完整的Modal部署脚本 - Qwen2-Audio Term-Level DDP训练
包含完整的训练代码，可以直接部署运行
"""

import modal
import json
import os

# 创建Modal App
app = modal.App("qwen2-audio-complete-training")

# 定义容器镜像
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git","wget","curl","ffmpeg","libsndfile1-dev","build-essential"])
    .pip_install([
        "torch==2.3.1", "torchvision==0.18.1", "torchaudio==2.3.1",
        "transformers==4.44.2", "accelerate==0.33.0", "datasets",
        "peft==0.11.1",
        # 在 Modal 上优先用 CPU 版 FAISS，GPU 版经常因为 CUDA 对不上失败
        "faiss-cpu==1.7.4",
        "soundfile==0.12.1", "librosa==0.10.1",
        "numpy==1.26.4", "scipy==1.11.4", "scikit-learn==1.3.2",
        "tqdm==4.66.1",
        # 用两个包替代 extras 语法
        "huggingface_hub",
        "hf-transfer",
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# 定义存储卷
volume = modal.Volume.from_name("qwen2-audio-training", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# 数据上传函数
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=3600,
)
def upload_data(data_dict: dict):
    """上传训练数据"""
    import json
    import os
    
    print(f"[INFO] Uploading {len(data_dict)} files to Modal volume...")
    
    for file_path, content in data_dict.items():
        full_path = f"/data/{file_path}"
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            if isinstance(content, (dict, list)):
                json.dump(content, f, indent=2, ensure_ascii=False)
            else:
                f.write(str(content))
        
        file_size = os.path.getsize(full_path)
        print(f"[INFO] Uploaded: {file_path} ({file_size / 1024 / 1024:.1f} MB)")
    
    volume.commit()
    print("[INFO] All data uploaded and committed.")

# 主训练函数
@app.function(
    image=image,
    gpu="A100-40GB:8",
    volumes={"/data": volume, "/root/.cache/huggingface": hf_cache_vol},
    timeout=86400,  # 24小时
    memory=256*1024,  # 256GB内存
    cpu=64,
    secrets=[modal.Secret.from_name("huggingface-token")]
)
def train_qwen2_audio_ddp():
    """执行DDP训练"""
    import subprocess
    import os
    import sys
    
    # 设置环境变量
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "12355", 
        "WORLD_SIZE": "8",
        "NCCL_DEBUG": "INFO",
        "NCCL_IB_DISABLE": "1",
        "NCCL_P2P_DISABLE": "0",
        "NCCL_SHM_DISABLE": "0",
        "NCCL_SOCKET_IFNAME": "lo",
        "NCCL_TIMEOUT": "1800",
        "OMP_NUM_THREADS": "4",
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1"
    })
    
    print("[INFO] Environment setup completed")
    print(f"[INFO] Available GPUs: {env.get('CUDA_VISIBLE_DEVICES')}")
    
    # 检查GPU
    import torch
    print(f"[INFO] PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"[INFO] GPU count: {torch.cuda.device_count()}")
    
    # 检查数据文件
    train_file = "/data/xl_cleaned_term_level_chunks_merged.json"
    # test_file = "/data/samples/xl/term_level_chunks_500000_1000000.json"
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    # if not os.path.exists(test_file):
    #     raise FileNotFoundError(f"Test file not found: {test_file}")
    
    print(f"[INFO] Training data: {train_file}")
    # print(f"[INFO] Test data: {test_file}")
    
    # 将训练代码写入文件
    training_code = '''
# === GPU设备设置 - 必须在任何CUDA操作之前进行 ===
import os
import sys

# DDP相关导入
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 设置CUDA环境
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# 导入torch
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
import faiss
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from typing import List, Union
import warnings
import librosa

warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 简化的模型定义
class Qwen2AudioSpeechEncoder:
    def __init__(self, model_name="Qwen/Qwen2-Audio-7B-Instruct", device="cuda"):
        self.device = device
        print(f"[INFO] Loading Qwen2-Audio model: {model_name}")
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=None
        )
        self.model.to(self.device)
        self.model.eval()
    
    def get_shared_model(self):
        return {'model': self.model, 'processor': self.processor}
    
    def predict(self, audio_paths: List[str], max_length: int = None) -> torch.Tensor:
        embeddings = []
        
        for audio_path in audio_paths:
            try:
                audio, sr = librosa.load(audio_path, sr=16000)
                
                if len(audio) == 0:
                    dummy_embedding = torch.zeros(4096, device=self.model.device)
                    embeddings.append(dummy_embedding)
                    continue
                
                max_samples = max_length if max_length else 16000 * 30
                if len(audio) > max_samples:
                    audio = audio[:max_samples]
                
                min_samples = 1600
                if len(audio) < min_samples:
                    audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')
                
                audio = np.array(audio, dtype=np.float32)
                
                inputs = self.processor(
                    text="<|audio_bos|><|AUDIO|><|audio_eos|>",
                    audios=audio,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(self.model.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_hidden_states=True,
                    return_dict_in_generate=True
                )
                
                if hasattr(outputs, 'encoder_hidden_states') and outputs.encoder_hidden_states:
                    last_hidden_state = outputs.encoder_hidden_states[-1]
                else:
                    last_hidden_state = outputs.hidden_states[0][-1]
                
                pooled_features = last_hidden_state.mean(dim=1)
                embedding = pooled_features.squeeze()
                embeddings.append(embedding)
                
            except Exception as e:
                print(f"[ERROR] Failed to process audio {audio_path}: {e}")
                dummy_embedding = torch.zeros(4096, device=self.model.device)
                embeddings.append(dummy_embedding)
        
        return torch.stack(embeddings)

class Qwen2AudioTextEncoder:
    def __init__(self, model_name="Qwen/Qwen2-Audio-7B-Instruct", device="cuda", shared_model=None):
        self.device = device
        
        if shared_model is not None:
            self.processor = shared_model['processor']
            self.model = shared_model['model']
        else:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=None
            )
            self.model.to(self.device)
            self.model.eval()
    
    def predict(self, texts: List[str], source_lang: str = "eng_Latn") -> torch.Tensor:
        embeddings = []
        
        for text in texts:
            try:
                inputs = self.processor.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                for key in inputs:
                    inputs[key] = inputs[key].to(self.model.device)
                
                outputs = self.model.language_model(**inputs, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1]
                
                attention_mask = inputs["attention_mask"]
                masked_embeddings = last_hidden_state * attention_mask.unsqueeze(-1)
                pooled_embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                
                embedding = pooled_embedding.squeeze()
                embeddings.append(embedding)
                
            except Exception as e:
                print(f"[ERROR] Failed to process text: {e}")
                dummy_embedding = torch.zeros(4096, device=self.model.device)
                embeddings.append(dummy_embedding)
        
        return torch.stack(embeddings)

class ContrastiveQwen2AudioModel(nn.Module):
    def __init__(self, speech_encoder, text_encoder, hidden_dim=4096, proj_dim=512, 
                 lora_r=16, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        self.speech_encoder = speech_encoder
        self.text_encoder = text_encoder
        
        self.proj_speech = nn.Linear(hidden_dim, proj_dim)
        self.proj_text = nn.Linear(hidden_dim, proj_dim)
        
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        
        # 冻结原始参数
        for param in self.speech_encoder.model.parameters():
            param.requires_grad = False
        for param in self.text_encoder.model.parameters():
            param.requires_grad = False
        
        # 应用LoRA
        if self.speech_encoder.model is self.text_encoder.model:
            self.speech_encoder.model = get_peft_model(self.speech_encoder.model, self.lora_config)
            self.text_encoder.model = self.speech_encoder.model
        else:
            self.speech_encoder.model = get_peft_model(self.speech_encoder.model, self.lora_config)
            self.text_encoder.model = get_peft_model(self.text_encoder.model, self.lora_config)
    
    def get_trainable_parameters(self):
        lora_params = sum(p.numel() for p in self.speech_encoder.model.parameters() if p.requires_grad)
        if self.speech_encoder.model is not self.text_encoder.model:
            lora_params += sum(p.numel() for p in self.text_encoder.model.parameters() if p.requires_grad)
        
        proj_params = sum(p.numel() for p in self.proj_speech.parameters()) + sum(p.numel() for p in self.proj_text.parameters())
        
        return {
            'lora_params': lora_params,
            'proj_params': proj_params,
            'total_trainable': lora_params + proj_params
        }
    
    def encode_audio(self, audio_paths: List[str]) -> torch.Tensor:
        if self.training:
            speech_embeddings = self.speech_encoder.predict(audio_paths)
        else:
            with torch.no_grad():
                speech_embeddings = self.speech_encoder.predict(audio_paths)
        
        if not isinstance(speech_embeddings, torch.Tensor):
            speech_embeddings = torch.from_numpy(speech_embeddings)
        speech_embeddings = speech_embeddings.float().to(self.proj_speech.weight.device)
        
        return F.normalize(self.proj_speech(speech_embeddings), dim=-1)
    
    def encode_text(self, texts: List[str], source_lang: str = "eng_Latn") -> torch.Tensor:
        if self.training:
            text_embeddings = self.text_encoder.predict(texts, source_lang=source_lang)
        else:
            with torch.no_grad():
                text_embeddings = self.text_encoder.predict(texts, source_lang=source_lang)
        
        if not isinstance(text_embeddings, torch.Tensor):
            text_embeddings = torch.from_numpy(text_embeddings)
        text_embeddings = text_embeddings.float().to(self.proj_text.weight.device)
        
        return F.normalize(self.proj_text(text_embeddings), dim=-1)

# 简化的数据集
class TermLevelDataset(Dataset):
    def __init__(self, path, split="train", train_ratio=0.998, test_path=None):
        if split == "test" and test_path:
            with open(test_path, "r") as f:
                all_samples = json.load(f)
            use_split_logic = False
        else:
            with open(path, "r") as f:
                all_samples = json.load(f)
            use_split_logic = True
        
        # 简化的数据过滤
        valid_samples = []
        for s in all_samples:
            terms = s.get('term_chunk_audio_ground_truth_terms', [])
            if s.get('term_chunk_text', '').strip() and s.get('term_chunk_audio', ''):
                valid_samples.append(s)
        
        if use_split_logic:
            import random
            random.seed(42)
            random.shuffle(valid_samples)
            
            split_idx = int(len(valid_samples) * train_ratio)
            
            if split == "train":
                self.samples = valid_samples[:split_idx]
            else:
                self.samples = valid_samples[split_idx:]
        else:
            self.samples = valid_samples
        
        print(f"[INFO] Loaded {len(self.samples)} samples for {split}")
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample["term_chunk_audio"]
        chunk_text = sample["term_chunk_text"]
        ground_truth_terms = sample.get('term_chunk_audio_ground_truth_terms', [])
        has_target = bool(ground_truth_terms)
        
        return ground_truth_terms, audio_path, chunk_text, has_target
    
    def __len__(self):
        return len(self.samples)

# 音频文件验证函数
def is_audio_valid(audio_path, min_duration=0.01, max_duration=30.0):
    try:
        if not os.path.exists(audio_path):
            return False, "File does not exist"
        import soundfile as sf
        data, sr = sf.read(audio_path)
        if len(data) == 0:
            return False, "Empty audio file"
        duration = len(data) / sr
        if duration < min_duration or duration > max_duration:
            return False, f"Duration {duration:.3f}s out of range"
        return True, "Valid"
    except Exception as e:
        return False, f"Failed to read: {str(e)}"

def validate_audio_batch(audio_paths, verbose=False):
    valid_paths = []
    valid_indices = []
    invalid_count = 0
    
    for i, path in enumerate(audio_paths):
        is_valid, reason = is_audio_valid(path)
        if is_valid:
            valid_paths.append(path)
            valid_indices.append(i)
        else:
            invalid_count += 1
            if verbose or invalid_count <= 5:
                print(f"[WARN] Invalid audio {i}: {path} - {reason}")
    
    return valid_paths, valid_indices

# 训练步骤
def train_step(model, batch, device, args, temperature=0.07):
    raw_model = model.module if isinstance(model, DDP) else model
    
    if len(batch) < 2:
        return torch.tensor(0.0, requires_grad=True).to(device)
    
    ground_truth_terms_list, audio_paths, chunk_texts, has_targets = zip(*batch)
    ground_truth_terms_list = [[t.lower() for t in terms if isinstance(t, str)] for terms in ground_truth_terms_list]
    chunk_texts = [text.lower() if isinstance(text, str) else "" for text in chunk_texts]
    
    # 验证音频文件
    valid_audio_paths, valid_audio_indices = validate_audio_batch(audio_paths, verbose=False)
    
    if len(valid_audio_paths) == 0:
        return torch.tensor(0.0, requires_grad=True).to(device)
    
    if len(valid_audio_paths) != len(audio_paths):
        # 重新组织batch，只保留有效的样本
        valid_batch_data = []
        for idx in valid_audio_indices:
            valid_batch_data.append((
                ground_truth_terms_list[idx],
                audio_paths[idx], 
                chunk_texts[idx],
                has_targets[idx]
            ))
        
        if len(valid_batch_data) < 2:
            return torch.tensor(0.0, requires_grad=True).to(device)
        
        ground_truth_terms_list, audio_paths, chunk_texts, has_targets = zip(*valid_batch_data)
        ground_truth_terms_list = list(ground_truth_terms_list)
        audio_paths = list(audio_paths)
        chunk_texts = list(chunk_texts)
        has_targets = list(has_targets)
    
    try:
        audio_emb = raw_model.encode_audio(audio_paths)
        text_emb = raw_model.encode_text(chunk_texts) if args.audio_text_loss_ratio > 0 else torch.zeros_like(audio_emb)
        
        # 检查embedding有效性
        if torch.isnan(audio_emb).any() or torch.isinf(audio_emb).any():
            return torch.tensor(0.0, requires_grad=True).to(device)
        if args.audio_text_loss_ratio > 0 and (torch.isnan(text_emb).any() or torch.isinf(text_emb).any()):
            return torch.tensor(0.0, requires_grad=True).to(device)
            
    except Exception as e:
        print(f"[ERROR] Encoding failed: {e}")
        return torch.tensor(0.0, requires_grad=True).to(device)
    
    batch_size = len(audio_paths)
    
    # 音频-文本对比损失
    if args.audio_text_loss_ratio > 0:
        sim_matrix = (audio_emb @ text_emb.T) / temperature
        if torch.isnan(sim_matrix).any() or torch.isinf(sim_matrix).any():
            return torch.tensor(0.0, requires_grad=True).to(device)
        
        labels = torch.arange(batch_size).to(device)
        
        try:
            loss_audio_to_text = F.cross_entropy(sim_matrix, labels)
            loss_text_to_audio = F.cross_entropy(sim_matrix.T, labels)
            
            if torch.isnan(loss_audio_to_text) or torch.isnan(loss_text_to_audio):
                return torch.tensor(0.0, requires_grad=True).to(device)
            
            contrastive_loss = (loss_audio_to_text + loss_text_to_audio) / 2
        except Exception as e:
            return torch.tensor(0.0, requires_grad=True).to(device)
    else:
        contrastive_loss = torch.tensor(0.0, device=device)
    
    # 音频-术语对比损失
    all_gt_terms = []
    audio_term_pairs = []
    
    for i, terms in enumerate(ground_truth_terms_list):
        for term in terms:
            if term and len(term.strip()) > 0:
                term_idx = len(all_gt_terms)
                all_gt_terms.append(term.strip())
                audio_term_pairs.append((i, term_idx))
    
    if len(all_gt_terms) > 0:
        terms_emb = raw_model.encode_text(all_gt_terms)
        audio_term_sim = (audio_emb @ terms_emb.T) / temperature
        
        if torch.isnan(audio_term_sim).any() or torch.isinf(audio_term_sim).any():
            return torch.tensor(0.0, requires_grad=True).to(device)
        
        audio_term_labels = []
        for i in range(batch_size):
            positive_terms = [term_idx for audio_idx, term_idx in audio_term_pairs if audio_idx == i]
            if positive_terms:
                import random
                audio_term_labels.append(random.choice(positive_terms))
            else:
                audio_term_labels.append(-1)
        
        valid_indices = [i for i, label in enumerate(audio_term_labels) if label >= 0]
        
        if len(valid_indices) > 0:
            valid_audio_term_sim = audio_term_sim[valid_indices]
            valid_labels = torch.tensor([audio_term_labels[i] for i in valid_indices], device=device)
            audio_term_loss = F.cross_entropy(valid_audio_term_sim, valid_labels)
        else:
            audio_term_loss = torch.tensor(0.0, device=device)
    else:
        audio_term_loss = torch.tensor(0.0, device=device)
    
    total_loss = args.audio_text_loss_ratio * contrastive_loss + args.audio_term_loss_ratio * audio_term_loss
    
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        return torch.tensor(0.0, requires_grad=True).to(device)
    
    return total_loss

# 简化的评估函数
def evaluate_model(model, test_dataset, device, max_eval=500):
    model.eval()
    raw_model = model.module if isinstance(model, DDP) else model
    
    # 提取所有术语用于构建索引
    all_terms = set()
    for sample in test_dataset.samples[:max_eval]:
        terms = sample.get('term_chunk_audio_ground_truth_terms', [])
        for term in terms:
            if isinstance(term, str) and len(term.strip()) > 0:
                all_terms.add(term.lower())
    
    if len(all_terms) == 0:
        return 0.0
    
    all_terms = list(all_terms)
    
    # 构建术语索引
    with torch.no_grad():
        try:
            text_embs = []
            for i in range(0, len(all_terms), 32):  # 分批编码
                batch_terms = all_terms[i:i+32]
                batch_emb = raw_model.encode_text(batch_terms)
                text_embs.append(batch_emb.cpu().numpy())
            
            text_embs = np.concatenate(text_embs, axis=0)
            
            # 创建FAISS索引
            import faiss
            index = faiss.IndexFlatL2(text_embs.shape[1])
            index.add(text_embs.astype(np.float32))
            
        except Exception as e:
            print(f"[EVAL ERROR] Failed to build index: {e}")
            return 0.0
    
    # 评估样本
    correct = 0
    total = 0
    
    eval_samples = test_dataset.samples[:max_eval]
    
    for sample in eval_samples:
        ground_truth_terms = sample.get('term_chunk_audio_ground_truth_terms', [])
        if not ground_truth_terms:
            continue
            
        audio_path = sample.get('term_chunk_audio', '')
        if not audio_path:
            continue
            
        gt_terms_lower = [t.lower() for t in ground_truth_terms if isinstance(t, str)]
        if not gt_terms_lower:
            continue
        
        try:
            # 编码音频
            with torch.no_grad():
                audio_emb = raw_model.encode_audio([audio_path])
                audio_emb_np = audio_emb.cpu().numpy().astype(np.float32)
            
            # 检索top-10
            D, I = index.search(audio_emb_np, 10)
            retrieved_terms = [all_terms[idx] for idx in I[0]]
            
            # 计算召回率
            matched = sum(1 for gt_term in gt_terms_lower if gt_term in retrieved_terms)
            correct += matched
            total += len(gt_terms_lower)
            
        except Exception as e:
            print(f"[EVAL ERROR] Failed to evaluate sample: {e}")
            continue
    
    model.train()
    recall = correct / total if total > 0 else 0.0
    return recall

# DDP设置
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

# 训练函数
def train_ddp(rank, world_size, args):
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        print(f"[INFO] Starting DDP training with {world_size} GPUs")
    
    # 模型初始化
    speech_encoder = Qwen2AudioSpeechEncoder(model_name=args.model_name, device=device)
    text_encoder = Qwen2AudioTextEncoder(model_name=args.model_name, device=device, shared_model=speech_encoder.get_shared_model())
    
    model = ContrastiveQwen2AudioModel(
        speech_encoder, text_encoder,
        hidden_dim=4096, proj_dim=512,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout
    ).to(device)
    
    model = DDP(model, device_ids=[rank])
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 数据集
    if args.test_samples_path:
        train_dataset = TermLevelDataset(args.train_samples_path, split="train", train_ratio=1.0)
        test_dataset = TermLevelDataset(None, split="test", test_path=args.test_samples_path)
    else:
        train_dataset = TermLevelDataset(args.train_samples_path, split="train")
        test_dataset = TermLevelDataset(args.train_samples_path, split="test")
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    # 定义可序列化的collate函数
    def identity_collate_fn(batch):
        return batch
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size // world_size,
        sampler=train_sampler,
        collate_fn=identity_collate_fn,
        num_workers=8
    )
    
    # 早停和学习率调度器
    best_recall = 0.0
    no_improve_epochs = 0
    patience = getattr(args, 'patience', 3)
    
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=(rank == 0))
    
    # 训练循环
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        
        # 训练一个epoch
        if rank == 0:
            pbar = tqdm(train_dataloader, desc=f"[Epoch {epoch+1}/{args.epochs}]")
        else:
            pbar = train_dataloader
            
        for batch in pbar:
            loss = train_step(model, batch, device, args)
            
            if loss.requires_grad and not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            elif torch.isnan(loss) or torch.isinf(loss):
                if rank == 0:
                    print(f"[WARNING] Skipping batch due to NaN/Inf loss")
                optimizer.zero_grad()
        
        # 同步所有进程的损失
        avg_loss = total_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        
        # 收集所有进程的损失进行平均
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / world_size
        
        if rank == 0:
            print(f"[INFO] Epoch {epoch+1}/{args.epochs}, Avg Loss: {avg_loss:.4f}")
        
        # 评估（只在主进程执行）
        if rank == 0:
            print(f"[INFO] Evaluating epoch {epoch+1}...")
            current_recall = evaluate_model(model, test_dataset, device, max_eval=500)
            print(f"[EVAL] Epoch {epoch+1} Recall@10: {current_recall:.2%}")
            
            # 更新学习率调度器
            scheduler.step(current_recall)
            
            # 打印当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[INFO] Current LR: {current_lr:.2e}")
            
            # 早停检查
            if current_recall > best_recall:
                best_recall = current_recall
                no_improve_epochs = 0
                
                # 保存最佳模型
                best_model_path = "/data/model_best.pt"
                torch.save(model.state_dict(), best_model_path)
                print(f"[INFO] New best model saved! Recall@10: {best_recall:.2%}")
                
            else:
                no_improve_epochs += 1
                print(f"[INFO] No improvement for {no_improve_epochs} epochs (best: {best_recall:.2%})")
                
                if no_improve_epochs >= patience:
                    print(f"[EARLY STOPPING] No improvement in {patience} epochs. Best Recall@10: {best_recall:.2%}")
                    break
            
            # 定期保存检查点
            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), f"/data/model_epoch_{epoch+1}.pt")
                print(f"[INFO] Checkpoint saved: epoch_{epoch+1}.pt")
        
        # 同步所有进程，确保主进程完成评估后再继续
        dist.barrier()
        
        # 如果主进程决定早停，通知其他进程
        should_stop = torch.tensor(no_improve_epochs >= patience, device=device, dtype=torch.bool)
        dist.broadcast(should_stop, src=0)
        
        if should_stop.item():
            if rank != 0:
                print(f"[INFO] Rank {rank}: Received early stopping signal")
            break
    
    # 保存最终模型
    if rank == 0:
        torch.save(model.state_dict(), "/data/model_final.pt")
        print(f"[INFO] Training completed! Best Recall@10: {best_recall:.2%}")
        print(f"[INFO] Final model saved to: /data/model_final.pt")
        print(f"[INFO] Best model saved to: /data/model_best.pt")
    
    cleanup_ddp()

# 主函数
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_samples_path', required=True)
    parser.add_argument('--test_samples_path', default=None)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_name', default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--audio_text_loss_ratio', type=float, default=0.3)
    parser.add_argument('--audio_term_loss_ratio', type=float, default=0.7)
    parser.add_argument('--patience', type=int, default=3,
                       help="Early stopping patience (default: 3)")
    
    args = parser.parse_args()
    
    world_size = int(os.environ.get('WORLD_SIZE', 8))
    
    mp.spawn(train_ddp, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
'''
    
    # 写入训练脚本
    script_path = "/tmp/qwen2_ddp_train.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(training_code)
    
    # 构建训练命令
    cmd = [
        sys.executable, script_path,
        "--train_samples_path", train_file,
        "--epochs", "40",
        "--batch_size", "128",
        "--lr", "1e-4",
        "--model_name", "Qwen/Qwen2-Audio-7B-Instruct",
        "--lora_r", "16",
        "--lora_alpha", "32",
        "--lora_dropout", "0.1",
        "--audio_text_loss_ratio", "0.3",
        "--audio_term_loss_ratio", "0.7",
        "--patience", "3"
    ]
    
    print(f"[INFO] Executing: {' '.join(cmd)}")
    
    # 执行训练
    process = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    
    # 实时输出日志
    for line in iter(process.stdout.readline, ''):
        print(line.rstrip())
    
    process.wait()
    
    if process.returncode != 0:
        raise RuntimeError(f"Training failed with return code {process.returncode}")
    
    # 提交模型到卷
    volume.commit()
    print("[INFO] Training completed and model saved!")
    
    return "DDP training completed successfully"

# 检查数据是否已存在的函数
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=300,
)
def check_data_exists():
    """检查数据是否已存在于volume中"""
    import os
    train_file = "/data/xl_cleaned_term_level_chunks_merged.json"
    return os.path.exists(train_file)

# 本地入口点
@app.local_entrypoint()
def main():
    """本地入口点 - 检查数据并启动训练"""
    import json
    
    # 检查数据是否已存在
    print("[INFO] Checking if data already exists in Modal volume...")
    data_exists = check_data_exists.remote()
    
    if not data_exists:
        print("[INFO] Data not found in volume, uploading...")
        
        # 定义要上传的数据文件
        data_files = {}
        
        # 训练数据
        train_path = "data/xl_cleaned_term_level_chunks_merged.json"
        if os.path.exists(train_path):
            print(f"[INFO] Loading training data from {train_path}")
            with open(train_path, 'r', encoding='utf-8') as f:
                data_files["xl_cleaned_term_level_chunks_merged.json"] = json.load(f)
            print(f"[INFO] Training data loaded: {len(data_files['xl_cleaned_term_level_chunks_merged.json'])} samples")
        else:
            print(f"[ERROR] Training data not found: {train_path}")
            return
        
        if not data_files:
            print("[ERROR] No data files found to upload!")
            return
        
        print(f"[INFO] Total data to upload: {len(data_files)} files")
        
        # 上传数据
        print("[INFO] Uploading data to Modal...")
        upload_data.remote(data_files)
        print("[INFO] Data upload completed!")
    else:
        print("[INFO] Data already exists in volume, skipping upload.")
    
    # 启动训练
    print("[INFO] Starting Qwen2-Audio DDP training on Modal...")
    result = train_qwen2_audio_ddp.remote()
    
    print(f"[INFO] Training result: {result}")
    print("[INFO] Training completed! Check the Modal dashboard for logs and results.")


if __name__ == "__main__":
    main()
