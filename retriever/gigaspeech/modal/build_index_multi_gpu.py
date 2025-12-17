#!/usr/bin/env python3
"""
多GPU生成文本索引
专门用于预先生成400万terms的FAISS索引

使用说明：
1. 如果遇到CUDA OOM错误，减小 --batch_size (默认为4)
2. 建议设置环境变量: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
3. 对于大型模型(如Qwen2-Audio-7B)，每个GPU需要约48GB显存
4. batch_size建议值：48GB显存用4-8，80GB显存用16-32

示例运行命令：
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python build_index_multi_gpu.py \
    --model_path /path/to/model.pt \
    --glossary_path /path/to/glossary.json \
    --output_path /path/to/output.pkl \
    --num_gpus 6 \
    --batch_size 4
"""

import os
import sys
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import faiss
import pickle

# 禁用 tokenizers 的并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from Qwen2_Audio_train import (
    Qwen2AudioSpeechEncoder,
    Qwen2AudioTextEncoder,
    ContrastiveQwen2AudioModel,
)
from peft import LoraConfig, get_peft_model, TaskType


def encode_texts_parallel(model_config, texts, batch_size=256, num_gpus=4):
    """
    使用多GPU真正并行编码文本
    每个GPU独立加载模型（包括LoRA），处理不同的数据chunk
    
    Args:
        model_config: 包含模型配置和权重的字典
        texts: 要编码的文本列表
        batch_size: 每个GPU的batch size
        num_gpus: 使用的GPU数量
    """
    print(f"[INFO] Encoding {len(texts)} texts using {num_gpus} GPUs (parallel with LoRA)...")
    
    # 将texts分配到多个GPU
    chunk_size = (len(texts) + num_gpus - 1) // num_gpus
    text_chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
    
    print(f"[INFO] Split into {len(text_chunks)} chunks, ~{chunk_size} texts each")
    
    import threading
    import queue
    
    results_queue = queue.Queue()
    model_load_lock = threading.Lock()  # 加载模型时加锁，避免并发冲突
    
    def process_on_gpu(gpu_id, text_chunk, chunk_idx):
        """在指定GPU上处理文本chunk"""
        try:
            device = torch.device(f"cuda:{gpu_id}")
            
            print(f"[GPU {gpu_id}] Waiting to load model...")
            
            # 加载模型到当前GPU（加锁避免并发冲突）
            with model_load_lock:
                print(f"[GPU {gpu_id}] Loading model with LoRA (locked)...")
                from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
                
                # 加载processor
                processor = AutoProcessor.from_pretrained(model_config['model_name'])
                
                # 直接加载到指定GPU，使用float16减少显存占用
                gpu_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    model_config['model_name'],
                    torch_dtype=torch.float16,
                    device_map={"": device},
                    low_cpu_mem_usage=True,
                )
                print(f"[GPU {gpu_id}] Model loaded directly to GPU {gpu_id}...")
                
                # 应用LoRA配置（与训练时一致）
                print(f"[GPU {gpu_id}] Applying LoRA configuration...")
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=model_config['lora_r'],
                    lora_alpha=model_config['lora_alpha'],
                    lora_dropout=0.0,  # 评估时禁用dropout
                    target_modules=model_config['target_modules'],
                    bias="none",
                )
                gpu_model = get_peft_model(gpu_model, lora_config)
                print(f"[GPU {gpu_id}] LoRA applied")
            
            # 创建text encoder
            text_encoder = Qwen2AudioTextEncoder.__new__(Qwen2AudioTextEncoder)
            text_encoder.device = device
            text_encoder.processor = processor
            text_encoder.model = gpu_model
            text_encoder._analyze_model_structure()
            
            # 创建wrapper
            import torch.nn as nn
            class TextOnlyModel(nn.Module):
                def __init__(self, text_encoder, proj_text):
                    super().__init__()
                    self.text_encoder = text_encoder
                    self.proj_text = proj_text
                
                def encode_text(self, texts):
                    with torch.no_grad():
                        emb = self.text_encoder.predict(texts)
                    if not isinstance(emb, torch.Tensor):
                        emb = torch.as_tensor(emb)
                    emb = emb.float().to(self.proj_text.weight.device)
                    return F.normalize(self.proj_text(emb), dim=-1)
            
            # 复制投影层到GPU
            proj_text = torch.nn.Linear(
                model_config['text_hidden'],
                512
            ).to(device)
            proj_text.load_state_dict(model_config['proj_text_state_dict'])
            
            gpu_wrapper = TextOnlyModel(text_encoder, proj_text)
            gpu_wrapper.eval()
            
            # 加载LoRA权重
            if model_config['lora_state_dict']:
                print(f"[GPU {gpu_id}] Loading LoRA weights...")
                missing, unexpected = gpu_model.load_state_dict(model_config['lora_state_dict'], strict=False)
                print(f"[GPU {gpu_id}] LoRA weights loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")
            
            # 清理缓存，确保有足够空间用于推理
            torch.cuda.empty_cache()
            
            print(f"[GPU {gpu_id}] Model fully loaded, processing {len(text_chunk)} texts...")
            
            # 打印GPU显存使用情况
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            print(f"[GPU {gpu_id}] Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            # 处理当前chunk
            chunk_embeddings = []
            failed_batches = 0
            
            for i in tqdm(range(0, len(text_chunk), batch_size), desc=f"GPU {gpu_id}", position=gpu_id):
                batch_texts = text_chunk[i:i+batch_size]
                try:
                    with torch.no_grad():
                        embeddings = gpu_wrapper.encode_text(batch_texts)
                        # 立即转为numpy并释放GPU tensor
                        embeddings_np = embeddings.cpu().float().numpy()
                        del embeddings
                    
                    chunk_embeddings.append(embeddings_np)
                    
                    # 每个batch后都清理缓存，更积极地管理内存
                    torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"[GPU {gpu_id} ERROR] Batch {i//batch_size}: {e}")
                    failed_batches += 1
                    dummy_emb = np.zeros((len(batch_texts), 512), dtype=np.float32)
                    chunk_embeddings.append(dummy_emb)
                    torch.cuda.empty_cache()
            
            if failed_batches > 0:
                print(f"[GPU {gpu_id} WARN] {failed_batches} batches failed and were replaced with zeros")
            
            chunk_embeddings = np.concatenate(chunk_embeddings, axis=0)
            print(f"[GPU {gpu_id}] ✅ Completed, encoded {len(chunk_embeddings)} texts")
            
            # 清理
            del gpu_model
            del text_encoder
            del gpu_wrapper
            del proj_text
            torch.cuda.empty_cache()
            
            # 将结果放入队列
            results_queue.put((chunk_idx, chunk_embeddings))
            
        except Exception as e:
            print(f"[GPU {gpu_id} ERROR] Fatal error: {e}")
            import traceback
            traceback.print_exc()
            results_queue.put((chunk_idx, None))
    
    # 串行加载模型，然后并行处理数据
    # 这样避免多线程并发加载时的CUDA冲突
    print("[INFO] Loading models sequentially to avoid CUDA conflicts...")
    
    gpu_models = []
    for gpu_id in range(len(text_chunks)):
        print(f"[INFO] Pre-loading model on GPU {gpu_id}...")
        device = torch.device(f"cuda:{gpu_id}")
        
        try:
            from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
            
            # 加载processor
            processor = AutoProcessor.from_pretrained(model_config['model_name'])
            
            # 直接加载到指定GPU
            gpu_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_config['model_name'],
                torch_dtype=torch.float16,
                device_map={"": device},
                low_cpu_mem_usage=True,
            )
            print(f"[GPU {gpu_id}] Model loaded")
            
            # 应用LoRA配置
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=model_config['lora_r'],
                lora_alpha=model_config['lora_alpha'],
                lora_dropout=0.0,
                target_modules=model_config['target_modules'],
                bias="none",
            )
            gpu_model = get_peft_model(gpu_model, lora_config)
            print(f"[GPU {gpu_id}] LoRA applied")
            
            # 加载LoRA权重
            if model_config['lora_state_dict']:
                missing, unexpected = gpu_model.load_state_dict(model_config['lora_state_dict'], strict=False)
                print(f"[GPU {gpu_id}] LoRA weights loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")
            
            # 清理缓存
            torch.cuda.empty_cache()
            
            gpu_models.append((gpu_id, device, processor, gpu_model))
            
        except Exception as e:
            print(f"[GPU {gpu_id} ERROR] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            gpu_models.append((gpu_id, None, None, None))
    
    print("[INFO] All models loaded, starting parallel processing...")
    
    # 修改process_on_gpu函数为使用预加载的模型
    def process_on_gpu_preloaded(gpu_id, device, processor, gpu_model, text_chunk, chunk_idx):
        """使用预加载的模型处理文本chunk"""
        try:
            if gpu_model is None:
                print(f"[GPU {gpu_id} ERROR] Model not loaded")
                results_queue.put((chunk_idx, None))
                return
            
            # 创建text encoder
            text_encoder = Qwen2AudioTextEncoder.__new__(Qwen2AudioTextEncoder)
            text_encoder.device = device
            text_encoder.processor = processor
            text_encoder.model = gpu_model
            text_encoder._analyze_model_structure()
            
            # 创建wrapper
            import torch.nn as nn
            class TextOnlyModel(nn.Module):
                def __init__(self, text_encoder, proj_text):
                    super().__init__()
                    self.text_encoder = text_encoder
                    self.proj_text = proj_text
                
                def encode_text(self, texts):
                    with torch.no_grad():
                        emb = self.text_encoder.predict(texts)
                    if not isinstance(emb, torch.Tensor):
                        emb = torch.as_tensor(emb)
                    emb = emb.float().to(self.proj_text.weight.device)
                    return F.normalize(self.proj_text(emb), dim=-1)
            
            # 复制投影层到GPU
            proj_text = torch.nn.Linear(
                model_config['text_hidden'],
                512
            ).to(device)
            proj_text.load_state_dict(model_config['proj_text_state_dict'])
            
            gpu_wrapper = TextOnlyModel(text_encoder, proj_text)
            gpu_wrapper.eval()
            
            print(f"[GPU {gpu_id}] Model fully loaded, processing {len(text_chunk)} texts...")
            
            # 打印GPU显存使用情况
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            print(f"[GPU {gpu_id}] Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            # 处理当前chunk
            chunk_embeddings = []
            failed_batches = 0
            
            for i in tqdm(range(0, len(text_chunk), batch_size), desc=f"GPU {gpu_id}", position=gpu_id):
                batch_texts = text_chunk[i:i+batch_size]
                try:
                    with torch.no_grad():
                        embeddings = gpu_wrapper.encode_text(batch_texts)
                        # 立即转为numpy并释放GPU tensor
                        embeddings_np = embeddings.cpu().float().numpy()
                        del embeddings
                    
                    chunk_embeddings.append(embeddings_np)
                    
                    # 每个batch后都清理缓存，更积极地管理内存
                    torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"[GPU {gpu_id} ERROR] Batch {i//batch_size}: {e}")
                    failed_batches += 1
                    dummy_emb = np.zeros((len(batch_texts), 512), dtype=np.float32)
                    chunk_embeddings.append(dummy_emb)
                    torch.cuda.empty_cache()
            
            if failed_batches > 0:
                print(f"[GPU {gpu_id} WARN] {failed_batches} batches failed and were replaced with zeros")
            
            chunk_embeddings = np.concatenate(chunk_embeddings, axis=0)
            print(f"[GPU {gpu_id}] ✅ Completed, encoded {len(chunk_embeddings)} texts")
            
            # 清理
            del gpu_model
            del text_encoder
            del gpu_wrapper
            del proj_text
            torch.cuda.empty_cache()
            
            # 将结果放入队列
            results_queue.put((chunk_idx, chunk_embeddings))
            
        except Exception as e:
            print(f"[GPU {gpu_id} ERROR] Fatal error: {e}")
            import traceback
            traceback.print_exc()
            results_queue.put((chunk_idx, None))
    
    # 启动多个线程，每个GPU一个线程（使用预加载的模型）
    threads = []
    for (gpu_id, device, processor, gpu_model), (chunk_idx, text_chunk) in zip(gpu_models, zip(range(len(text_chunks)), text_chunks)):
        t = threading.Thread(target=process_on_gpu_preloaded, args=(gpu_id, device, processor, gpu_model, text_chunk, chunk_idx))
        t.start()
        threads.append(t)
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    # 收集结果
    results = {}
    while not results_queue.empty():
        chunk_idx, embeddings = results_queue.get()
        if embeddings is not None:
            results[chunk_idx] = embeddings
    
    # 按顺序合并
    all_embeddings = []
    for i in range(len(text_chunks)):
        if i in results:
            all_embeddings.append(results[i])
        else:
            print(f"[WARN] Missing results for chunk {i}")
    
    if not all_embeddings:
        raise RuntimeError("No embeddings generated!")
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"\n[INFO] ✅ Encoded {len(all_embeddings)} texts total (parallel with LoRA)")
    
    return all_embeddings


def main():
    parser = argparse.ArgumentParser(description="多GPU生成文本索引（支持LoRA）")
    
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型路径')
    parser.add_argument('--glossary_path', type=str, required=True, help='词汇表JSON路径')
    parser.add_argument('--output_path', type=str, required=True, help='索引输出路径（.pkl）')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2-Audio-7B-Instruct", help='基础模型名称')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--num_gpus', type=int, default=4, help='使用的GPU数量')
    parser.add_argument('--batch_size', type=int, default=4, help='每个GPU的batch size')
    parser.add_argument('--import_glossary', type=str, default=None, help='Optional glossary JSON that overrides --glossary_path')
    parser.add_argument('--exclude_confused', action='store_true', help='Exclude glossary entries flagged as confused')
    
    args = parser.parse_args()
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available!")
        return 1
    
    available_gpus = torch.cuda.device_count()
    if args.num_gpus > available_gpus:
        print(f"[WARN] Requested {args.num_gpus} GPUs but only {available_gpus} available")
        args.num_gpus = available_gpus
    
    print(f"[INFO] Using {args.num_gpus} GPUs with LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
    print(f"[INFO] Batch size per GPU: {args.batch_size}")
    
    # 检查显存并给出建议
    for gpu_id in range(args.num_gpus):
        props = torch.cuda.get_device_properties(gpu_id)
        total_memory_gb = props.total_memory / 1024**3
        print(f"[INFO] GPU {gpu_id}: {props.name}, {total_memory_gb:.1f}GB")
        
        if total_memory_gb < 40:
            print(f"[WARN] GPU {gpu_id} has less than 40GB memory, may encounter OOM issues")
            print(f"[WARN] Consider reducing --batch_size (current: {args.batch_size})")
    
    # 检查环境变量
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        print("[WARN] PYTORCH_CUDA_ALLOC_CONF not set")
        print("[WARN] Recommend: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    
    # 加载词汇表
    print("\n" + "="*80)
    print("LOADING GLOSSARY")
    print("="*80)
    glossary_path = args.import_glossary or args.glossary_path
    print(f"[INFO] Loading glossary from: {glossary_path}")
    with open(glossary_path, "r", encoding="utf-8") as f:
        glossary_raw = json.load(f)

    if isinstance(glossary_raw, dict):
        glossary_items = list(glossary_raw.items())
    elif isinstance(glossary_raw, list):
        glossary_items = []
        for item in glossary_raw:
            if isinstance(item, dict) and item.get("term"):
                glossary_items.append((item["term"], item))
            elif isinstance(item, str):
                glossary_items.append((item, {"term": item}))
    else:
        raise ValueError(f"Unsupported glossary format: {type(glossary_raw)}")

    filtered_entries = []
    seen_terms = set()
    dropped_confused = 0
    for term, payload in glossary_items:
        if not term:
            continue
        entry = dict(payload) if isinstance(payload, dict) else {"term": term}
        entry.setdefault("term", term)
        normalized = entry["term"].strip().lower()
        if normalized in seen_terms:
            continue
        if args.exclude_confused and entry.get("confused", False):
            dropped_confused += 1
            continue
        seen_terms.add(normalized)
        filtered_entries.append(entry)

    if not filtered_entries:
        raise RuntimeError("No glossary entries left after filtering.")

    all_glossary_terms = [entry["term"] for entry in filtered_entries]
    print(f"[INFO] Loaded {len(filtered_entries)} terms (raw: {len(glossary_items)}, dropped_confused: {dropped_confused})")
    
    # 预先下载模型（避免多线程并发下载冲突）
    print("\n" + "="*80)
    print("PRE-DOWNLOADING MODEL (once)")
    print("="*80)
    
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    
    print("[INFO] Pre-downloading processor...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    
    print("[INFO] Pre-downloading model to cache (CPU)...")
    print("[INFO] This ensures all GPU threads use cached files, avoiding conflicts")
    # 预先下载到缓存，不占用GPU
    temp_model_cpu = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
    )
    print("[INFO] ✅ Model cached successfully")
    
    # 创建临时text encoder获取hidden size
    text_encoder_temp = Qwen2AudioTextEncoder.__new__(Qwen2AudioTextEncoder)
    text_encoder_temp.device = torch.device("cpu")
    text_encoder_temp.model_name = args.model_name
    text_encoder_temp.processor = processor
    text_encoder_temp.model = temp_model_cpu
    text_encoder_temp._analyze_model_structure()
    
    text_hidden = text_encoder_temp.get_hidden_size()
    print(f"[INFO] Text hidden size: {text_hidden}")
    
    # 释放CPU模型
    del temp_model_cpu
    del text_encoder_temp
    
    # 加载checkpoint并分离权重
    print(f"\n[INFO] Loading trained weights from: {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 处理DDP前缀
        if list(state_dict.keys())[0].startswith('module.'):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            state_dict = new_state_dict
        
        # 分离投影层权重和LoRA权重
        proj_text_dict = {}
        lora_state_dict = {}
        
        for k, v in state_dict.items():
            if 'proj_text' in k:
                proj_text_dict[k.replace('proj_text.', '')] = v
            elif 'lora_' in k or 'base_model' in k:
                # LoRA权重，需要映射
                if k.startswith('speech_qwen2_model.') or k.startswith('text_qwen2_model.'):
                    # 去掉前缀
                    new_key = k.split('.', 1)[1] if '.' in k else k
                    lora_state_dict[new_key] = v
                else:
                    lora_state_dict[k] = v
        
        print(f"[INFO] Found {len(proj_text_dict)} projection layer weights")
        print(f"[INFO] Found {len(lora_state_dict)} LoRA weights")
        
        # 加载投影层权重
        import torch.nn as nn
        proj_text_cpu = nn.Linear(text_hidden, 512)
        
        if proj_text_dict:
            proj_text_cpu.load_state_dict(proj_text_dict)
            print(f"[INFO] ✅ Loaded projection layer weights")
        else:
            print("[WARN] ⚠️  No proj_text weights found, using random initialization")
        
        # 统计LoRA权重类型
        lora_types = {}
        for k in lora_state_dict.keys():
            if 'lora_A' in k:
                lora_types['lora_A'] = lora_types.get('lora_A', 0) + 1
            elif 'lora_B' in k:
                lora_types['lora_B'] = lora_types.get('lora_B', 0) + 1
        print(f"[INFO] LoRA weight statistics:")
        for lora_type, count in lora_types.items():
            print(f"  - {lora_type}: {count} layers")
    
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 确定LoRA目标模块（与训练时一致）
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    
    # 创建模型配置字典
    model_config = {
        'model_name': args.model_name,
        'text_hidden': text_hidden,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'target_modules': target_modules,
        'proj_text_state_dict': proj_text_cpu.state_dict(),
        'lora_state_dict': lora_state_dict,
    }
    
    # 多GPU编码
    print("\n" + "="*80)
    print("ENCODING TEXTS WITH MULTI-GPU (with LoRA)")
    print("="*80)
    
    text_embeddings = encode_texts_parallel(
        model_config, 
        all_glossary_terms, 
        batch_size=args.batch_size,
        num_gpus=args.num_gpus
    )
    
    # 构建FAISS索引
    print("\n" + "="*80)
    print("BUILDING FAISS INDEX")
    print("="*80)
    
    index = faiss.IndexFlatL2(512)
    index.add(text_embeddings)
    print(f"[INFO] ✅ Built FAISS index with {index.ntotal} vectors")
    
    # 保存索引和term list
    print("\n" + "="*80)
    print("SAVING INDEX")
    print("="*80)
    
    index_data = {
        'faiss_index': faiss.serialize_index(index),
        'term_list': filtered_entries,
        'num_terms': len(all_glossary_terms),
        'embedding_dim': 512
    }
    
    with open(args.output_path, 'wb') as f:
        pickle.dump(index_data, f)
    
    print(f"[INFO] ✅ Saved index to: {args.output_path}")
    
    # 打印统计信息
    file_size = os.path.getsize(args.output_path) / 1024**3
    print(f"[INFO] Index file size: {file_size:.2f} GB")
    print(f"[INFO] Number of terms: {len(all_glossary_terms)}")
    
    print("\n" + "="*80)
    print("INDEX BUILDING COMPLETED")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

