#!/usr/bin/env python3
"""
mmap 音频读取器：在 Modal 上零拷贝随机读取音频数据
支持多进程 DataLoader 共享 .dat 的只读 mmap
"""

import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset

class MMapAudioShard:
    """单个音频分片的 mmap 读取器"""
    
    def __init__(self, dat_path: str, index_path: str):
        self.dat_path = Path(dat_path)
        self.idx = np.load(index_path, allow_pickle=True)
        
        # 索引数据
        self.key = self.idx["key"]          # object[]
        self.relpath = self.idx["relpath"]  # object[]
        self.offset = self.idx["offset"]    # int64[]
        self.length = self.idx["length"]    # int64[]
        self.sr = int(self.idx["sr"][0])
        self.dtype = np.int16 if self.idx["dtype"][0] == "int16" else np.float32
        
        # 创建内存映射
        self.mm = np.memmap(self.dat_path, mode="r", dtype=self.dtype)

    def __len__(self):
        return len(self.key)

    def get_by_index(self, i: int):
        """通过索引获取音频数据"""
        off = int(self.offset[i])
        n = int(self.length[i])
        
        # 零拷贝视图
        view = self.mm[off:off+n]
        
        # 转换为 float32
        if self.dtype == np.int16:
            wav = (view.astype(np.float32) / 32767.0)
        else:
            wav = view.astype(np.float32, copy=False)
        
        return wav, self.sr, str(self.key[i]), str(self.relpath[i])

    def close(self):
        """关闭内存映射"""
        if hasattr(self, 'mm'):
            del self.mm

class MMapAudioCollection:
    """聚合所有分片，并提供 key/relpath 到 (shard,i) 的映射"""
    
    def __init__(self, shard_dir: str):
        shard_dir = Path(shard_dir)
        dats = sorted(shard_dir.glob("shard_*.dat"))
        
        if not dats:
            raise FileNotFoundError(f"No .dat files found in {shard_dir}")
        
        print(f"[INFO] Loading {len(dats)} audio shards from {shard_dir}")
        
        self.shards = []
        self.k2loc = {}     # key -> (s_idx, i)
        self.r2loc = {}     # relpath -> (s_idx, i)
        
        for s_idx, dat in enumerate(dats):
            idx = dat.with_suffix(".index.npz")
            if not idx.exists():
                raise FileNotFoundError(f"Index file not found: {idx}")
            
            shard = MMapAudioShard(dat, idx)
            self.shards.append(shard)
            
            # 建立 key 和 relpath 的映射
            for i, k in enumerate(shard.key):
                self.k2loc[str(k)] = (s_idx, i)
            for i, r in enumerate(shard.relpath):
                self.r2loc[str(r)] = (s_idx, i)
        
        print(f"[INFO] Loaded {len(self)} audio samples from {len(self.shards)} shards")

    def __len__(self):
        return sum(len(s) for s in self.shards)

    def get_by_key(self, key: str):
        """通过 key 获取音频数据"""
        if key not in self.k2loc:
            raise KeyError(f"Audio key not found: {key}")
        s, i = self.k2loc[key]
        return self.shards[s].get_by_index(i)

    def get_by_relpath(self, relpath: str):
        """通过相对路径获取音频数据"""
        if relpath not in self.r2loc:
            raise KeyError(f"Audio relpath not found: {relpath}")
        s, i = self.r2loc[relpath]
        return self.shards[s].get_by_index(i)

    def close(self):
        """关闭所有分片"""
        for s in self.shards:
            s.close()

class MMapAudioDataset(Dataset):
    """基于 mmap 的音频数据集，用于 PyTorch DataLoader"""
    
    def __init__(self, shard_dir, keys=None, max_len=16000*30, min_len=1600):
        """
        Args:
            shard_dir: mmap 分片目录
            keys: 要使用的 audio key 列表，如果为 None 则使用所有可用的 key
            max_len: 最大音频长度（采样点数）
            min_len: 最小音频长度（采样点数）
        """
        self.db = MMapAudioCollection(shard_dir)
        self.keys = list(keys) if keys is not None else list(self.db.k2loc.keys())
        self.max_len = max_len
        self.min_len = min_len
        
        print(f"[INFO] MMapAudioDataset initialized with {len(self.keys)} samples")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        """获取单个样本"""
        key = self.keys[idx]
        wav, sr, key, rel = self.db.get_by_key(key)
        
        # 简单裁切/补零处理
        if wav.shape[0] > self.max_len:
            wav = wav[:self.max_len]
        if wav.shape[0] < self.min_len:
            pad = self.min_len - wav.shape[0]
            wav = np.pad(wav, (0, pad), mode="constant")
        
        return key, torch.from_numpy(wav.copy()), sr, rel

    def close(self):
        """关闭数据库连接"""
        self.db.close()

def collate_pad(batch):
    """DataLoader 的 collate 函数，支持动态填充"""
    keys, waves, srs, rels = zip(*batch)
    
    # 找到最大长度并填充
    max_len = max(w.shape[0] for w in waves)
    padded_waves = []
    
    for w in waves:
        if w.shape[0] < max_len:
            pad = max_len - w.shape[0]
            w = torch.nn.functional.pad(w, (0, pad), mode='constant', value=0.0)
        padded_waves.append(w)
    
    waves = torch.stack(padded_waves, dim=0)  # [B, T]
    return keys, waves, srs, rels

def extract_audio_key_from_path(audio_path: str) -> str:
    """从音频文件路径提取 mmap key
    
    例如：
    /mnt/gemini/data1/jiaxuanluo/term_chunks/POD/POD0000000002/POD0000000002_S0000091_term_Artesia_9.12_9.74_ctx1.0s.wav
    -> POD/POD0000000002/POD0000000002_S0000091_term_Artesia_9.12_9.74_ctx1.0s
    """
    path = Path(audio_path)
    
    # 找到 term_chunks 后的部分
    parts = path.parts
    if 'term_chunks' in parts:
        idx = parts.index('term_chunks')
        if idx + 3 < len(parts):  # term_chunks/POD/POD0000000002/file.wav
            top_dir = parts[idx + 1]  # POD/YOU/AUD
            sub_dir = parts[idx + 2]  # POD0000000002
            filename = path.stem       # 去掉后缀
            return f"{top_dir}/{sub_dir}/{filename}"
    
    # 如果无法解析，直接使用文件名（去后缀）
    return path.stem

# 测试函数
def test_mmap_reader(shard_dir: str):
    """测试 mmap 读取器"""
    print(f"[TEST] Testing MMapAudioCollection with {shard_dir}")
    
    try:
        db = MMapAudioCollection(shard_dir)
        print(f"[TEST] Loaded {len(db)} audio samples")
        
        # 测试随机读取
        if len(db.k2loc) > 0:
            test_key = list(db.k2loc.keys())[0]
            wav, sr, key, rel = db.get_by_key(test_key)
            print(f"[TEST] Sample audio: key={key}, shape={wav.shape}, sr={sr}, relpath={rel}")
        
        db.close()
        print("[TEST] Test completed successfully")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_mmap_reader(sys.argv[1])
    else:
        print("Usage: python mmap_audio_reader.py <shard_dir>")



















