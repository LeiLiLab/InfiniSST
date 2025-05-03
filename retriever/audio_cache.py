import os
import torch
import numpy as np
from pathlib import Path
import torchaudio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
from audio_utils import get_audio_full_path

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

class AudioCache:
    def __init__(self, cache_dir="./data/audio_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, audio_path, start_time, end_time):
        # 生成缓存文件名
        base_name = Path(audio_path).stem
        if start_time is not None and end_time is not None:
            cache_name = f"{base_name}_{start_time:.2f}_{end_time:.2f}.pt"
        else:
            cache_name = f"{base_name}_full.pt"
        return self.cache_dir / cache_name
    
    def load_audio(self, audio_path: str, start_time: float = None, end_time: float = None, target_sr: int = 48000) -> torch.Tensor:
        cache_path = self.get_cache_path(audio_path, start_time, end_time)
        
        # 如果缓存存在，直接加载
        if cache_path.exists():
            return torch.load(cache_path)
        
        # 否则处理音频并缓存
        waveform, sr = torchaudio.load(audio_path)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)

        if start_time is not None and end_time is not None:
            start_sample = int(start_time * target_sr)
            end_sample = int(end_time * target_sr)
            waveform = waveform[:, start_sample:end_sample]

        audio_data = waveform.mean(dim=0).numpy()
        usable_length = (audio_data.shape[0] // target_sr) * target_sr
        if usable_length < target_sr:
            raise ValueError("Audio too short after processing.")
        audio_data = audio_data[:usable_length]
        audio_data = audio_data.reshape(1, -1)
        audio_data = torch.from_numpy(
            int16_to_float32(float32_to_int16(audio_data))
        ).float()
        
        # 保存到缓存
        torch.save(audio_data, cache_path)
        return audio_data

    def preprocess_gigaspeech_samples(self, json_path=None, target_sr=48000, max_workers=16):
        """预处理 Gigaspeech 测试样本中的所有音频文件"""
        print(f"Loading test samples from {json_path}..." if json_path else "Loading full GigaSpeech dataset...")

        if json_path:
            with open(json_path, 'r') as f:
                test_samples = json.load(f)
        else:
            with open("/mnt/taurus/data/siqiouyang/datasets/gigaspeech/GigaSpeech.json", 'r') as f:
                raw = json.load(f)
                test_samples = []
                for doc in raw.get("audios", []):
                    for segment in doc.get("segments", []):
                        sid = segment.get("sid")
                        begin_time = segment.get("begin_time", 0)
                        end_time = segment.get("end_time", 0)
                        if sid and end_time - begin_time >= 1.0:
                            test_samples.append({
                                "sid": sid,
                                "begin_time": begin_time,
                                "end_time": end_time
                            })
        
        # 获取所有需要处理的音频文件路径
        audio_files = []
        for sample in test_samples:
            try:
                audio_path = get_audio_full_path(sample['sid'])
                audio_files.append((audio_path, sample['begin_time'], sample['end_time']))
            except Exception as e:
                print(f"Error getting audio path for {sample['sid']}: {e}")
        
        print(f"Found {len(audio_files)} audio files to process")
        
        # 使用多线程处理所有音频文件
        from concurrent.futures import as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self.load_audio,
                    audio_path,
                    start_time=start_time,
                    end_time=end_time,
                    target_sr=target_sr
                )
                for audio_path, start_time, end_time in audio_files
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Gigaspeech audio files"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error during audio processing: {e}")
        
        print("Gigaspeech preprocessing completed!")

if __name__ == "__main__":
    # 创建缓存实例
    cache = AudioCache()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_test_samples', action='store_true', help="If set, use test_samples.json as the input JSON file.")
    parser.add_argument('--max_workers', type=int, default=16, help="Maximum number of threads to use.")
    args = parser.parse_args()

    json_path = "data/gigaspeech_test_samples.json" if args.use_test_samples else None
    cache.preprocess_gigaspeech_samples(json_path=json_path, max_workers=args.max_workers)