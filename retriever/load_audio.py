
import numpy as np
import librosa
import torch
import laion_clap
import json
import os
from transformers import ClapModel, ClapProcessor

import functools
print = functools.partial(print, flush=True)

def get_audio_full_path(sid):
    doc_id = sid.split("_")[0]  # 提取文档ID，比如 'POD0000001165'
    source_prefix = doc_id[:3]  # POD, AUD, YOU
    id_num = int(doc_id[3:])  # 比如 '0000001165' -> 1165
    subdir_num = id_num // 100 + 1  # 每1000个放一个Pxxxx子目录，比如0-999是P0001，1000-1999是P0002
    subdir = f"P{subdir_num:04d}"  # 格式化成P0001这样

    # source到文件夹名字的映射
    source_map = {
        "POD": "podcast",
        "YOU": "youtube",
        "AUD": "audiobook"
    }
    source_folder = source_map.get(source_prefix)
    if source_folder is None:
        raise ValueError(f"Unknown source prefix: {source_prefix}")

    return os.path.join(
        "/mnt/taurus/data/siqiouyang/datasets/gigaspeech/audio",
        source_folder,
        subdir,
        f"{doc_id}.opus"
    )


def load_audio(audio_path: str, start_time: float = None, end_time: float = None, target_sr: int = 48000) -> torch.Tensor:
    audio_data, sr = librosa.load(audio_path, sr=target_sr)
    if start_time is not None and end_time is not None:
        start_sample = int(start_time * target_sr)
        end_sample = int(end_time * target_sr)
        audio_data = audio_data[start_sample:end_sample]

    # 🔥 确保是1D array
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=0)

    # 🔥 保证长度是480000 samples
    desired_length = 48000*5
    current_length = audio_data.shape[0]
    if current_length < desired_length:
        pad_width = desired_length - current_length
        audio_data = np.pad(audio_data, (0, pad_width), mode='constant')
    elif current_length > desired_length:
        audio_data = audio_data[:desired_length]

    # 🔥 转成(1, T)，然后规范float32
    audio_data = audio_data.reshape(1, -1)
    audio_data = torch.from_numpy(
        int16_to_float32(float32_to_int16(audio_data))
    ).float()

    print(f"[DEBUG] Loaded audio shape: {audio_data.shape}, path: {audio_path}, start_time: {start_time}, end_time: {end_time}")
    return audio_data




# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()  # download the default pretrained checkpoint

# 加载刚刚筛选的测试集
with open("gigaspeech_test_samples.json") as f:
    test_samples = json.load(f)

sample = test_samples[0]  # 拿第一个样本
sid = sample['sid']
print(f'test:{sid}')
audio_path = get_audio_full_path(sid)
begin_time = sample['begin_time']
end_time = sample['end_time']

audio_tensor = load_audio(audio_path, start_time=start_time, end_time=end_time)  # shape (1, T)
raw_tensor = audio_tensor.squeeze(0)
try:
    if raw_tensor is None or not torch.isfinite(raw_tensor).all():
        print(f"[ERROR] Invalid audio input (NaN or None) for sample #{idx}: {sid}")

    print(f"[DEBUG] Processing audio input shape: {raw_tensor.shape}, dtype: {raw_tensor.dtype}")

    try:
        model_named = "laion/clap-htsat-unfused"
        processor = ClapProcessor.from_pretrained(model_name)
        model = ClapModel.from_pretrained(model_name).to(device)
        inputs = processor(audios=raw_tensor, sampling_rate=48000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(600)  # 600 seconds = 10 minutes
        try:
            with torch.no_grad():
                audio_emb = retriever.model.get_audio_features(**inputs)
        except TimeoutException:
            print(f"[ERROR] Timeout during inference for sample #{idx}: {sid}")
        finally:
            signal.alarm(0)  # Cancel alarm
        audio_emb_list.append(audio_emb.squeeze(0))  # Remove batch dim
    except Exception as inner_e:
        print(f"[ERROR] Exception during audio embedding extraction: {inner_e}")
except BaseException as crash:
    print(f"[CRITICAL] Low-level crash during audio embedding for sample #{idx}: {sid} | {crash}")
#
# # 从文件直接提取embedding
# audio_embed = model.get_audio_embedding_from_filelist(x=[audio_path], use_tensor=True)
# print(f"Audio embedding shape (from file): {audio_embed.shape}")
# print(audio_embed[0][:10])
#
# # 或者，从音频数据提取embedding
# audio_data, _ = librosa.load(audio_path, sr=48000)  # CLAP要求48000Hz
# audio_data = audio_data.reshape(1, -1)  # (1, T)
# audio_data = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()
#
# audio_embed_from_data = model.get_audio_embedding_from_data(x=audio_data, use_tensor=True)
# print(f"Audio embedding shape (from data): {audio_embed_from_data.shape}")
# print(audio_embed_from_data[0][:10])



