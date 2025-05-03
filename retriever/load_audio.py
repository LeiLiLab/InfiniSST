import gc

import numpy as np
import librosa
import torch
import laion_clap
import json
import os
from transformers import ClapModel, ClapProcessor
import torchaudio

import functools
print = functools.partial(print, flush=True)

def get_audio_full_path(sid):
    doc_id = sid.split("_")[0]  # 提取文档ID，比如 'POD0000001165'
    source_prefix = doc_id[:3]  # POD, AUD, YOU
    id_num = int(doc_id[3:])  # 比如 '0000001165' -> 1165
    subdir_num = id_num // 100 + 1  # 每100个放一个Pxxxx子目录，比如0-99是P0001，100-199是P0002
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
    waveform, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    if start_time is not None and end_time is not None:
        start_sample = int(start_time * target_sr)
        end_sample = int(end_time * target_sr)
        waveform = waveform[:, start_sample:end_sample]

    audio_data = waveform.mean(dim=0).numpy()

    desired_length = target_sr * 5
    current_length = audio_data.shape[0]
    if current_length < desired_length:
        pad_width = desired_length - current_length
        audio_data = np.pad(audio_data, (0, pad_width), mode='constant')
    elif current_length > desired_length:
        audio_data = audio_data[:desired_length]

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


def main():
    #model = laion_clap.CLAP_Module(enable_fusion=False)
    #model.load_ckpt()  # download the default pretrained checkpoint
    output_dir = "./audio_embeddings"
    os.makedirs(output_dir, exist_ok=True)
    model_name = "laion/clap-htsat-unfused"
    processor = ClapProcessor.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ClapModel.from_pretrained(model_name).to(device)
    # 加载刚刚筛选的测试集
    with open("gigaspeech_test_samples.json") as f:
        test_samples = json.load(f)

    for sample in test_samples:
        sid = sample['sid']
        print(f'test:{sid}')
        audio_path = get_audio_full_path(sid)
        start_time = sample['begin_time']
        end_time = sample['end_time']

        audio_tensor = load_audio(audio_path, start_time=start_time, end_time=end_time)  # shape (1, T)
        raw_tensor = audio_tensor.squeeze(0)
        try:
            if raw_tensor is None or not isinstance(raw_tensor, torch.Tensor) or raw_tensor.isnan().any() or raw_tensor.isinf().any():
                print(f"[ERROR] Invalid audio input (NaN or Inf) for {sid}")
                continue
            print(f"[DEBUG] Processing audio input shape: {raw_tensor.shape}, dtype: {raw_tensor.dtype}")
            try:
                inputs = processor(audios=raw_tensor, sampling_rate=48000, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    audio_emb = model.get_audio_features(**inputs)

                # Save to file
                torch.save(audio_emb, os.path.join(output_dir, f"{sid}.pt"))
                print(f"[✅] Saved {sid} embedding.")

                # 清理显存和引用
                del inputs, audio_emb
                torch.cuda.empty_cache()
                gc.collect()
                # audio_emb_list.append(audio_emb.squeeze(0))  # Remove batch dim
                # print(f"now is {len(audio_emb_list)}")
            except Exception as inner_e:
                print(f"[ERROR] Exception during audio embedding extraction: {inner_e}")
        except BaseException as crash:
            print(f"[CRITICAL] Low-level crash during audio embedding for sample #{idx}: {sid} | {crash}")


if __name__ == '__main__':
    main()
    # import torchaudio
    #
    # audio_path = "/mnt/taurus/data/siqiouyang/datasets/gigaspeech/audio/youtube/P0053/YOU0000005205.opus"
    # start = 15114.5
    # end = 15117.8
    #
    # # 手动载入并打印 shape、是否有 NaN
    # waveform, sr = torchaudio.load(audio_path)
    # clip = waveform[:, int(start * sr):int(end * sr)]
    #
    # print("Clip shape:", clip.shape)
    # print("Has NaN:", not torch.isfinite(clip).all())
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
