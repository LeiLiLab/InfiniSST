from new_giga_speech import load_preprocessed_samples
import torch
import torch.nn.functional as F

if __name__ == "__main__":
    path = "data/test_preprocessed_samples_merged.json"
    test_samples = load_preprocessed_samples(path)[:2]


    # 重采样 audio_tensor 从 48k 到 16k
    import torchaudio
    orig_tensor = test_samples[0]["audio_tensor"]

    if isinstance(orig_tensor, torch.Tensor):
        orig_tensor = orig_tensor.squeeze()
    else:
        orig_tensor = torch.tensor(orig_tensor).squeeze()

    resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)
    sample = resampler(orig_tensor).numpy()

    import torch
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

    t2vec_model = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=torch.device("cuda"),
        dtype=torch.float32,
    )

    texts = [i['text'] for i in test_samples]
    embeddings = t2vec_model.predict(texts, source_lang="eng_Latn")
    print(embeddings.shape)
    # torch.Size([2, 1024])


    # embedding the audio
    from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
    import librosa

    s2vec_model = SpeechToEmbeddingModelPipeline(encoder="sonar_speech_encoder_eng",device=torch.device("cuda"),
        fbank_dtype=torch.float32)

    audio_path = test_samples[0]["audio"]
    waveform, _ = librosa.load(audio_path, sr=16000)

    speech_res = s2vec_model.predict([audio_path])[0]
    print(speech_res.shape)

    # 正例
    sim_pos = F.cosine_similarity(
        torch.tensor(t2vec_model.predict([texts[0]],source_lang="eng_Latn")[0]),
        torch.tensor(s2vec_model.predict([test_samples[0]["audio"]])[0]),
        dim=0,
    ).item()

    # 负例
    sim_neg = F.cosine_similarity(
        torch.tensor(t2vec_model.predict([texts[0]],source_lang="eng_Latn")[0]),
        torch.tensor(s2vec_model.predict([test_samples[1]["audio"]])[0]),
        dim=0,
    ).item()

    print(f"正例相似度: {sim_pos:.4f}, 负例相似度: {sim_neg:.4f}")



