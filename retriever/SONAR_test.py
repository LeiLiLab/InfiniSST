from new_giga_speech import load_preprocessed_samples
from chunk_splitter import load_and_chunk_preprocessed_samples
import torch
import torch.nn.functional as F

if __name__ == "__main__":
    path = "data/test_preprocessed_samples_merged.json"
    
    # Load chunked samples (2-second chunks)
    print("Loading and chunking samples...")
    chunked_samples = load_and_chunk_preprocessed_samples(
        path, 
        chunk_duration=2.0, 
        target_sr=48000, 
        overlap=0.0,
        max_samples=2  # Only process first 2 original samples
    )
    
    # Use the first two chunks for testing
    test_samples = chunked_samples[:2]
    print(f"Using {len(test_samples)} chunks for testing")
    print(f"First chunk shape: {test_samples[0]['audio_tensor'].shape}")
    print(f"Second chunk shape: {test_samples[1]['audio_tensor'].shape}")


    # 重采样 audio_tensor 从 48k 到 16k
    import torchaudio
    orig_tensor = test_samples[0]["audio_tensor"]

    if isinstance(orig_tensor, torch.Tensor):
        orig_tensor = orig_tensor.squeeze()
    else:
        orig_tensor = torch.tensor(orig_tensor).squeeze()

    resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)
    sample = resampler(orig_tensor).numpy()
    
    print(f"Original 48kHz chunk duration: {orig_tensor.shape[0] / 48000:.2f}s")
    print(f"Resampled 16kHz chunk duration: {sample.shape[0] / 16000:.2f}s")

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

    # For chunked samples, we need to handle audio differently
    # We'll use the resampled tensor data instead of loading from path
    audio_path = test_samples[0].get("audio")
    if isinstance(audio_path, dict):
        audio_path = audio_path.get("path")
    
    # Use the already processed chunk data
    waveform = sample  # This is the 16kHz resampled chunk
    
    # For speech embedding, we can use the original audio path or save the chunk temporarily
    import tempfile
    import soundfile as sf
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        sf.write(tmp_file.name, waveform, 16000)
        speech_res = s2vec_model.predict([tmp_file.name])[0]
        import os
        os.unlink(tmp_file.name)  # Clean up temporary file
    print(speech_res.shape)

    # 正例 - first chunk with its corresponding text
    def get_chunk_speech_embedding(chunk_sample):
        """Get speech embedding for a chunk sample"""
        chunk_tensor = chunk_sample["audio_tensor"]
        if isinstance(chunk_tensor, torch.Tensor):
            chunk_tensor = chunk_tensor.squeeze()
        
        # Resample to 16kHz for SONAR
        resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)
        chunk_16k = resampler(chunk_tensor).numpy()
        
        # Save temporarily for SONAR processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, chunk_16k, 16000)
            embedding = s2vec_model.predict([tmp_file.name])[0]
            os.unlink(tmp_file.name)
        return embedding
    
    chunk1_embedding = get_chunk_speech_embedding(test_samples[0])
    chunk2_embedding = get_chunk_speech_embedding(test_samples[1])
    
    sim_pos = F.cosine_similarity(
        torch.tensor(t2vec_model.predict([texts[0]],source_lang="eng_Latn")[0]),
        torch.tensor(chunk1_embedding),
        dim=0,
    ).item()

    # 负例 - first text with second chunk
    sim_neg = F.cosine_similarity(
        torch.tensor(t2vec_model.predict([texts[0]],source_lang="eng_Latn")[0]),
        torch.tensor(chunk2_embedding),
        dim=0,
    ).item()

    print(f"正例相似度: {sim_pos:.4f}, 负例相似度: {sim_neg:.4f}")
    print(f"\nChunk information:")
    print(f"Chunk 1 - Original ID: {test_samples[0].get('original_segment_id', 'N/A')}")
    print(f"Chunk 1 - Chunk index: {test_samples[0].get('chunk_index', 'N/A')}")
    print(f"Chunk 1 - Total chunks: {test_samples[0].get('total_chunks', 'N/A')}")
    print(f"Chunk 2 - Original ID: {test_samples[1].get('original_segment_id', 'N/A')}")
    print(f"Chunk 2 - Chunk index: {test_samples[1].get('chunk_index', 'N/A')}")
    print(f"Chunk 2 - Total chunks: {test_samples[1].get('total_chunks', 'N/A')}")



