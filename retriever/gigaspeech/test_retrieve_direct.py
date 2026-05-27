#!/usr/bin/env python3
"""
Test script for retrieve_direct method.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.streaming_rag_retriever import StreamingTermRAGRetriever

def test_retrieve_direct():
    """Test the retrieve_direct method."""
    
    print("=" * 80)
    print("Testing retrieve_direct method")
    print("=" * 80)
    print()
    
    # Configuration
    RAG_INDEX_PATH = "/mnt/gemini/data2/jiaxuanluo/indices/qwen2_audio_term_index_used_terms_lowercase.pkl"
    RAG_MODEL_PATH = "/mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_level_modal_v2_best.pt"
    RAG_BASE_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    RAG_DEVICE = "cuda:0"
    
    print("Initializing StreamingTermRAGRetriever...")
    retriever = StreamingTermRAGRetriever(
        index_path=RAG_INDEX_PATH,
        model_path=RAG_MODEL_PATH,
        base_model_name=RAG_BASE_MODEL,
        device=RAG_DEVICE,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        top_k=10,
        target_lang="zh",
        score_threshold=0.0,
        chunk_size=2.0,
        hop_size=1.0,
        batch_size=32,
        enable_top_n_filter=False,
    )
    
    if not retriever.enabled:
        print("❌ Retriever failed to initialize")
        return
    
    print("✅ Retriever initialized successfully")
    print()
    
    # Test with a real audio file
    test_audio_path = "/mnt/gemini/data1/jiaxuanluo/audio_clips_siqi_v3/YOU0000010238/66/0.wav"
    
    print(f"Testing with audio file: {test_audio_path}")
    
    try:
        import librosa
        audio, sr = librosa.load(test_audio_path, sr=16000, mono=True)
        print(f"Loaded audio: {len(audio)} samples, {len(audio)/16000:.2f} seconds")
        print()
        
        # Test retrieve_direct
        print("Calling retrieve_direct...")
        results = retriever.retrieve_direct(audio, top_k=10)
        
        print(f"Retrieved {len(results)} terms:")
        print()
        
        for i, ref in enumerate(results, 1):
            print(f"{i}. {ref['term']} = {ref['translation']}")
            print(f"   key: {ref['key']}, score: {ref['score']:.4f}")
        
        print()
        print("=" * 80)
        print("✅ Test completed successfully!")
        print("=" * 80)
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_retrieve_direct()


















