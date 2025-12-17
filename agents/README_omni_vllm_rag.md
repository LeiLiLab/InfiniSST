# InfiniSST Omni VLLM RAG Agent

This agent integrates RAG (Retrieval-Augmented Generation) functionality into the Qwen3-Omni model for simultaneous speech translation with glossary support.

## Key Features

1. **RAG-Enhanced Translation**: Retrieves relevant terminology from a pre-built glossary index to improve translation quality
2. **VLLM Support**: Uses vLLM for efficient inference with prefix caching
3. **Multi-turn Conversation**: Maintains conversation history with automatic trimming
4. **Streaming Translation**: Processes audio chunks incrementally for low-latency translation

## Architecture

### Main Components

1. **TermRAGRetriever**: 
   - Loads FAISS index and trained contrastive model
   - Retrieves top-k relevant terms for each audio chunk
   - Returns term-translation pairs in target language

2. **InfiniSSTOmniVLLMRAG**:
   - Main agent class handling the translation pipeline
   - Integrates audio processing, RAG retrieval, and generation

### Data Flow

```
Audio Input → Audio Preprocessing → RAG Retrieval → Prompt Construction → Model Generation → Translation Output
```

## RAG Format

The references are added to the user prompt in this format:

```json
{
  "role": "user",
  "content": [
    {"type": "audio", "audio": <audio_array>},
    {"type": "text", "text": ", references: {\"reference\": [{\"term\": \"XXX\", \"translation\": \"YYY\"}, ...]}"}
  ]
}
```

This matches the training format from `term_training_dev.jsonl`:

```json
{"role": "user", "content": "<audio>, references: {\"term\": \"Kenneth D Rose\", \"translation\": \"Kenneth D Rose\"}"}
```

## Usage

### Prerequisites

1. RAG index file (`.pkl`):
   - Contains FAISS index + term list with translations
   - Example: `/mnt/gemini/data2/jiaxuanluo/indices/qwen2_audio_term_index_acl6060.pkl`

2. RAG model checkpoint (`.pt`):
   - Trained contrastive model for audio embedding
   - Example: `/mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_level_modal_v2_best.pt`

3. Qwen3-Omni model:
   - Base model or LoRA-tuned version
   - Example: `/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct-lora/v4-20251114-122213-hf`

### Running with SLURM

```bash
sbatch scripts/infer/infinisst_omni_vllm_rag.sh
```

### Command-Line Arguments

#### RAG-Specific Arguments

- `--rag-enabled`: Enable RAG retrieval (flag)
- `--rag-index-path`: Path to FAISS index pickle file
- `--rag-model-path`: Path to trained RAG model checkpoint
- `--rag-base-model`: Base model name for RAG encoder (default: `Qwen/Qwen2-Audio-7B-Instruct`)
- `--rag-device`: Device for RAG model (default: `cuda:1`)
- `--rag-top-k`: Number of terms to retrieve per chunk (default: 5)
- `--rag-target-lang`: Target language code for translations (default: `zh`)
- `--rag-lora-r`: LoRA rank for RAG model (default: 16)
- `--rag-lora-alpha`: LoRA alpha for RAG model (default: 32)
- `--rag-lora-dropout`: LoRA dropout for RAG model (default: 0.0)

#### Model Arguments

- `--use-vllm`: Enable vLLM (1) or use HuggingFace (0) (default: 0)
- `--model-name`: Path to Qwen3-Omni model
- `--max-cache-chunks`: Maximum audio chunks to cache (default: 120)
- `--keep-cache-chunks`: Number of chunks to keep when trimming (default: 60)

#### Generation Arguments

- `--temperature`: Sampling temperature (default: 0.6)
- `--top-p`: Nucleus sampling parameter (default: 0.95)
- `--top-k`: Top-k sampling parameter (default: 20)
- `--max-new-tokens`: Maximum tokens to generate per turn

## Implementation Details

### RAG Retrieval Process

1. **Audio Preprocessing**: Convert audio chunk to tensor
2. **Embedding Extraction**: Use trained contrastive model to encode audio
3. **FAISS Search**: Find top-k nearest neighbors in term index
4. **Result Formatting**: Extract term-translation pairs for target language

### Message Construction

The agent maintains a conversation history:

```python
[
  {"role": "system", "content": [{"type": "text", "text": "..."}]},
  {"role": "user", "content": [{"type": "audio", ...}, {"type": "text", ...}]},
  {"role": "assistant", "content": [{"type": "text", "text": "..."}]},
  ...
]
```

When the history exceeds `2 * max_cache_chunks + 1` messages, it trims to keep only:
- System prompt (always kept)
- Last `2 * keep_cache_chunks` messages (user-assistant pairs)

### RAG Device Assignment

The RAG retriever uses a **separate GPU** (`cuda:2` by default) to avoid competing with the main translation model:

```
GPU 0: vLLM (tensor_parallel_size=2, first half)
GPU 1: vLLM (tensor_parallel_size=2, second half)
GPU 2: RAG Retriever (Qwen2-Audio-7B)
```

**Why 3 GPUs?**
- vLLM requires `tensor_parallel_size=2` for Qwen3-Omni-30B-MoE model
- RAG model (~7B params) needs its own GPU to avoid conflicts
- Code sets `CUDA_VISIBLE_DEVICES=0,1` so vLLM only sees GPU 0-1

## Differences from Original Agent

### Changes from `infinisst_omni.py` (OpenAI API version):

1. **Model Backend**: Direct vLLM support instead of OpenAI-compatible API
2. **Message Format**: Uses Qwen3-Omni's native format with audio arrays
3. **Audio Encoding**: No base64 encoding needed (direct array passing)
4. **Reference Format**: Slightly different JSON structure for consistency

### Changes from Student's Original Code:

1. **Added RAG Integration**: Complete `TermRAGRetriever` class
2. **Modified `_prepare_inputs`**: Accepts `references` parameter and adds to user content
3. **Enhanced `policy`**: Performs RAG retrieval before prompt construction
4. **Added State Field**: `references` field in `S2TAgentStates`

## Troubleshooting

### Common Issues

1. **FAISS not found**: Install with `pip install faiss-cpu` or `faiss-gpu`
2. **NumPy version conflict**: Use `numpy<2` for FAISS compatibility
3. **Out of memory**: Reduce `max_cache_chunks` or use smaller model
4. **RAG model fails to load**: Check device availability and model path

### Debug Output

The agent prints diagnostic information:
- `len(messages)`: Current message count
- `len(audios)`: Number of audio chunks being processed
- `[RAG] {...}`: Retrieved references (when available)
- `input_ids size`: Input tensor shape
- `before trim` / `after trim`: Message count during history trimming

## Performance Considerations

- **RAG Retrieval**: Adds ~0.1-0.3s per chunk (on dedicated GPU)
- **Memory Usage**: 
  - GPU 0-1: ~18GB each (vLLM with TP=2)
  - GPU 2: ~8GB (RAG model)
  - Total: ~44GB across 3 GPUs
- **Cache Management**: Automatic trimming prevents unbounded memory growth
- **Parallel Processing**: RAG on GPU 2, vLLM on GPU 0-1 (no interference)

## Future Enhancements

- [ ] Support for multiple target languages in single session
- [ ] Dynamic top-k based on audio length
- [ ] Confidence-based filtering of retrieved terms
- [ ] Batch retrieval for multiple chunks
- [ ] Integration with online glossary updates

