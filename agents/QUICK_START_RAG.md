# Quick Start Guide: Adding RAG to Your Agent

This guide shows the key changes made to integrate RAG functionality into the Qwen3-Omni agent.

## Core Changes Summary

### 1. Import RAG Dependencies

```python
import json
import pickle
from peft import LoraConfig, get_peft_model

try:
    import faiss
except ImportError:
    faiss = None
```

### 2. Add `TermRAGRetriever` Class

Copy the entire `TermRAGRetriever` class from `infinisst_omni.py` (lines 102-350).

Key methods:
- `__init__()`: Loads FAISS index and contrastive model
- `_load_index()`: Deserializes FAISS index from pickle
- `_load_model()`: Loads Qwen2-Audio encoder with LoRA
- `retrieve()`: Returns list of `{term, translation}` dicts

### 3. Update Agent States

Add `references` field to state dataclass:

```python
@dataclass
class S2TAgentStates(AgentStates):
    src_len: int
    target_ids: list
    segment_idx: int
    messages: list
    references: list  # ← NEW
    MAX_SRC_LEN = 16000 * 30

    def reset(self):
        super().reset()
        self.src_len = 0
        self.target_ids = []
        self.segment_idx = 0
        self.messages = []
        self.references = []  # ← NEW
```

### 4. Initialize RAG Retriever in `__init__()`

```python
def __init__(self, args):
    super().__init__(args)
    # ... existing initialization ...
    
    # RAG retriever
    self.rag_retriever: Optional[TermRAGRetriever] = None
    self.rag_top_k = getattr(args, "rag_top_k", 5)
    self.rag_target_lang = getattr(args, "rag_target_lang", "zh")
    
    if getattr(args, "rag_enabled", False):
        logger.info("Initializing RAG retriever...")
        self.rag_retriever = TermRAGRetriever(
            index_path=getattr(args, "rag_index_path", None),
            model_path=getattr(args, "rag_model_path", None),
            base_model_name=getattr(args, "rag_base_model", "Qwen/Qwen2-Audio-7B-Instruct"),
            device=getattr(args, "rag_device", "cuda:1"),
            lora_r=getattr(args, "rag_lora_r", 16),
            lora_alpha=getattr(args, "rag_lora_alpha", 32),
            lora_dropout=getattr(args, "rag_lora_dropout", 0.0),
            top_k=self.rag_top_k,
            target_lang=self.rag_target_lang,
        )
        if not self.rag_retriever or not self.rag_retriever.enabled:
            logger.warning("RAG retriever not operational; continuing without references")
            self.rag_retriever = None
    else:
        logger.info("RAG retrieval disabled")
```

### 5. Add RAG Arguments

```python
@staticmethod
def add_args(parser):
    # ... existing args ...
    
    parser.add_argument("--rag-enabled", action="store_true", 
                        help="Enable glossary RAG retrieval")
    parser.add_argument("--rag-index-path", type=str, default=None,
                        help="Path to FAISS index (.pkl)")
    parser.add_argument("--rag-model-path", type=str, default=None,
                        help="Path to RAG model checkpoint (.pt)")
    parser.add_argument("--rag-base-model", type=str, 
                        default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--rag-device", type=str, default="cuda:1")
    parser.add_argument("--rag-top-k", type=int, default=5)
    parser.add_argument("--rag-target-lang", type=str, default="zh")
    parser.add_argument("--rag-lora-r", type=int, default=16)
    parser.add_argument("--rag-lora-alpha", type=int, default=32)
    parser.add_argument("--rag-lora-dropout", type=float, default=0.0)
```

### 6. Modify `_prepare_inputs()` to Accept References

**BEFORE:**
```python
def _prepare_inputs(self, states, increment):
    # ... system prompt ...
    
    states.messages.append({
        "role": "user",
        "content": [{"type": "audio", "audio": increment}]
    })
```

**AFTER:**
```python
def _prepare_inputs(self, states, increment, references):  # ← Add references parameter
    # ... system prompt ...
    
    # Build user content with audio and references
    user_content = [{"type": "audio", "audio": increment}]
    
    # Add references in the format: ", references: [...]"
    if references:
        reference_payload = {"reference": references}
        reference_text = f", references: {json.dumps(reference_payload, ensure_ascii=False)}"
    else:
        reference_text = ", references: []"
    
    user_content.append({"type": "text", "text": reference_text})
    
    states.messages.append({
        "role": "user",
        "content": user_content
    })
```

### 7. Perform RAG Retrieval in `policy()`

**BEFORE:**
```python
def policy(self, states):
    # ... length checks ...
    
    with synchronized_timer('generate'):
        increment = self._prepare_speech(states)
        inputs = self._prepare_inputs(states, increment)  # ← No references
```

**AFTER:**
```python
def policy(self, states):
    # ... length checks ...
    
    with synchronized_timer('generate'):
        increment = self._prepare_speech(states)
        
        # RAG retrieval
        references: List[Dict[str, str]] = []
        if self.rag_retriever:
            rag_audio_tensor = torch.tensor(increment, dtype=torch.float32)
            if rag_audio_tensor.numel() > 0:
                references = self.rag_retriever.retrieve(
                    rag_audio_tensor,
                    top_k=self.rag_top_k,
                    target_lang=self.rag_target_lang,
                )
                states.references = references
                if references:
                    print(f"[RAG] {json.dumps({'reference': references}, ensure_ascii=False)}")
        else:
            states.references = []
        
        inputs = self._prepare_inputs(states, increment, references)  # ← Pass references
```

## Testing

### 1. Verify RAG Files Exist

```bash
ls -lh /mnt/gemini/data2/jiaxuanluo/indices/qwen2_audio_term_index_acl6060.pkl
ls -lh /mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_level_modal_v2_best.pt
```

### 2. Run Test

```bash
sbatch scripts/infer/infinisst_omni_vllm_rag.sh
```

### 3. Check Logs

```bash
# Check for RAG initialization
tail -f scripts/infer/logs/infer_infinisst_omni_vllm_rag_*_1.err

# Look for:
# - "TermRAGRetriever initialized with X terms"
# - "[RAG] {reference: [{term: ..., translation: ...}]}"
```

## Expected Output

When RAG is working, you should see output like:

```
TermRAGRetriever initialized with 5234 terms (embedding_dim=512, top_k=5, target_lang=zh)
len(messages): 2
len(audios): 1
[RAG] {"reference": [{"term": "Kenneth D Rose", "translation": "Kenneth D Rose"}, {"term": "New Mexico", "translation": "新墨西哥州"}]}
input_ids size: torch.Size([1, 456])
generate: 1.2345 seconds
```

## Troubleshooting

### No RAG output?

Check:
1. `--rag-enabled` flag is set
2. Index and model paths are correct
3. RAG device is available (not used by main model)

### RAG retrieval too slow?

- Move RAG to GPU: `--rag-device cuda:1`
- Reduce top-k: `--rag-top-k 3`

### GPU memory issues?

- Use separate GPUs for RAG and main model
- Or use CPU for RAG: `--rag-device cpu` (slower but works)

## File Structure

```
agents/
├── infinisst_omni_vllm_rag.py  ← New agent with RAG
├── infinisst_omni.py           ← Reference implementation (OpenAI API)
└── README_omni_vllm_rag.md     ← Detailed documentation

scripts/infer/
└── infinisst_omni_vllm_rag.sh  ← SLURM script with RAG flags
```

## Key Differences from Training Format

Training format (from `term_training_dev.jsonl`):
```json
"<audio>, references: {\"term\": \"XXX\", \"translation\": \"YYY\"}"
```

Inference format (this implementation):
```json
", references: {\"reference\": [{\"term\": \"XXX\", \"translation\": \"YYY\"}, ...]}"
```

Note: The comma at the beginning is intentional to match the training format where audio comes first.

