# Term Map Dataset Construction Script

## Overview

This script (`handle_train_dataset_for_term_map_v2_buzz.py`) constructs an omni fine-tuning dataset by augmenting audio chunks with candidate terminology translations from a RAG retrieval system.

## Purpose

The script enhances simultaneous interpretation training data by:
1. Matching ground truth (GT) terms from a glossary using FlashText
2. Retrieving additional candidate terms using a RAG (Retrieval-Augmented Generation) system
3. Mixing GT and candidate terms with random sampling
4. Formatting them as `term_map` entries in the training messages

## Input Files

1. **`train_s_zh_baseline.jsonl`**: Original training messages
   - Format: Each line contains `messages` (conversation) and `audios` (list of audio file paths)
   - Example:
     ```json
     {
       "messages": [
         {"role": "system", "content": "..."},
         {"role": "user", "content": "<audio>"},
         {"role": "assistant", "content": "translation chunk 1"},
         {"role": "user", "content": "<audio>"},
         {"role": "assistant", "content": "translation chunk 2"}
       ],
       "audios": ["path/to/chunk0.wav", "path/to/chunk1.wav"]
     }
     ```

2. **`train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv`**: Alignment data
   - TSV format with columns: `id`, `audio`, `n_frames`, `speaker`, `src_lang`, `tgt_lang`, `src_trajectory`, `asr`, `src_text`, `tgt_text`, `trajectory`
   - The `id` field matches the utter_id extracted from audio paths
   - `src_trajectory` and `trajectory` are lists of text segments

3. **`glossary_used.json`**: Glossary for terminology matching
   - Format: `{term_key: {term, target_translations: {zh, de, es, ...}}}`

4. **RAG Index and Model**:
   - FAISS index: `/mnt/gemini/data2/jiaxuanluo/indices/qwen2_audio_term_index_used_terms.pkl`
   - Model checkpoint: `/mnt/gemini/data2/jiaxuanluo/ckpts/qwen2_audio_siqi_contrastive_lora_used_terms_epoch5/checkpoint_epoch_5.pt`

## Output

**`train_s_zh_with_candidates.jsonl`**: Augmented messages with term_map

Example output:
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<audio>\n\nterm_map:\nsocial statement=社会声明\nlet=让\nrelationship=关系"},
    {"role": "assistant", "content": "translation"}
  ],
  "audios": ["path/to/audio.wav"]
}
```

## Algorithm

### Step 1: Extract utter_id
- From audio path `/path/to/YOU0000010238/66/0.wav` → extract `YOU0000010238_66`

### Step 2: Match TSV row
- Look up utter_id in TSV to get `src_trajectory`, `tgt_trajectory`, `src_text`, `tgt_text`

### Step 3: Split trajectories into chunks
- Split source and target trajectories evenly based on number of audio chunks
- Each chunk gets a portion of the trajectory

### Step 4: Match GT terms (FlashText)
- For each chunk's source text, use FlashText to find terms from glossary
- Extract (source_term, target_translation) pairs

### Step 5: RAG retrieval
- Batch process all audio chunks through RAG model
- Retrieve top-10 candidate terms for each chunk

### Step 6: Sample and mix candidates
- For each chunk with GT terms:
  - Randomly select a multiple `m` from [1, 4]
  - Sample `|GT| * m` candidates from RAG results (excluding GT terms)
  - Combine GT + sampled candidates
  - Shuffle and deduplicate (case-insensitive)

### Step 7: Generate term_map
- Format: 
  ```
  term_map:
  source_term1=target_translation1
  source_term2=target_translation2
  ```
- Append to `<audio>` placeholder in user messages

## Usage

### Full Processing
```bash
python handle_train_dataset_for_term_map_v2_buzz.py
```

### Dry Run (test with 10 messages)
```bash
python handle_train_dataset_for_term_map_v2_buzz.py --dry-run
```

### Custom Message Limit
```bash
python handle_train_dataset_for_term_map_v2_buzz.py --max-messages 100
```

## Configuration

Edit the script to modify:
- `RAG_INDEX_PATH`: Path to FAISS index
- `RAG_MODEL_PATH`: Path to RAG model checkpoint
- `RAG_TOP_K`: Number of candidates to retrieve (default: 10)
- `RAG_BATCH_SIZE`: Batch size for RAG inference (default: 32)
- `MULTIPLE_RANGE`: Random sampling multiplier range (default: [1, 4])

## Dependencies

```bash
pip install torch transformers peft faiss-gpu flashtext librosa tqdm numpy
```

## Notes

1. **Memory**: The script loads the entire TSV into memory for fast lookup
2. **GPU**: RAG retrieval requires CUDA (configured for `cuda:0`)
3. **Two-pass processing**: 
   - Pass 1: Collect all audio paths
   - Pass 2: Batch RAG retrieval + augmentation
4. **Error handling**: Skips entries with missing audio files or TSV mismatches
5. **Reproducibility**: Uses random sampling, set `random.seed()` for deterministic results

## Troubleshooting

### TSV too large
- The script uses streaming + indexing to handle large TSV files
- Only keeps necessary fields in memory

### Audio files not found
- Check that audio paths in jsonl match actual file locations
- Script logs warnings for missing files but continues processing

### RAG model OOM
- Reduce `RAG_BATCH_SIZE` (default: 32)
- Use smaller model or quantization

### No candidates generated
- Check that glossary contains terms matching the source text
- Verify RAG index and model are correctly loaded
- Use `--dry-run` to inspect intermediate results

## Example Workflow

```bash
# 1. Test with dry run
python handle_train_dataset_for_term_map_v2_buzz.py --dry-run

# 2. Check output
head -n 3 /mnt/gemini/data1/jiaxuanluo/train_s_zh_with_candidates.jsonl | jq .

# 3. Run full processing
python handle_train_dataset_for_term_map_v2_buzz.py

# 4. Verify output
wc -l /mnt/gemini/data1/jiaxuanluo/train_s_zh_with_candidates.jsonl
```

## Author

Created for InfiniSST simultaneous interpretation project.


















