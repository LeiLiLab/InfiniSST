# Data-driven term_map (v3): NER/Heuristics + LLM Alignment + RAG Hard Negatives

This v3 pipeline shifts from **static glossary-driven GT** to **data-driven GT alignment**:

- **Candidate extraction** from `src_text` (heuristics + optional TF-IDF).
- **LLM alignment**: find the corresponding Chinese expression in `tgt_text` (or `null`).
- **GT assignment to chunks** by locating the aligned term in `src_chunk_text`.
- **Hard negative mining** using existing audio RAG (sliding window, max pooling).

All logs and user-facing strings are in English.

## Scripts (recommended two-stage split)

Because some environments have spaCy but broken FAISS (or vice versa), we recommend a two-stage pipeline:

1) **Stage 1 (GT only)**: spaCy + vLLM alignment (no FAISS/RAG dependency)
- `retriever/gigaspeech/handle_train_dataset_for_term_map_v3_stage1_gt.py`

2) **Stage 2 (RAG + term_map)**: StreamingTermRAGRetriever hard negative mining + inject final `term_map`
- `retriever/gigaspeech/handle_train_dataset_for_term_map_v3_stage2_rag.py`

## Stage 1: LLM backend

Stage 1 uses **in-process vLLM only** (no OpenAI API calls).

```bash
python /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/handle_train_dataset_for_term_map_v3_stage1_gt.py \
  --align-model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --align-batch-size 8 \
  --align-gpu-memory-util 0.85 \
  --align-max-num-seqs 16 \
  --align-tensor-parallel-size 1 \
  --align-max-model-len 4096

## Stage 2: RAG hard negatives + final term_map

```bash
python /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/handle_train_dataset_for_term_map_v3_stage2_rag.py \
  --input-gt-jsonl /mnt/gemini/data1/jiaxuanluo/train_s_zh_v3_gt_terms.jsonl \
  --output-base /mnt/gemini/data1/jiaxuanluo/train_s_zh_with_candidates_v3_data_driven \
  --rag-top-k 20 \
  --rag-batch-size 64 \
  --multiple-range 0 9 \
  --all-negative-ratio 0.1
```
```

## Term extraction knobs

- `--max-terms-per-utterance`: cap candidates per utterance.
- `--max-global-term-freq`: low-frequency filter. Keep terms whose extracted frequency ≤ this.
- `--min-term-chars`: minimum chars for a term to be considered.

## Output

- Single GPU: `${output_base}.jsonl`
- Multi GPU: `${output_base}_gpu{gpu_id}.jsonl`

## Notes

- Candidate extraction uses **spaCy NER (PERSON/ORG/GPE/PRODUCT) + noun_chunks + heuristics**.
- TF-IDF extraction is optional: if `scikit-learn` is not installed, TF-IDF extraction is skipped.
- spaCy model is configurable via `--spacy-model` (default: `en_core_web_sm`).


