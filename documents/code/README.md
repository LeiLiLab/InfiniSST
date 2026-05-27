# Code (dataset utilities)

This folder contains small, reproducible utilities used in experiments described in `documents/data/sst_omni_train_dataset.md`.

## 1) term_map sampling (reduce density)

Script: `sample_term_map_dataset.py`

Typical usage (ablation with different keep ratios):

```bash
python /home/jiaxuanluo/InfiniSST/documents/code/sample_term_map_dataset_fixed_term_maps_limit.py \
  --input /mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_enriched_with_negatives.jsonl \
  --output /mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_enriched_with_negatives.sample_keep0.8.seed1.jsonl \
  --keep-chunk-ratio 0.8 \
  --entry-sample-ratio 1 \
  --seed 1
```

Notes:
- `--keep-chunk-ratio` controls the fraction of chunks that keep a non-empty `term_map`.
- `--entry-sample-ratio` controls how many entries remain inside a kept `term_map`.
- If `gt_terms_by_chunk` exists (zh dataset), GT terms are always kept in `term_map`; `--entry-sample-ratio` only samples non-GT entries. `gt_terms_by_chunk` itself is not modified.
- If `audios` exists, this script requires `len(audios)` to exactly match the number of user `"<audio>"` chunks in `messages` (mismatch will raise an error).

## 2) automated training for multiple sampling ratios

Script: `auto_train_sampling.sh`

This runs `megatron sft` + `swift export` for multiple keep ratios and updates the table in `documents/data/sst_omni_train_dataset.md`.

```bash
bash documents/code/auto_train_sampling.sh
```

Logs:
- By default, training logs are written to `documents/logs/auto_train_sampling/`.
- You can override with `TRAIN_LOG_DIR=/your/path`.




Results:
/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_enriched_with_negatives.sample_keep0.5.seed1.jsonl
instances=12500
chunks_total=70269
chunk_with_term_map_ratio_before=0.976519 after=0.488210
mean_terms_per_chunk_before=16.925629 after=8.492493




```
python documents/code/sample_term_map_dataset.py \
  --input /mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_enriched_with_negatives.jsonl \
  --output /mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_enriched_with_negatives.sample_keep0.8.seed1.jsonl \
  --keep-chunk-ratio 0.8 \
  --seed 1
```

instances=12500
chunks_total=70269
chunk_with_term_map_ratio_before=0.976519 after=0.780700
mean_terms_per_chunk_before=16.925629 after=13.579260



0.3:
python documents/code/sample_term_map_dataset.py \
  --input /mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_enriched_with_negatives.jsonl \
  --output /mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_enriched_with_negatives.sample_keep0.3.seed1.jsonl \
  --keep-chunk-ratio 0.3 \
  --seed 1


sampling_result:
instances=12500
chunks_total=70269
chunk_with_term_map_ratio_before=0.976519 after=0.290996
mean_terms_per_chunk_before=16.925629 after=5.049780



1.0:

python documents/code/sample_term_map_dataset.py \
  --input /mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_enriched_with_negatives.jsonl \
  --output /mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_enriched_with_negatives.sample_keep1.0.seed1.jsonl \
  --keep-chunk-ratio 1 \
  --seed 1

instances=12500
chunks_total=70269
chunk_with_term_map_ratio_before=0.976519 after=0.976519
mean_terms_per_chunk_before=16.925629 after=16.917930

这个就相当于只是限制了100这个max_term_maps






python documents/code/sample_term_map_dataset.py \
  --input /mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_enriched_with_negatives.jsonl \
  --output /mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_enriched_with_negatives.sample_keep0.5.seed1.jsonl \
  --keep-chunk-ratio 1.0 \
  --entry-sample-ratio 0.5 \
  --seed 1


Done.
instances=12500
chunks_total=70269
chunk_with_term_map_ratio_before=0.976519 after=0.976519
mean_terms_per_chunk_before=16.925629 after=8.514124



python documents/code/sample_term_map_dataset.py \
  --input /mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_enriched_with_negatives.jsonl \
  --output /mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_enriched_with_negatives.sample_keep0.8.seed1.jsonl \
  --keep-chunk-ratio 1.0 \
  --entry-sample-ratio 0.8 \
  --seed 1

Done.
instances=12500
chunks_total=70269
chunk_with_term_map_ratio_before=0.976519 after=0.976519
mean_terms_per_chunk_before=16.925629 after=13.555366




python documents/code/sample_term_map_dataset.py \
  --input /mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_enriched_with_negatives.jsonl \
  --output /mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_enriched_with_negatives.sample_keep0.3.seed1.jsonl \
  --keep-chunk-ratio 1.0 \
  --entry-sample-ratio 0.3 \
  --seed 1

instances=12500
chunks_total=70269
chunk_with_term_map_ratio_before=0.976519 after=0.976519
mean_terms_per_chunk_before=16.925629 after=5.203703