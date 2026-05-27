## Hypothesis

Historical TM-SFT SLMs with the HN1024 retriever and tau=0.78 should provide a stable tagged-ACL reference curve for zh/de/ja across lm=1,2,3,4 when empty retrieved term maps are omitted rather than rendered as `term_map: NONE`.

## Background / Motivation

The previous de lm=2 TM-SFT+HN1024 comparison was useful, but it covered only one language/lm and used the older default empty-term-map behavior. The current diagnostic evidence suggests `empty_term_map_policy=omit` better matches training-time no-term chunks.

## What changed vs baseline

- Evaluate the historical TM-SFT SLMs:
  - `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4`
  - `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4`
  - `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-ja-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4`
- Use tagged ACL raw inputs and `acl6060_tagged_gt_raw_min_norm2`.
- Use HN1024, top-k=10, tau=0.78, lookback=1.92s.
- Use same-lm batch eval, max_new_tokens=80, `VLLM_LIMIT_AUDIO=128`, `VLLM_MAX_MODEL_LEN=12288`.
- Set `empty_term_map_policy=omit`.

## Expected metrics

Expect de lm=2 to be close to the previously verified TM-SFT+HN1024 de lm=2 point, with possible BLEU/TERM_ACC changes from omitting empty term-map blocks. The main goal is a complete 3-language, 4-lm reference curve, not model selection.

## Verdict

Pending. Validate from per-lm `eval_results.tsv`, `instances.log`, `instances.strip_term.log`, and the merged summary TSV under the output root.
