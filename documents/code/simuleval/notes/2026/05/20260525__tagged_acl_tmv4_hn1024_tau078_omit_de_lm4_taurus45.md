## Hypothesis

For En-De tagged ACL at lm=4, the historical TM-SFT SLM plus HN1024 at tau=0.78 with omitted empty term maps should provide the high-latency comparison point against the lm=2 omit run and the older non-omit TM-SFT reference.

## Background / Motivation

The previous Taurus lm=2 omit run completed with valid eval artifacts. This run checks whether the same setting at lm=4 preserves BLEU and terminology when the streaming policy has more context.

## What changed vs baseline

- Model: `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4`
- Retriever: HN1024, top-k=10, tau=0.78, lookback=1.92s
- Dataset/glossary: tagged ACL raw, `acl6060_tagged_gt_raw_min_norm2`
- Batch setting: same-lm batch, de/lm=4 only, five talks, max_new_tokens=80
- Runtime prompt policy: `empty_term_map_policy=omit`
- Compute: Taurus GPU pair 4,5; launch waits until the pair is idle.

## Expected metrics

Expected to be comparable to the older TM-SFT+HN1024 curve while testing whether omitting empty term-map blocks changes the lm=4 BLEU/TERM_ACC tradeoff.

## Verdict

Completed with valid eval artifacts. The launcher reported `[ALL DONE]`, wrote `eval_results.tsv`, and produced both per-lm and top-level summary TSVs.

Verified metrics from `eval_results.tsv`: BLEU=32.5332, StreamLAAL=2671.0635, StreamLAAL_CA=541.2869, TERM_ACC=0.8406 (786/935). Both `instances.log` and `instances.strip_term.log` contain 5 rows.
