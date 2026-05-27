## Hypothesis

Speech LLM SFT should see term_map noise that matches the intended inference
path: the lh1b88kw retriever is called on the current streaming speech chunk
plus 1.92s lookback, then filtered at tau=0.73 without GT backfill.

## Background / Motivation

The oracle-GT term_map run measures an upper bound, but it does not expose the
Speech LLM to missed terms, no-term chunks with residual retriever noise, or
timeline-boundary behavior.  V1 constructs zh SFT data from the existing
streaming chunks instead of forcing lm=1, so the training distribution keeps the
original lm=1..12 random chunk schedule.

Reference retriever background:
`documents/code/train/term_train/reports/20260518_lh1b88kw_tau073_retriever_readout_for_speech_llm.md`.

## What changed vs baseline

- Baseline data: `/mnt/gemini/data/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl`
- Dev data: `/mnt/gemini/data1/jiaxuanluo/dev_s_zh_v4_ner_baseline_aligned_freq_k20.jsonl`
- Retriever checkpoint: `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt`
- Glossary: `/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample100000_zh100k_train_gt.json`
- Retrieval policy: for each existing streaming chunk, encode `[chunk_start - 1.92s, chunk_end]`; only keep MaxSim evidence windows overlapping the current chunk.
- Filtering: `top_k=10`, `score_threshold=0.73`.
- Term-map policy: no GT backfill; `term_map:NONE` when no retrieved terms survive threshold; chunk-specific GT zh replaces glossary zh only when the retriever actually retrieves the same source term.

## Expected metrics

Dataset QA should report:

- rows/chunks kept and any dropped-row reasons;
- term_map density and no-GT noise exposure;
- GT-term recall of retriever-generated term maps;
- duration-bucket breakdown for lm1, lm2-4, lm5-6, and lm7+ chunks;
- sample chunks with GT terms and retriever term_map entries.

For the downstream SFT, the first comparable run should use LoRA `r=8, alpha=32`
to isolate the data-policy effect from capacity changes.  Larger `r=32,
alpha=64` should be a separate capacity ablation.

## Verdict

Deprecated before training use.  The V1 dataset generation itself completed on
Taurus hold `45269`, but post-run QA showed the input zh100k glossary does not
cover the source JSONL `gt_terms_by_chunk` vocabulary well enough for the
reported GT-term recall to be meaningful.  A large part of the 47.20% recall is
therefore a glossary/GT surface mismatch, not a retriever-quality signal.

Do not use the generated V1 train/dev JSONL for SFT.  Regenerate with a
GT-union glossary first, then re-check coverage and retriever-term hit stats.
