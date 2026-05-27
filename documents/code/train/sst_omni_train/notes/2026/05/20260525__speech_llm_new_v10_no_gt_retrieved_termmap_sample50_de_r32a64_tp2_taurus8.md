## Hypothesis

Training de on New V10 sample50 no-GT-retrieved term-map data should recover
BLEU lost by NewV9 no-GT-zero while preserving most of the RASST terminology
gain.

## Background / Motivation

The clean de NewV9 SLM used `term_map:NONE` for no-GT chunks and produced high
TERM_ACC but weak BLEU. New V10 repairs only that exposure mismatch by keeping
retriever-filtered term maps on no-GT chunks, then applying the same
assistant-side `<term>` GT target wrapping as NewV9.  Full V10 was too dense on
no-GT chunks, so this route uses term-level 50% sampling.

## What changed vs baseline

- Data event: `20260525T0025__data_prepare__new_v10_no_gt_retrieved_termmap_sample50_mfa_npfilter_oldnewv3_de`
- Train data: `/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v10_no_gt_retrieved_termmap_sample50_mfa_npfilter_oldnewv3_de_20260525/train_s_de_new_v10_no_gt_retrieved_termmap_sample50_mfa_npfilter_oldnewv3.jsonl`
- Dev data: `/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v10_no_gt_retrieved_termmap_sample50_mfa_npfilter_oldnewv3_de_20260525/dev_s_de_new_v10_no_gt_retrieved_termmap_sample50_mfa_npfilter_oldnewv3_first355.jsonl`
- no-GT term-map entries: `206,314 -> 102,842` by term-level Bernoulli
  sampling with `keep_prob=0.5`.
- LoRA and runtime match clean de NewV9: rank 32, alpha 64, TP=2, EP=2, 8 Taurus GPUs, one epoch.

## Expected metrics

After export, first gate is tagged ACL raw `de/lm=2` with HN1024, `tau=0.79`,
same-lm batch, and `max_new_tokens=80`. The gate target is BLEU no worse than
the verified no-RAG baseline while TERM_ACC stays clearly above no-RAG.

## Verdict

Pending training completion and HF export.
