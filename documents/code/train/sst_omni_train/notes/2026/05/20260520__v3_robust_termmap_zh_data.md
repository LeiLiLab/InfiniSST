## Hypothesis

A deployment-stress SFT curriculum built from source-glossary exact GT plus real retriever term_map entries will train the Speech LLM to handle empty/sparse term_map chunks and large-glossary noisy term_map chunks better than the dense V2 retriever dataset.

## Background / Motivation

The prior V2 retriever-timeline dataset was too dense: almost every chunk had a non-empty top-10 term_map, including chunks with no GT term.  This does not match the failure cases seen in simuleval: `de lm3/raw` can enter English-copy mode when early chunks have no term_map, while `ja lm1/gs10k` can over-copy noisy false positives.  The historical LLM-extracted `gt_terms_by_chunk` is not the main target here; input GT is the source-glossary exact-match `srcmatch100k` JSONL.

## What changed vs baseline

- Baseline data event: `20260519T1235__data_prepare__retriever_timeline_termmap_zh_lh1b88kw_tau073_srcmatch100k`
- Input train JSONL: `/mnt/gemini/data1/jiaxuanluo/speech_llm_retriever_timeline_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260519/train_s_zh_retriever_timeline_lh1b88kw_tau073_srcmatch100k_k10_lb1p92.jsonl`
- Input dev JSONL: `/mnt/gemini/data1/jiaxuanluo/speech_llm_retriever_timeline_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260519/dev_s_zh_retriever_timeline_lh1b88kw_tau073_srcmatch100k_k10_lb1p92.jsonl`
- New outputs:
  - `real`: plain term_map, density-stratified robust mixture.
  - `tagged`: same mixture, but term entries are formatted as `[TERM] source => target [/TERM]`.
  - `adv`: same plain-format mixture with a small adversarial bucket using translation-swap and false-positive distractors.
- Builder: `documents/code/train/sst_omni_train/src/build_robust_termmap_sft.py`

## Expected metrics

Training data should have much lower no-GT non-empty term_map rate than V2, lower average term_map density, and explicit mode coverage for `empty`, `sparse_noise`, `clean_gt`, `realistic`, `partial_noisy`, `term_critical`, and `adversarial` where applicable.

## Verdict

Success.  Built all three V3 train/dev datasets with no dropped rows.  Compared with the dense V2 retriever term_map data, train no-GT non-empty term_map rate dropped from about 91.8% to 26-27%, and average term_map entries per chunk dropped from 9.64 to about 2.26 for real/tagged and 2.43 for adv.  The adv variant includes 3699 translation-swap adversarial entries.
