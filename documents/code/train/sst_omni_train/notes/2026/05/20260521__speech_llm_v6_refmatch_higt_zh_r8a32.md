## Hypothesis

V6 high-GT refmatch SFT should better match the deployed retriever's 90%+
strict-term recall regime and recover zh exact TERM_ACC while retaining sparse
empty/noisy term-map robustness.

## Background / Motivation

V5 removed reference-conflicting GT translations but only placed about 78% of
filtered GT terms into term maps.  Since the actual retriever usually recalls
strict terms at 90%+, the SFT distribution should not underexpose the model to
available GT terms.

## What changed vs baseline

- Baseline W&B candidate: `sst_omni/dvpmpqma` (`v2_srcm_r8a32`)
- Data manifest: `20260521T1335__data_prepare__v6_refmatch_higt_termmap_zh`
- Training data: `/mnt/gemini/data1/jiaxuanluo/speech_llm_v6_refmatch_higt_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260521/train_s_zh_v6_refmatch_higt_termmap_lh1b88kw_tau073_srcmatch100k.jsonl`
- Dev data: `/mnt/gemini/data1/jiaxuanluo/speech_llm_v6_refmatch_higt_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260521/dev_s_zh_v6_refmatch_higt_termmap_lh1b88kw_tau073_srcmatch100k.jsonl`
- LoRA: rank 8, alpha 32
- Compute: taurus hold job `45269`, 2 GPUs when available
- Term-map format: legacy plain `source=target`

## Expected metrics

Primary downstream check is tagged ACL `zh lm2/raw`.  V6 should outperform V3
and V5-style retriever-SFT on exact TERM_ACC and ideally narrow the gap to
no-TM-SFT, while keeping the robustness gains that motivated term-map SFT.

## Verdict

Pending.
