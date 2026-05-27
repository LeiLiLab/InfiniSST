## Hypothesis

V5 refmatch precision SFT should recover zh exact TERM_ACC by training only on
GT term-map targets that exact-match the SFT reference, while retaining sparse
real retriever noise for robustness.

## Background / Motivation

V3 real/tagged/adv showed that term-map SFT can improve robustness, but zh
lm2/raw regressed against no-TM-SFT.  Error analysis showed many misses were
exact wording regressions, and the V4 target-match audit showed many source
match 100k glossary translations conflict with the SFT reference.

## What changed vs baseline

- Baseline W&B candidate: `sst_omni/dvpmpqma` (`v2_srcm_r8a32`)
- Prior negative control: V3-real `sst_omni/k2xo8quk`
- Data manifest: `20260521T1314__data_prepare__v5_refmatch_precision_termmap_zh`
- Training data: `/mnt/gemini/data1/jiaxuanluo/speech_llm_v5_refmatch_precision_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260521/train_s_zh_v5_refmatch_precision_termmap_lh1b88kw_tau073_srcmatch100k.jsonl`
- Dev data: `/mnt/gemini/data1/jiaxuanluo/speech_llm_v5_refmatch_precision_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260521/dev_s_zh_v5_refmatch_precision_termmap_lh1b88kw_tau073_srcmatch100k.jsonl`
- LoRA: rank 8, alpha 32
- Compute: taurus hold job `45269`, 2 GPUs when available
- Term-map format: legacy plain `source=target`

## Expected metrics

Primary check is downstream tagged ACL `zh lm2/raw`.  V5 should close the
TERM_ACC gap against no-TM-SFT while not losing the robustness benefits observed
from V3 term-map SFT.

## Verdict

Pending.
