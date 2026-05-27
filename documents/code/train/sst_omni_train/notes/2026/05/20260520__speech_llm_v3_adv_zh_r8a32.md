## Hypothesis

Adding a small adversarial bucket with false-positive and translation-swap term_map entries should improve robustness to large-glossary noise, especially `lm=1/gs10k`, without needing higher LoRA rank.

## Background / Motivation

The `ja lm1/gs10k` failure shows high strict-term coverage but severe generation collapse from noisy term_map entries.  This run keeps plain term_map formatting and adds adversarial chunks where unsupported terms or deliberately wrong translations appear in the term_map; the reference output remains unchanged, teaching the model to ignore unsupported entries.

## What changed vs baseline

- Baseline W&B candidate: `sst_omni/dvpmpqma` (`v2_srcm_r8a32`)
- Training data: `/mnt/gemini/data1/jiaxuanluo/speech_llm_v3_robust_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260520/train_s_zh_v3_adv_termmap_lh1b88kw_tau073_srcmatch100k.jsonl`
- Dev data: `/mnt/gemini/data1/jiaxuanluo/speech_llm_v3_robust_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260520/dev_s_zh_v3_adv_termmap_lh1b88kw_tau073_srcmatch100k.jsonl`
- Data manifest: `20260520T0000__data_prepare__v3_robust_termmap_zh`
- LoRA: rank 8, alpha 32
- Compute: taurus hold job `45269`, 2 GPUs
- Term-map format: legacy plain `source=target`
- Adversarial entries: translation-swap plus real retriever false positives.

## Expected metrics

This should reduce noisy glossary over-copy and glossary enumeration in gs10k settings, while retaining enough clean/term-critical supervision to avoid large oracle TERM_ACC loss.

## Verdict

Pending.
