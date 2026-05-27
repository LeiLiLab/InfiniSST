## Hypothesis

Training the speech LLM on the new_v3 TCM term-map dataset with LoRA rank 32 and alpha 64 should test whether the denser tau-filtered negatives improve term adoption without changing model capacity relative to the rank-32 baseline.

## Background / Motivation

The schema-compliant rank-32 baseline `5e1iu7zo` used the corrected new_v2 source-final dataset with GT zh override. The new_v3 dataset keeps the same baseline-final source JSONL and GT override, but rebuilds term maps from the d9, tau=0.75, post-filter-cap retriever output with a hard 20-term cap.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/sst_omni/runs/5e1iu7zo
- Diff: train on `/mnt/gemini/data1/jiaxuanluo/tcm_wiki100k_gt_zh_tau075_d9_k20_postfiltercap_termmap_v2_sourcefinal_gtzh/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_sourcefinal_tcmwiki100kgt_tau075_d9_k20_postfiltercap_gtzhoverride.jsonl`; keep Qwen3-Omni base, max length 3072, one epoch, 8-GPU EP=4/TP=2/SP, LoRA rank 32, and raise alpha from 32 to 64.
- Historical comparison target: old v4_ner HF checkpoint `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4` remains non-WandB historical debt for SimulEval only.

## Expected metrics

Primary follow-up is ACL one-paper/per-paper SimulEval with raw/gs1k/gs10k glossaries at tau 0 and 0.75. Expect denser new_v3 term maps to improve TERM_ACC or adoption over `5e1iu7zo`; a TERM_FCR increase would argue that the extra negatives are too noisy.

## Verdict

Pending training and targeted SimulEval.
