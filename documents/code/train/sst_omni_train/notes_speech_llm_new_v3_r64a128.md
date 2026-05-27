## Hypothesis

Increasing LoRA capacity to rank 64 with alpha 128 on the full new_v3 dataset may help the speech LLM use denser tau-filtered term maps while keeping the same source-final and GT-override fixes.

## Background / Motivation

The corrected rank-32 baseline `5e1iu7zo` improved over the earlier new_v1 data but still left a gap to the historical v4_ner checkpoint. The full new_v3 dataset raises term-map density through d9 tau-filtered retrieval with a hard 20-term cap, so this run tests whether extra LoRA capacity can absorb that larger term context.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/sst_omni/runs/5e1iu7zo
- Diff: train on `/mnt/gemini/data1/jiaxuanluo/tcm_wiki100k_gt_zh_tau075_d9_k20_postfiltercap_termmap_v2_sourcefinal_gtzh/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_sourcefinal_tcmwiki100kgt_tau075_d9_k20_postfiltercap_gtzhoverride.jsonl`; keep max length 3072 and one epoch; change LoRA from rank 32 / alpha 32 baseline to rank 64 / alpha 128 on the 8-GPU recipe.
- Historical comparison target: old v4_ner HF checkpoint `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4` remains non-WandB historical debt for SimulEval only.

## Expected metrics

Primary follow-up is ACL one-paper/per-paper SimulEval with raw/gs1k/gs10k glossaries at tau 0 and 0.75. Expect rank64/alpha128 to improve adoption over rank32/alpha64 if capacity is the bottleneck; worse TERM_FCR would argue against using the full dense new_v3 maps.

## Verdict

Pending training and targeted SimulEval.
