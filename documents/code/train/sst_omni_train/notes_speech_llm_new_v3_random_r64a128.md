## Hypothesis

Combining rank 64 / alpha 128 with the random-thinned new_v3 dataset may recover the benefits of the corrected TCM term maps while avoiding the noise cost of very dense retriever-only negatives.

## Background / Motivation

The random new_v3 dataset preserves GT terms and halves retriever-only negatives in expectation, bringing average term-map size back near the legacy speech-LLM distribution. This run tests whether the higher-rank adapter benefits from that cleaner term context more than from the full dense new_v3 maps.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/sst_omni/runs/5e1iu7zo
- Diff: train on `/mnt/gemini/data1/jiaxuanluo/tcm_wiki100k_gt_zh_tau075_d9_k20_postfiltercap_termmap_v2_sourcefinal_gtzh/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_sourcefinal_tcmwiki100kgt_tau075_d9_k20_postfiltercap_gtzhoverride_new_v3_random.jsonl`; preserve GT terms, halve retriever-only negatives in expectation with seed 42, and use LoRA rank 64 / alpha 128 on the 8-GPU recipe.
- Historical comparison target: old v4_ner HF checkpoint `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4` remains non-WandB historical debt for SimulEval only.

## Expected metrics

Primary follow-up is ACL one-paper/per-paper SimulEval with raw/gs1k/gs10k glossaries at tau 0 and 0.75. Expect this run to beat full new_v3 rank64/alpha128 if negative density is the main source of false-copy errors.

## Verdict

Pending training and targeted SimulEval.
