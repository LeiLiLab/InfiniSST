## Hypothesis

Randomly thinning new_v3 retriever-only negatives while preserving GT terms should reduce training-time term-map noise and make the distribution closer to the legacy speech-LLM dataset, improving adoption without increasing false-copy rate.

## Background / Motivation

The old speech-LLM dataset had an accidental LLM-generate negative path plus a random negative-count selection, yielding about 8.16 term-map entries per chunk. The full new_v3 dataset has about 14.49 entries per chunk; the new_v3_random dataset keeps the same GT terms but samples retriever-only negatives so the average negative count is roughly halved.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/sst_omni/runs/5e1iu7zo
- Diff: train on `/mnt/gemini/data1/jiaxuanluo/tcm_wiki100k_gt_zh_tau075_d9_k20_postfiltercap_termmap_v2_sourcefinal_gtzh/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_sourcefinal_tcmwiki100kgt_tau075_d9_k20_postfiltercap_gtzhoverride_new_v3_random.jsonl`; preserve GT terms, keep a random prefix length from the existing retriever-only negatives per chunk with seed 42, use LoRA rank 32 and alpha 64 on the same 8-GPU recipe.
- Historical comparison target: old v4_ner HF checkpoint `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4` remains non-WandB historical debt for SimulEval only.

## Expected metrics

Primary follow-up is ACL one-paper/per-paper SimulEval with raw/gs1k/gs10k glossaries at tau 0 and 0.75. Expect lower TERM_FCR and similar or better TERM_ACC than full new_v3 if excess negatives were hurting the SLM.

## Verdict

Pending training and targeted SimulEval.
