# GSDedup TCM-off conv5 ACL raw 1k/10k eval

- **Family / data / task**: `sst_ood_hardneg` / `3variant_gsv2full_gsfix_mfa_gsdedup_acl_raw1k10k_eval` / `eval`
- **Variant tag**: `acl_raw1k10k_gsdedup_conv5`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_gsdedup_tcmoff_conv5_acl_raw1k10k_eval_v3.sh`
- **Checkpoint source run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/7xu2b4so
- **Checkpoint**: `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_gsdedup_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_conv5_bs12k_smallest_dense_normAGGR_8gpu_aries_best.pt`

## Hypothesis

The deduplicated TCM-off conv5 best checkpoint should preserve the high dev
gs10000 recall observed during training while showing whether the GigaSpeech
deduplication changes ACL6060 raw/gs1k/gs10k transfer quality.

## Background / Motivation

Job 45195 improved the deduplicated dev gs10000 best from `0.9732` to `0.9763`.
The training run did not include ACL6060 process metrics, so this eval-only job
uses the established ACL raw 1k/10k launcher setup to measure transfer on the
new best checkpoint.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/7xu2b4so
- Diff:
  - checkpoint: evaluate the gsdedup conv5 primary best checkpoint
  - eval set: ACL6060 extracted-paper dev plus dev-v3 no-term data
  - glossary: ACL6060 GT-union gs10000 with sizes `1000 10000`
  - inference tau: fixed sweep point `0.75`
  - loss/training: eval-only, TCM loss disabled

## Expected metrics

ACL6060 `topk10_filtered_recall@tau_0p75_gs10000` should be competitive with
the non-dedup TCM-off baseline, while no-term emitted average should remain
low enough to keep downstream Speech LLM filtering practical.

## Verdict

PENDING: update after the ACL raw 1k/10k eval finishes.
