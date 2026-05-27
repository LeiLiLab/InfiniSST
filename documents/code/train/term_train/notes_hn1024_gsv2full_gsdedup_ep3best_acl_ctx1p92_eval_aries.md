# HN1024 GSV2full gsdedup ep3 best ACL 1.92s readout

- **Family / data / task**: `sst_ood_hardneg` / `acl6060_ctx192_ep3best_eval` / `eval`
- **Variant tag**: `acl_ctx192_gsdedup_ep3best`
- **Launcher**: `documents/code/train/term_train/run_eval_mfa_smallest_dense_hn1024_gsv2full_gsdedup_ep3best_acl_ctx1p92_1gpu_aries.sh`
- **Checkpoint**: `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_gsdedup_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep3_bs12k_smallest_dense_normAGGR_8gpu_aries_best.pt`

## Hypothesis

The 1.92s GSV2-dedup ep3 best checkpoint should reproduce the old short-window
ACL behavior when evaluated on the 1.92s extracted-paper ACL JSONL. This is the
control needed before comparing the same checkpoint on 3.84s ACL chunks.

## Background / Motivation

The current `dxwrgbln` run evaluates ACL6060 with 3.84s chunks. To isolate
whether any ACL change comes from chunk duration or model training, the same
`ah9u1bao` ep3 checkpoint should be read out on both 1.92s and 3.84s ACL data
under the same eval code path.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/ah9u1bao
- Diff:
  - model: evaluate the `ah9u1bao` ep3 `_best.pt` checkpoint without further training
  - eval data: use the existing 1.92s ACL extracted-paper JSONL `/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl`
  - audio length: eval fixed waveform length remains `1.92s`
  - task: eval-only ACL gs10000 readout, with tau sweep thresholds matching the current 3.84s training run
  - compute: one Aries GPU

## Expected metrics

This run is the short-window control for the sibling 3.84s readout. The main
comparison is ACL gs10000 raw recall and tau0.80 filtered recall/noise.

## Verdict

PENDING: update after the eval-only run finishes and compare against the 3.84s
readout and `dxwrgbln` inline ACL metrics.
