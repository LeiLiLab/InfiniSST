# HN1024 GSV2full gsdedup ep3 best ACL 3.84s readout

- **Family / data / task**: `sst_ood_hardneg` / `acl6060_ctx384_ep3best_eval` / `eval`
- **Variant tag**: `acl_ctx384_gsdedup_ep3best`
- **Launcher**: `documents/code/train/term_train/run_eval_mfa_smallest_dense_hn1024_gsv2full_gsdedup_ep3best_acl_ctx3p84_1gpu_aries.sh`
- **Checkpoint**: `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_gsdedup_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep3_bs12k_smallest_dense_normAGGR_8gpu_aries_best.pt`

## Hypothesis

The 1.92s GSV2-dedup ep3 best checkpoint may already be competitive when
evaluated on the same 3.84s ACL chunks used by the current context-expansion
run. This isolates whether current ACL changes come from the wider eval window
or from retraining on 3.84s expanded positives.

## Background / Motivation

The current `dxwrgbln` training run logs inline ACL6060 metrics on regenerated
3.84s ACL chunks. Existing older ACL readouts for the gsdedup line used 1.92s
ACL chunks and a later conv5 checkpoint, so they are not a fair comparison to
the current run at roughly matched training scale.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/ah9u1bao
- Diff:
  - model: evaluate the `ah9u1bao` ep3 `_best.pt` checkpoint without further training
  - eval data: old ACL 1.92s extracted-paper JSONL -> `/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary_ctx3p84/acl6060_dev_dataset.jsonl`
  - audio length: eval fixed waveform length `1.92s` -> `3.84s`
  - task: eval-only ACL gs10000 readout, with tau sweep thresholds matching the current 3.84s training run
  - compute: one Aries GPU

## Expected metrics

The readout should be compared against current `dxwrgbln` inline ACL metrics at
steps 240/320 and later. Useful evidence is whether the ep3 1.92s-trained
checkpoint on 3.84s ACL chunks reaches similar `eval_acl6060/recall@10_gs10000`
and tau0.80 filtered recall.

## Verdict

PENDING: update after the eval-only run finishes and compare ACL gs10000 raw and
tau-filtered metrics against `dxwrgbln`.
