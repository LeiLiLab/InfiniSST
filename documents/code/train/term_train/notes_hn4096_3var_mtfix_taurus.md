# HN depth rerun - `k=4096`, original 3variant data, current MFA/HN fixes

- **Family / data / task**: `sst_ood_hardneg` / `3variant_1m_mfa_mtfix` / `train`
- **Variant tag**: `hn4096_3var_mtfix`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hn4096_3var_mtfix_6gpu_taurus.sh`
- **Primary baseline run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/iaiyi1m8
- **Full GSV2 comparison run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/ly6sc2mr

## Hypothesis

Re-running the `iaiyi1m8` recipe with the current MFA-supervised MaxSim fixes
will isolate how much of `ly6sc2mr`'s gain comes from code correctness rather
than full GSV2 speaker-diverse data.

## Background / Motivation

The current training code makes MFA-supervised positives term-scoped, so
same-chunk different terms are neutral instead of positives scored with the
wrong MFA window. It also attaches every known GT term for a chunk to each row
and uses that set to exclude per-sample hard negatives that would otherwise be
false negatives.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/iaiyi1m8
- Diff:
  - train JSONL remains `/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_mfa.jsonl`
  - hparam `hard_neg_k_per_sample`: `4096` (unchanged)
  - hparam `tcm_loss_weight`: `0.0` (unchanged)
  - hparam `batch_size`: `6144` (unchanged)
  - compute changes from Aries 8GPU to Taurus 6GPU with `PER_GPU_BATCH=1024`
  - current code enables MFA term-scoped positives via `mfa_positive_scope=auto`
  - current code masks per-sample HN false negatives using all GT terms in the chunk

## Expected metrics

Compare `iaiyi1m8`, this rerun, and `ly6sc2mr` using:

```bash
python documents/code/general/wandb_tool.py --project qwen3_rag compare iaiyi1m8 <new_run_id> ly6sc2mr \
  --preset retriever_eval retriever_train --at-best-step --anchor-metric both
```

If this rerun closes most of the gap to `ly6sc2mr`, then the apparent full GSV2
gain was largely from the MFA/HN fix. If it stays near `iaiyi1m8`, the full GSV2
data and GigaSpeech cleanup are the likely drivers.

## Verdict

SUCCESS: the rerun shows that the current MFA/HN masking fixes alone do not
explain the full GSV2 result. The original 3variant data improves on dev and
some ACL filtered metrics under the fixed code, but it does not close the ACL
gs10k gap to the full GSV2 gsfix run.
