# HN256 step-1200 resume with ACL metric reset, Taurus 6GPU

## Hypothesis

Resuming HN256 from the frozen step-1200 checkpoint and resetting best trackers
lets the new ACL readout metrics choose fresh checkpoint files without being
blocked by the previous `eval_dev/recall@10_gs10000` and
`eval_acl6060/recall@10` best values.

## Background / Motivation

The prior resume run `lrdx14pm` tied `eval_acl6060/recall@10=0.9924` at steps
`1200` and `1280`, but only the strict-improvement best-secondary checkpoint at
step `1200` was persisted. The step-1280 latest checkpoint was later overwritten
by step `1360`. To avoid carrying forward the wrong best checkpoint semantics,
this run restarts from the frozen step-1200 checkpoint and explicitly changes
the checkpoint metrics.

At launch time, Taurus GPUs `4` and `5` were occupied by another user's process,
so this launcher defaults to clean GPUs `0,1,2,3,6,7` with an effective global
batch of `8190`. The HN256 objective, GradCache chunk size, data, eval sets, and
TCM-off settings are otherwise unchanged.

## What changed vs baseline

- Resume source run: `lrdx14pm`.
- Source checkpoint:
  `/mnt/gemini/home/jiaxuanluo/train_outputs/lrdx14pm_hn256_bestsec_acl6060r10_0p9924_step1200_tie1280_frozen_20260523.pt`
- Reset best trackers on resume:
  - `reset_best_on_resume=true`
  - `reset_scheduler=false`
- New checkpoint metrics:
  - primary: `eval_acl6060/top1`
  - secondary: `eval_acl6060/recall@10_gs10000`
- Hard-negative setting remains:
  - `hard_neg_k=0`
  - `hard_neg_k_per_sample=256`
  - `grad_cache_chunk_size=256`
  - TCM off
- Compute default:
  - GPU list: `0,1,2,3,6,7`
  - effective global batch: `8190 = 6 * 1365`
- Eval hygiene from the prior HN ablation remains:
  - top-100 per-sample eval logging disabled
  - TCM threshold sweep restricted to `0.75`
  - latest checkpoint overwritten after every eval

## Expected metrics

At the first eval after resume, the run should save new best checkpoint files
for both requested ACL metrics because prior best values are intentionally not
restored. Compare against `lrdx14pm` history around steps `1200` and `1280`:
`eval_acl6060/top1` was `0.9054` at step `1200` and `0.9224` at step `1280`;
`eval_acl6060/recall@10_gs10000` was `0.9218` at step `1200` and `0.9329` at
step `1280`.

## Verdict

RUNNING as W&B run `gsjheh6r`. Startup confirmed:

- resumed from the frozen step-1200 checkpoint;
- `reset_best_on_resume=true` caused both primary and secondary best values to
  be ignored;
- primary checkpoint metric is `eval_acl6060/top1`;
- secondary checkpoint metric is `eval_acl6060/recall@10_gs10000`.

The process is detached on Taurus with launcher PID `129472`, torchrun PID
`129492`, and log
`/mnt/gemini/data1/jiaxuanluo/logs/hn256_step1200_aclmetric_reset_taurus6_20260523T0335Z.direct.log`.
