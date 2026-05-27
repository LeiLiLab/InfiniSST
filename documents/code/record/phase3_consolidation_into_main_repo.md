# Phase 3 Consolidation into Main Repo

- **Date**: 2026-04-18
- **Trigger**: Handoff doc `/mnt/taurus/home/jiaxuanluo/InfiniSST_logs/phase3_handoff.md`.
  User instruction: fully copy the `udc` worktree's phase3 files into the main repo and stop using worktrees.

## Source

`/mnt/taurus/home/jiaxuanluo/.cursor/worktrees/InfiniSST__SSH__taurus.cs.ucsb.edu_/udc/documents/code/train/term_train/`

## Destination

`/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/`

## Backup of Previous State

Pre-consolidation snapshot of the destination was saved to:

`/mnt/taurus/home/jiaxuanluo/InfiniSST_logs/pre_phase3_consolidation_backup_20260418_032835/`

(excludes `__pycache__`)

## Operation

```
rsync -av --exclude='__pycache__' \
  <udc>/documents/code/train/term_train/ \
  <main>/documents/code/train/term_train/
```

Post-copy `diff -rq` between source and destination (excluding `__pycache__`
and the pre-existing `wiki_hard_neg_disjoint.json` that only lives in main)
returned empty.

## Files added to main repo (previously missing)

New top-level:
- `documents/code/train/term_train/run_phase3_eval_with_head.sh`

New under `documents/code/train/term_train/confidence_head/`:
- `confidence_head_dataset.py`
- `confidence_head_eval_wrapper.py`
- `confidence_head_metrics.py`
- `confidence_head_modeling.py`
- `modelA_vs_B_dev_comparison.md`
- `run_train_confidence_head.sh`
- `sbatch_train_confhead_a_aries.sh`
- `sbatch_train_confhead_b_aries.sh`
- `train_confidence_head.py`

## Files updated in main repo

- `documents/code/train/term_train/qwen3_glossary_neg_train.py`
  - Overwritten with the `udc` version that contains the Phase 3 confidence-head
    integration (`--confidence_head_ckpt`, `--confidence_head_topk`,
    `--confidence_head_softmax_temp`, `--eval_only` flags; head-aware
    `run_sample_eval` summary).

## Files untouched in main repo

- `documents/code/train/term_train/confidence_head/cache_confidence_head_data.py`
- `documents/code/train/term_train/confidence_head/run_cache_confidence_head_data.sh`
- `documents/code/train/term_train/wiki_hard_neg_disjoint.json`
- All other siblings that already matched byte-for-byte.

## Post-copy fixes (stale worktree references)

Only two files referenced the `udc` worktree path. Both were updated to point
at the main repo:

- `documents/code/train/term_train/confidence_head/sbatch_train_confhead_a_aries.sh`
- `documents/code/train/term_train/confidence_head/sbatch_train_confhead_b_aries.sh`

Before:

```
WORKTREE_ROOT="/mnt/taurus/home/jiaxuanluo/.cursor/worktrees/InfiniSST__SSH__taurus.cs.ucsb.edu_/udc"
SCRIPT_PATH="${WORKTREE_ROOT}/documents/code/train/term_train/confidence_head/train_confidence_head.py"
```

After:

```
REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"
SCRIPT_PATH="${REPO_ROOT}/documents/code/train/term_train/confidence_head/train_confidence_head.py"
```

`grep -rn "WORKTREE_ROOT\|\.cursor/worktrees" documents/code/train/term_train/`
now returns nothing.

## Smoke verification

- `python -m py_compile` passes for every Python file in `term_train/` and
  `term_train/confidence_head/`.
- `bash -n` passes for `run_phase1_eval_only.sh`, `run_phase3_eval_with_head.sh`,
  `run_cache_confidence_head_data.sh`, `run_train_confidence_head.sh`,
  `sbatch_train_confhead_{a,b}_aries.sh`.

## Follow-up fix applied to the default ckpt paths

`run_phase3_eval_with_head.sh` had two defaults that were non-portable per the
cross-node path rule (pitfall #7 in `phase3_handoff.md` section 5):

Before:
```
DEFAULT_RETRIEVER_CKPT="/mnt/taurus/data/jiaxuanluo/train_outputs/q3rag_scale_...pt"   # node-local prefix, fails on Aries
DEFAULT_HEAD_CKPT="/mnt/data4/jiaxuanluo/confidence_head_runs/modelB_v1/head_best.pt"   # omits partition name
```

After:
```
DEFAULT_RETRIEVER_CKPT="/mnt/aries/data4/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.1_maxsim_mfa_final_C_best_acl6060_gs10000.pt"
DEFAULT_HEAD_CKPT="/mnt/aries/data4/jiaxuanluo/confidence_head_runs/modelB_v1/head_best.pt"
```

Paths were verified to resolve on the Aries node via a smoke `srun -p aries`
check before submission.

## Phase 3 eval resubmission

Submitted after consolidation + path fix:

```
cd /home/jiaxuanluo && sbatch -p aries --gres=gpu:1 --mem=64G --cpus-per-task=8 --time=02:00:00 \
  --wrap="bash /mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_phase3_eval_with_head.sh \
  /mnt/aries/data4/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.1_maxsim_mfa_final_C_best_acl6060_gs10000.pt \
  /mnt/aries/data4/jiaxuanluo/confidence_head_runs/modelB_v1/head_best.pt"
```

- Job ID: 43688
- Phase 3 log: `/mnt/taurus/home/jiaxuanluo/InfiniSST_logs/phase3_eval/phase3_eval_20260418_033217.log`
- SLURM stdout (from Taurus view): `/mnt/aries/home/jiaxuanluo/slurm-43688.out`

First-minute progress confirmed:
- `[RESUME]` loaded retriever from Aries-local ckpt (epoch=3 step=1419).
- `[CONF_HEAD]` loaded Model B (variant=B, hidden=[128,64], softmax_temp=0.0700, dev metric=0.9953).
- `[EVAL_ONLY]` mode entered; train loop skipped.
- Dev eval started on 4646 dev samples + 3623 ACL dev samples.
- gs1000 dev bank correctly skipped because GT bank size (1852) already exceeds 1000 (expected behavior).

## Operating rule going forward

No further edits will be made to files under
`/mnt/taurus/home/jiaxuanluo/.cursor/worktrees/` for this project. All phase3
work continues in the main repo `/mnt/taurus/home/jiaxuanluo/InfiniSST/`.
