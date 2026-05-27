## Hypothesis

Evaluating the new_v3_random rank32/alpha64 speech LLM with deployment-like MaxSim filtering (tau=0.75) to see if random thinning degrades term adoption when retrieval is also filtered.

## Background / Motivation

Training run `y59wfnmp` completed on the new_v3_random dataset (halved expected negatives via random thinning), 8-GPU Aries, rank32/alpha64. This one-paper ACL eval checks the matched training/inference tau=0.75 setting on `2022.acl-long.110` at lm=1.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/sst_omni/runs/5e1iu7zo
- Diff: use HF export from training run `y59wfnmp` (new_v3_random, rank32/alpha64) and run one-paper eval with `RAG_SCORE_THRESHOLD=0.75`, `TARGET_PAPER=2022.acl-long.110`, `TARGET_LM=1`, timeline mode with full MaxSim windows.

## Expected metrics

Compare against new_v3 full tau=0.75 and new_v2 tau=0.75. Expect potentially lower TERM_ACC than full new_v3, with TERM_FCR depending on false-copy rate.

## Verdict

Pending one-paper SimulEval.
