## Hypothesis

Evaluating the new_v3_random rank32/alpha64 speech LLM (trained on randomly thinned term maps) with unfiltered MaxSim retrieval (tau=0.0) to compare against the full new_v3 and new_v2 baselines.

## Background / Motivation

Training run `y59wfnmp` completed on the new_v3_random dataset (halved expected negatives via random thinning), 8-GPU Aries, rank32/alpha64. This one-paper ACL eval checks performance on `2022.acl-long.110` at lm=1 with raw, gs1k, gs10k glossary conditions.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/sst_omni/runs/5e1iu7zo
- Diff: use HF export from training run `y59wfnmp` (new_v3_random, rank32/alpha64) and run one-paper eval with `RAG_SCORE_THRESHOLD=0.0`, `TARGET_PAPER=2022.acl-long.110`, `TARGET_LM=1`, timeline mode with full MaxSim windows.

## Expected metrics

Compare against new_v3 full (mazrc3id) and new_v2 (v2 r32). Random thinning may reduce TERM_ACC if model saw fewer term examples during training.

## Verdict

Pending one-paper SimulEval.
