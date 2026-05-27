## Hypothesis

Evaluating the new_v3 rank32/alpha64 speech LLM with deployment-like MaxSim filtering (`tau=0.75`) should reduce false-copy noise while preserving the gains from corrected dense training term maps.

## Background / Motivation

Training run `mazrc3id` completed on the full new_v3 dataset, which itself was built from tau=0.75 retriever outputs and a hard 20-term cap. This eval tests the matched training/inference filter setting on ACL paper `2022.acl-long.110` at latency multiplier 1.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/sst_omni/runs/5e1iu7zo
- Diff: use HF export from training run `mazrc3id` (`new_v3`, rank32/alpha64) and run `run_acl_onepaper_lm_raw1k10k_taurus.sh` with `RAG_SCORE_THRESHOLD=0.75`, `TARGET_PAPER=2022.acl-long.110`, and `TARGET_LM=1`.

## Expected metrics

Compare TERM_ACC, TERM_FCR, BLEU, and StreamLAAL against tau=0.0, earlier one-paper new_v2 runs, and the historical v4_ner outputs. Expect lower TERM_FCR than tau=0.0, with TERM_ACC depending on whether filtered retrieval still supplies enough useful terms.

## Verdict

Pending one-paper SimulEval.
