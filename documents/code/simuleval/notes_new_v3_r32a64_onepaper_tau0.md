## Hypothesis

Evaluating the new_v3 rank32/alpha64 speech LLM with unfiltered MaxSim retrieval (`tau=0.0`) should show the upper bound of term exposure for the corrected dense training data, but may increase false-copy rate.

## Background / Motivation

Training run `mazrc3id` completed on the full new_v3 dataset, which uses d9, tau=0.75, post-filter capped term maps with GT zh override. This one-paper ACL eval checks whether the model benefits from raw, gs1k, and gs10k glossary conditions on `2022.acl-long.110` at latency multiplier 1.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/sst_omni/runs/5e1iu7zo
- Diff: use HF export from training run `mazrc3id` (`new_v3`, rank32/alpha64) and run `run_acl_onepaper_lm_raw1k10k_taurus.sh` with `RAG_SCORE_THRESHOLD=0.0`, `TARGET_PAPER=2022.acl-long.110`, and `TARGET_LM=1`.

## Expected metrics

Compare TERM_ACC, TERM_FCR, BLEU, and StreamLAAL against earlier one-paper new_v2 and historical v4_ner outputs. Expect tau=0.0 to maximize adoption but possibly raise TERM_FCR relative to tau=0.75.

## Verdict

Pending one-paper SimulEval.
