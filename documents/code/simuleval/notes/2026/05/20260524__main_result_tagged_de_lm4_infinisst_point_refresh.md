# Main Result Tagged ACL De Lm4 InfiniSST Point Refresh

## Hypothesis

The En-De `lm=4` InfiniSST point in `new_main_result_tagged.pdf` should use the verified no-TM-SFT/no-RAG rerun rather than the abnormal older cached value.

## Background / Motivation

The previous canonical ACL tagged raw figure used an abnormal En-De `lm=4` InfiniSST row. A targeted rerun completed as W&B `3upoqej5` with fixed raw tagged ACL scoring and produced BLEU 33.3008, TERM_ACC 0.6909, StreamLAAL 2824.4372 ms, and StreamLAAL_CA 4100.5704 ms.

## What changed vs baseline

The figure-building script now treats the En-De `lm=4` InfiniSST row as a verified `eval_results.tsv` override sourced from the rerun artifact. The canonical TSV and `new_main_result_tagged.{pdf,png}` were regenerated. Medicine figures were not copied from the temporary regeneration directory during this refresh.

## Expected metrics

The canonical TSV row for `(acl_tagged_raw, InfiniSST, de, 4)` should report BLEU 33.3008, TERM_ACC 0.6909, StreamLAAL 2824.4372, StreamLAAL_CA 4100.5704, and `wandb_run_id=3upoqej5`.

## Verdict

Success. The tagged ACL main-result figure was regenerated with the verified En-De `lm=4` InfiniSST rerun point. The visual En-De InfiniSST curve now ends near 2824 ms / 33.3 BLEU / 69.1% TERM_ACC, matching the rerun artifact.
