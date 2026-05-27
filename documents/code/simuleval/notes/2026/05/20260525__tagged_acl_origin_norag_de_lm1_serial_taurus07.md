## Hypothesis

A serial InfiniSST/no-RAG En-De `lm=1` rerun on Taurus should clarify whether
the lower same-LM batch BLEU is a batch-driver effect rather than an intrinsic
baseline weakness.

## Background / Motivation

The same-LM batch rerun with `norag_prompt_policy=serial_compat` produced
`lm=1` BLEU 26.5639. The older main-result table carries an En-De `lm=1`
InfiniSST reference BLEU 27.4672, but that row is user-supplied rather than a
verified filesystem artifact. A verified serial run using the original
InfiniSST no-RAG agent path is needed for a fair timing and metric check.

## What changed vs baseline

- Language: En-De.
- Latency multiplier: `lm=1`.
- Model: `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_origin-bsz4`.
- Runtime mode: serial SimulEval no-RAG baseline via
  `bypass_simuleval_rank32_iter_0000452_hf_baseline_no_rag_sweep.sh`.
- Taurus GPU pair: `0,7`, leaving the active six-GPU training job on GPUs
  `1,2,3,4,5,6` untouched.
- Decode cap follows the verified serial lm4 launcher default:
  `MAX_NEW_TOKENS=40`.
- Tagged raw glossary is used only for post-eval denominator/bookkeeping.

## Expected metrics

The run should produce one `instances.log` with five ACL talks and a post-eval
`eval_results.tsv` with `TERM_TOTAL=935`. Runtime elapsed seconds are recorded
in `runtime_seconds.txt` so we can estimate the cost of replacing batch readouts
with serial runs.

## Verdict

Pending.
