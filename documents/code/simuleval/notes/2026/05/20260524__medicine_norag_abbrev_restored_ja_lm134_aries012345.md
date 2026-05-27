# Medicine no-RAG baseline, restored ESO, ja lm1/3/4 on aries

## Hypothesis

Japanese streaming no-RAG Qwen3-Omni baselines at `lm=1,3,4` provide the
remaining latency settings for hard medicine term analysis.

## Background / Motivation

Run the restored ESO medicine no-RAG baseline for:

```text
lang=ja
lm=1 on aries GPUs 0,1
lm=3 on aries GPUs 2,3
lm=4 on aries GPUs 4,5
samples=404 545006 596001 605000 606
```

Input ESO test root:

```text
/mnt/taurus/home/jiaxingxu/rag-sst/eso-dataset/outputs_v2_abbrev_exact_match_abbrev_restored/test
```

## What changed vs baseline

This reuses the restored ESO batched no-RAG launcher, restricted to `ja` and one
latency multiplier per child process. A single Slurm step enters the aries hold
allocation, then starts three child launchers concurrently. Each child process
owns a disjoint physical GPU pair.

## Expected metrics

Primary generation outputs are `instances.log`, `hypotheses.tsv`, and
`timing.tsv` under each per-lm output root. Hard-term StreamLAAL rescoring can
use the manually checked hard glossary after generation.

## Verdict

Running.
