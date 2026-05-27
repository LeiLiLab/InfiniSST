# RAG compute RTF figure

## Hypothesis

Because timeline RAG retrieval runs once per vLLM generation step on a separate
GPU, its compute contribution should be a small fraction of the vLLM cadence
interval and should shrink as the latency multiplier increases.

## Background / Motivation

The streaming agent invokes vLLM every `0.96s * LM`.  For RAG, each retrieval
encodes the current vLLM audio span plus a fixed 1.92s look-back window, so the
retriever input spans are 2.88s, 3.84s, 4.80s, and 5.76s for LM 1--4.

The figure uses the verified En-Zh medicine hardraw raw-bank runtime traces
because all four LM settings include timed `rag_window` / `vllm_timeline`
records in their runtime JSONL files.

## What changed vs baseline

Added a paper plotting script:

```text
documents/code/train/term_train/src/plot_rag_compute_rtf.py
```

The script reads the canonical fixed-raw summary TSV, follows each
`eval_results.tsv` path to the corresponding runtime JSONL, computes per-call
retriever timing statistics, and plots:

```text
RAG compute RTF = retriever call time / (0.96s * LM)
```

It writes a provenance TSV and copies the regenerated figure into the paper
LaTeX figure directory.

## Expected metrics

The median retriever compute RTF should be under 4% for LM=1 and decrease as LM
increases.  Mean RTF is also recorded in the TSV to expose long-tail runtime
variation.

## Verdict

Success.  The generated TSV records timed runtime JSONL evidence for LM 1--4,
and the paper now references `latex/figures/rag_compute_rtf.pdf` instead of the
old placeholder efficiency panel.
