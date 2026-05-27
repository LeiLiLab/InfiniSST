# En-De lm4 InfiniSST Batch vs Serial Diagnosis

## Hypothesis

The same-LM batched no-RAG En-De lm4 result differs from the serial result
because the batched driver is not prompt- and scheduler-equivalent to the
serial SimulEval driver.

## Background / Motivation

The no-retriever lm4 serial rerun produced higher BLEU/TERM_ACC than the new
same-LM batched run, despite using the same speech LLM checkpoint, language,
latency multiplier, and raw tagged ACL metric glossary.

## What changed vs baseline

This analysis compares the existing serial and batch artifacts directly:
manifests, eval TSVs, instance logs, runtime vLLM JSONL logs, source lists, and
the two driver code paths.

## Expected metrics

No new metric computation is introduced. The analysis reads the verified
standalone eval TSV artifacts linked by the two simuleval manifests.

## Verdict

The two runs are not equivalent. Source lists, references, source lengths, and
TERM_TOTAL match, but the no-RAG prompts differ: the batch path injects a
term-map instruction and `term_map:NONE` into every vLLM request, while the
serial no-RAG path does not. The batch path also decodes multiple active streams
with `max_num_seqs=5` and stochastic sampling, whereas the serial path uses
`max_num_seqs=1`. The current batch result should be treated as a diagnostic
run, not as a replacement for the serial main-result point.

Report:
`documents/code/simuleval/reports/20260525_de_lm4_infinisst_batch_vs_serial_diagnosis.md`
