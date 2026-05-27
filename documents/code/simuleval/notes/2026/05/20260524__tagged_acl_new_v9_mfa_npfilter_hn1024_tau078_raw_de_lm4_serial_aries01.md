# Tagged ACL clean New V9 HN1024 de lm4 serial rerun

## Hypothesis

The same-lm batch evaluator may be affecting the En-De tagged ACL RASST BLEU
readout. Re-running `lm=4` through the older serial SimulEval path with the
current clean German Speech LLM and HN1024 retriever should isolate the batch
driver from the model/retriever setting.

## Background / Motivation

The verified no-RAG En-De `lm=4` rerun used the old SimulEval/no-RAG launcher
and recovered BLEU 33.3008. The current clean-SLM + HN1024 En-De rows were
produced by the same-lm batch-vLLM evaluator, so the low BLEU could be a
driver artifact rather than an inherent RASST degradation.

## What changed vs baseline

This run keeps the tagged ACL raw denominator, German language, `lm=4`,
HN1024 retriever, tau `0.78`, timeline lookback `1.92s`, and the current clean
MFA/source-filtered New V9 German Speech LLM. It changes the evaluation driver:
the run uses `20260520__tagged_acl_origin_bsz4_tau073_sweep_taurus45269.sh`
and `eval_density_unified.sh`, not `20260524__batched_vllm_rag_eval.sh`.
Decoding uses `max_new_tokens=40`, matching the no-RAG rerun strategy.

## Expected metrics

If the batch evaluator caused the BLEU drop, this serial run should move closer
to the no-RAG `lm=4` BLEU region while retaining high TERM_ACC from retrieval.
If the result stays close to the same-lm batch row, the lower BLEU is likely a
model/RAG behavior rather than the batch implementation alone.

## Verdict

Success. The non-batch serial SimulEval readout completed on Aries GPU `0,1`
with W&B run `uvf3r9if`. The initial post-run TSV reported StreamLAAL
`1611.4236`, but that value was invalid: German word-level `<term>` stripping
left prediction tokens and delay entries misaligned in `instances.strip_term.log`.
After fixing strip to preserve word-level timing alignment and to remove a
malformed `<term` prefix, the corrected raw tagged ACL metrics are BLEU
`30.0309`, TERM_ACC `0.8460` (`791/935`), StreamLAAL `2657.5659`, and
StreamLAAL_CA `4358.3993`.

The corrected result still does not support batch-vLLM as the sole cause of the
lower En-De RASST BLEU: the serial result is also below the no-RAG `lm=4` rerun
BLEU `33.3008`.

Post-run caveat: generation/eval/W&B logging completed, but the wrapper failed
afterwards while writing the small summary file because bare `python` was not in
the Aries login PATH. The launcher was patched to use the explicit spaCyEnv
Python, and the summary TSV/MD were recovered from the completed
`eval_results.tsv`.

Correction note: `eval_results.invalid_wordstrip_20260524T2359.tsv` preserves
the invalid pre-fix values for audit. The current `eval_results.tsv` and summary
files use the corrected word-aligned strip output.
