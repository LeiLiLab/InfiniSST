## Hypothesis

Reducing runtime retriever top-k from 10 to 5 may reduce unsupported glossary exposure for En-De tagged ACL RASST, improving BLEU while preserving most of the term accuracy gain.

## Background / Motivation

The latest cap16-denoise Speech LLM was trained to tolerate noisy retrieved term maps, but prior diagnostics suggested local term-map over-exposure remains a likely BLEU drag. This readout isolates the runtime retriever top-k setting under the same batch protocol used for the cap16-denoise top-k 10 comparison.

## What changed vs baseline

Only `rag_top_k` changes from 10 to 5. The model, HN1024 retriever checkpoint, tau 0.78, tagged ACL raw glossary, fixed chunk cache length (`max_cache_chunks=30`, `keep_cache_chunks=30`), prompt style (`given_chunks`), empty-map policy (`omit`), and max-new-token policy (`20*lm`) remain fixed.

## Expected metrics

For lm=1 and lm=4, BLEU should improve or at least stay near the top-k 10 cap16-denoise batch result. TERM_ACC may decrease mildly because fewer terms are exposed.

## Verdict

Pending. Metrics should be read from the validated `eval_results.tsv` and merged summary TSV linked by the event manifest.
