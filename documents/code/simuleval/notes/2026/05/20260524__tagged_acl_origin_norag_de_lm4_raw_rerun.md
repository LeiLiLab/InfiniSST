# Tagged ACL origin no-RAG de lm4 raw rerun

## Hypothesis

The current ACL tagged raw main-result figure has an abnormal En-De InfiniSST baseline point at latency multiplier 4. Re-running the no-term-map-SFT, no-RAG `gigaspeech-de-s_origin-bsz4` baseline under the raw tagged glossary scoring setup should clarify whether the plotted row is a bad cached value or a reproducible generation result.

## Background / Motivation

The plotted `new_main_result_tagged.pdf` currently uses fixed raw tagged ACL denominator rows. The En-De InfiniSST `lm=4` row is lower than expected relative to neighboring InfiniSST points and the older user-supplied baseline table. This run isolates exactly `lang=de`, `lm=4`, raw tagged ACL glossary, no runtime retrieval, and no term-map SFT.

## What changed vs baseline

Only the single En-De `lm=4` no-RAG baseline setting is re-run. The Speech LLM checkpoint remains `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_origin-bsz4`, RAG remains disabled, decoding remains the baseline sampling setup, and post-evaluation uses `offline_streamlaal_eval.py` with the fixed raw tagged ACL glossary.

## Expected metrics

The primary readouts are BLEU, StreamLAAL, StreamLAAL_CA, and TERM_ACC. For false-copy diagnostics in no-RAG mode, the post-eval uses `source_ref_negative_sentence` because no aligned runtime term map exists.

## Verdict

Success. The Aries rerun completed after rewriting `dev.source` to portable `/mnt/taurus` audio paths and recovering post-evaluation with the correct `spaCyEnv` Python. The fixed raw tagged ACL readout for En-De `lm=4`, raw glossary, no-TM-SFT/no-RAG InfiniSST is BLEU 33.3008, TERM_ACC 0.6909 (646/935), StreamLAAL 2824.4372 ms, and StreamLAAL_CA 4100.5704 ms. This is close to the older expected baseline region and should replace the abnormal plotted En-De `lm=4` point.
