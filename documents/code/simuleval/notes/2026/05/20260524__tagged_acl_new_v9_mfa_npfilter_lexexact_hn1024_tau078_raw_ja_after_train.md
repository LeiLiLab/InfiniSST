## Hypothesis

The clean ja New V9 MFA/npfilter Speech LLM should be evaluated on tagged ACL raw glossary immediately after HF export, using the same HN1024 tau=0.78 retriever setting as the current main line.

## Background / Motivation

The ja model is training on aries GPUs 0,1,2,5 as W&B run `332v0v6n`.  This event waits for the HF export from that training run and then launches `lm=1,2,3,4` tagged ACL raw evaluations.  Each latency multiplier is run as a same-lm batch over the five ACL talks so one vLLM instance handles the five samples in parallel.

## What changed vs baseline

- Speech LLM: `/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_ja_r32a64_tp2_aries4/keep1.0_r32/v0-20260525-020815-hf`
- Retriever: HN1024 `lh1b88kw`, tau `0.78`, lookback `1.92s`, top-k `10`.
- Glossary: fixed raw tagged ACL glossary, `acl6060_tagged_gt_raw_min_norm2.json`.
- Eval driver: same-lm batch vLLM RAG evaluator with five talks per lm.
- Decode budget: fixed `max_new_tokens=80`.
- vLLM audio limit: `VLLM_LIMIT_AUDIO=128`.
- Assistant-only `<term>` tags are stripped before metrics.

## Expected metrics

The run should produce four rows for ja tagged ACL raw glossary, one for each `lm=1,2,3,4`, with BLEU, TERM_ACC, REAL_ADOPT, TERM_FCR, and StreamLAAL.  These rows supersede the polluted old ja New V9 line and should be used as the clean ja main-result candidate.

## Verdict

Pending training completion, HF export, and all four same-lm batch evals.
