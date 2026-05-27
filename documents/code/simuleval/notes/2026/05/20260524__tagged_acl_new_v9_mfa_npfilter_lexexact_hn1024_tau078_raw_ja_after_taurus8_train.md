## Hypothesis

The clean ja New V9 MFA/npfilter Speech LLM trained on Taurus 8 GPUs should be evaluated on tagged ACL raw glossary immediately after HF export, using the same HN1024 tau=0.78 retriever setting as the current main line.

## Background / Motivation

The earlier Aries 4-GPU ja training run was canceled because it was too slow. This watcher replaces the old after-train watcher and tracks the Taurus 8-GPU train run `7qqcy5oj`; once HF export is complete, it dispatches `lm=1,2,3,4` tagged ACL raw evaluations.

## What changed vs baseline

- Speech LLM target after export: `/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_ja_r32a64_tp2_taurus8/keep1.0_r32/v0-20260525-045920-hf`
- Train manifest: `20260524T2057__speech_llm_train__new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_ja_r32a64_tp2_taurus8`
- Retriever: HN1024 `lh1b88kw`, tau `0.78`, lookback `1.92s`, top-k `10`.
- Glossary: fixed raw tagged ACL glossary, `acl6060_tagged_gt_raw_min_norm2.json`.
- Eval driver: same-lm batch vLLM RAG evaluator with five talks per lm.
- Decode budget: fixed `max_new_tokens=80`.
- vLLM audio limit: `VLLM_LIMIT_AUDIO=128`.
- Assistant-only `<term>` tags are stripped before metrics.

## Expected metrics

The run should produce four rows for ja tagged ACL raw glossary, one for each `lm=1,2,3,4`, with BLEU, TERM_ACC, REAL_ADOPT, TERM_FCR, and StreamLAAL. These rows supersede the polluted old ja New V9 line and the canceled Aries 4-GPU train attempt.

## Verdict

Pending training completion, HF export, and all four same-lm batch evals.
