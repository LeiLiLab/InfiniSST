## Hypothesis

The clean De New V9 Speech LLM with HN1024 tau=0.78 should provide the medicine-domain de readout under the same five-sample same-LM batch protocol, using `max_new_tokens=80` and `VLLM_LIMIT_AUDIO=128`.

## Background / Motivation

The tagged ACL de main run uses the clean `mfa_npfilter_srcunion_lexexact` Speech LLM checkpoint. This run applies that same de model to the medicine hardraw five-sample readout with one vLLM process per latency multiplier.

## What changed vs baseline

- Language: `de`
- Domain: medicine hardraw five-sample batch
- Speech LLM: clean New V9 `mfa_npfilter_srcunion_lexexact` de model
- Retriever: HN1024 checkpoint, tau=0.78, top-k=10, timeline lookback=1.92s
- Batch shape: lm=1,2,3,4; each lm uses one TP=2 vLLM process and evaluates all five samples together
- Decode cap: `max_new_tokens=80`
- vLLM audio limit: `VLLM_LIMIT_AUDIO=128`

## Expected metrics

Expected to be comparable to the previous de medicine hardraw run but with the clean de model and the fixed batch evaluator settings. The run should produce one `eval_results.tsv` per lm and a final summary TSV/MD after post-processing.

## Verdict

Running.
