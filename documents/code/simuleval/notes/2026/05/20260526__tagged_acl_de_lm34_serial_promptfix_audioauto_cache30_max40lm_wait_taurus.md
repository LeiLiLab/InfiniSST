## Hypothesis

For larger latency multipliers, reducing the vLLM prompt audio budget to the
effective cache window may improve quality by avoiding overlong prompt history.

## Background / Motivation

The user observed that `max_cache_chunks` affects quality and requested a rerun
of German tagged-ACL `lm=3` and `lm=4` with `VLLM_LIMIT_AUDIO=auto` while keeping
`max_cache_chunks=30` and `keep_cache_chunks=30`.

## What changed vs baseline

- Rerun tagged ACL raw En-De `lm=3,4`.
- Use the fixed serial MaxSim RAG agent.
- Use `VLLM_LIMIT_AUDIO=auto`, resolved by the agent to `max_cache_chunks`.
- Keep `max_new_tokens=40*lm`.
- Wait for Taurus GPU pairs to become idle before launching each job.

## Expected metrics

The rerun should produce verified `eval_results.tsv`, `instances.log`, and
`instances.strip_term.log` for both `lm=3` and `lm=4`. Runtime prompt validation
should confirm exactly one system prompt per `llm_input`.

## Verdict

Pending.
