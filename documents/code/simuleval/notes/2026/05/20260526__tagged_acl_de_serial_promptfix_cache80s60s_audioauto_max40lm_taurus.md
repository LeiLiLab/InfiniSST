## Hypothesis

For serial de tagged-ACL RASST, cache length should follow the historical InfiniSST seconds-based policy rather than a fixed chunk count: `max_cache_seconds=80` and `keep_cache_seconds=60`, converted by the current latency chunk size.

## Background / Motivation

Earlier de promptfix serial runs used a fixed 30/30 cache policy. The user observed that small-lm runs may need more retained context, while larger-lm runs may benefit from less retained context. The old InfiniSST behavior was seconds-based, so the rerun uses `floor(80 / (0.96 * lm))` for max chunks and `floor(60 / (0.96 * lm))` for kept chunks.

## What changed vs baseline

- lm=1 defaults to `83/62` cache chunks.
- lm=3 defaults to `27/20` cache chunks.
- `VLLM_LIMIT_AUDIO=auto` is kept so prompt audio limit follows `max_cache_chunks`.
- Model, retriever, tau, glossary, promptfix agent, empty-map policy, and `max_new_tokens=40*lm` are unchanged.

## Expected metrics

lm=1 may recover quality from additional context. lm=3 tests whether the old seconds-based keep policy avoids the fixed-cache regression while preserving terminology gains.

## Verdict

Pending rerun.
