## Hypothesis

vLLM model startup is dominated by repeated reads of 15 safetensors shards from `/mnt/gemini/data1`. Staging the two active German SLM HF directories onto each host's local disk should reduce repeated model-load latency for subsequent serial or batch eval runs.

## Background / Motivation

The de tagged-ACL lm1/lm4 jobs spent roughly 20 minutes loading model shards. Both candidate HF model directories are about 66G and live on the Gemini NFS mount. Taurus has sufficient local space under `/mnt/data1`; Aries has sufficient local space under `/mnt/data3`.

## What changed vs baseline

This maintenance event copies the following model directories to host-local cache paths:

- cap16 exact-boundary SLM
- cap16 denoise ttag SLM

The copy is done with `rsync`, a fixed bandwidth limit, logs, pid files, and a `.stage_complete` marker. No eval metrics are produced by this event.

## Expected metrics

No task metrics. Expected operational metric is lower vLLM model-load time when launchers use the local cached `MODEL_NAME` path.

## Verdict

Pending. Verify by checking the stage logs, the `.stage_complete` marker, 15 safetensors shards per copied HF directory, and a later vLLM load-time comparison.
