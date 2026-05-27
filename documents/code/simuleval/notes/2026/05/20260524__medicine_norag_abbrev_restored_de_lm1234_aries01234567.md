# Medicine no-RAG baseline, restored ESO, de lm1/2/3/4 on aries

## Hypothesis

German streaming no-RAG Qwen3-Omni baselines at `lm=1,2,3,4` provide the
remaining medicine hard-term baseline surface.

## Background / Motivation

Run the restored ESO medicine no-RAG baseline for:

```text
lang=de
lm=1 on aries GPUs 2,3
lm=2 on aries GPUs 4,5
lm=3 on aries GPUs 6,7
lm=4 on aries GPUs 0,1 after ja/lm1 releases them
samples=404 545006 596001 605000 606
```

Input ESO test root:

```text
/mnt/taurus/home/jiaxingxu/rag-sst/eso-dataset/outputs_v2_abbrev_exact_match_abbrev_restored/test
```

## What changed vs baseline

This reuses the restored ESO batched no-RAG launcher, restricted to `de` and one
latency multiplier per child process. `lm=1,2,3` are launched immediately on
currently free GPU pairs through direct detached execution on aries. `lm=4`
waits for `ja/lm1` to finish, then uses GPUs `0,1`.

## Expected metrics

Primary generation outputs are `instances.log`, `hypotheses.tsv`, and
`timing.tsv` under each per-lm output root. Hard-term StreamLAAL rescoring uses
the manually checked hard glossary after each generation finishes.

## Verdict

`lm=1,2,3` submitted immediately through direct aries wrapper PID `524937`;
child PIDs are `524944`, `524947`, and `524951`. `lm=4` is queued behind the
still-running `ja/lm=1` aries wrapper through waiter PID `958327`.

Update 2026-05-24 03:45 UTC: the first direct `lm=1,2,3` launch failed during
vLLM engine initialization because Triton tried to write cache under the full
aries filesystem `/mnt/data7/jiaxuanluo/.cache/triton`. Those failed outputs
are preserved. Rerun submitted with explicit `TRITON_CACHE_DIR`, `TMPDIR`,
`XDG_CACHE_HOME`, `TORCHINDUCTOR_CACHE_DIR`, and `CUDA_CACHE_PATH` under
`/mnt/gemini/data1/jiaxuanluo/cache/medicine_norag_de_rerun_tritoncache_20260524T0345`.
Rerun wrapper PID is `586601`, inner wrapper PID is `586603`, and child PIDs
are `586606`, `586613`, and `586624`. The old `lm4` waiter/post-eval watcher
were cancelled and replaced by waiter PID `1312782` and post-eval watcher PID
`1312783`.

Update 2026-05-24 03:52 UTC: the first rerun fixed the Triton cache location
but failed immediately because the new `TMPDIR` path under `/mnt/gemini/data1`
made vLLM/ZMQ IPC socket paths longer than 107 characters. That rerun is also
preserved. A second rerun is now submitted with short `TMPDIR` values under
`/dev/shm/jxde{1,2,3,4}` while keeping Triton/torch/cuda caches on
`/mnt/gemini/data1`. New wrapper PID is `605662`, inner wrapper PID is
`605664`, child PIDs are `605667`, `605673`, and `605684`. The short-TMPDIR
`lm4` waiter PID is `1344487`; post-eval watcher PID is `1344488`.

Update 2026-05-24 04:03 UTC: taurus had idle GPUs, so the aries `lm4` waiter
was cancelled and `de/lm=4` was submitted directly on taurus GPUs `4,5`.
Output root:

```text
/mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260524_de_rerun_shorttmp_lm4_taurus45
```

The generation wrapper PID is `1423748`, child PID is `1423753`. The hard-term
post-eval watcher was replaced with a mixed watcher for aries `lm=1,2,3` plus
taurus `lm=4`; mixed watcher PID is `1424347`, completion notifier PID is
`1426472`.
