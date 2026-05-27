# Medicine No-RAG Baseline On PSC

## Hypothesis

The restored ESO medicine no-RAG baseline can be completed on PSC Bridges-2 by
running one language per 2xV100 job.  Because RAG is disabled, the jobs should
only need the origin speech-LLM HF export, the five restored medicine samples,
and the StreamLAAL post-eval tooling.

## Background / Motivation

The medicine strict-term task uses streaming Qwen3-Omni no-RAG as the primary
baseline/filter.  A sanity run already exists for `zh lm=2`; this event fills
the remaining baseline grid:

- `zh`: latency multipliers `1 3 4`
- `de`: latency multipliers `1 2 3 4`
- `ja`: latency multipliers `1 2 3 4`

The paper-extracted ACL main-result workflow is paused while this baseline is
prioritized.

## What changed vs baseline

- Added a PSC wrapper that downloads one public HF origin model per language.
- Runs the existing batched medicine no-RAG launcher unchanged in task semantics:
  one `(language, latency_multiplier)` run contains all five medicine samples.
- Uses 2 V100 GPUs per language job with `tensor_parallel_size=2`.
- Adds PSC-safe vLLM overrides for model length, audio prompt count, cache
  window, and custom all-reduce behavior.
- Runs StreamLAAL term post-eval and miss export after each successful
  `(language, lm)` run.

Operational update: private HF upload hit the account storage limit during the
first attempt.  The active transfer mode is now temporary public HF model repos,
which the PSC launcher validates with `config.json` plus 15 safetensor shards
before eval.

## Expected metrics

Metrics are expected under the PSC output root as per-setting TSVs and one
language-job summary TSV.  For no-RAG baseline selection, use `TERM_ACC` and
the exported miss TSVs; `TERM_ADOPTION`, `REAL_TERM_ADOPT`, and `TERM_FCR` are
diagnostic only.

## Verdict

Reopened with user approval to use temporary public HF repos.  PSC code and
five restored medicine samples were transferred and validated.  The earlier
private/direct transfer blockers were:

- Private HF upload failed at commit time with the account private-storage
  limit.
- Direct `tar` over the current `psc` SSH alias goes through the Mac reverse
  tunnel and is too slow for 66G model directories.
- A temporary token HTTP server on Taurus was reachable locally, but PSC has no
  route to `taurus.cs.ucsb.edu:18080`; the server was stopped.

Public HF upload completed for all three origin models:

- `gavinlaw/infinisst-no-tmsft-origin-bsz4-zh`
- `gavinlaw/infinisst-no-tmsft-origin-bsz4-de`
- `gavinlaw/infinisst-no-tmsft-origin-bsz4-ja`

Initial PSC job `40968103` failed because the watcher passed
`TARGET_LMS="1 3 4"` through SSH without robust quoting; the watcher was
patched and resubmitted `zh lm=1 3 4` as PSC job `40968124` on 2xV100 with a
12-hour limit.  The target PSC jobs are now:

- `40968124`: `zh lm=1 3 4`, failed during the `lm=1` smoke because
  2xV100/TP=2 ran out of memory in vLLM engine initialization after loading
  14/15 shards.
- `40968239`: `de lm=1 2 3 4`, canceled after the `zh` 2xV100 OOM.
- `40968241`: `ja lm=1 2 3 4`, canceled after the `zh` 2xV100 OOM.
- `40968295`: `zh lm=1 3 4`, canceled before start because the queued
  command still had an arbitrary `GPU_MEMORY_UTILIZATION_OVERRIDE=0.75` and
  would have rerun `zh lm=4 sample=605000`.
- `40968296`: `de lm=1 2 3 4`, canceled before start for the same config
  hygiene reason.
- `40968297`: `ja lm=1 2 3 4`, canceled before start for the same config
  hygiene reason.
- `40968578`: `zh lm=1 3`, 5 samples, 4xV100/TP=4, cache `80.0/60.0`,
  failed after model load because the combined source file still pointed to
  Taurus/local audio paths under `/home/jiaxingxu/rag-sst/...`, which are absent
  on PSC.
- `40968579`: `zh lm=4`, samples `404 545006 596001 606`, 4xV100/TP=4,
  cache `80.0/60.0`, canceled after the same missing-audio-path issue was
  confirmed before it finished loading.
- `40968580`: `de lm=1 2 3 4`, 5 samples, 4xV100/TP=4, cache `80.0/60.0`,
  canceled after the same non-portable source paths were confirmed; its vLLM
  startup also hit a Triton cache `Disk quota exceeded` error.
- `40968581`: `ja lm=1 2 3 4`, 5 samples, 4xV100/TP=4, cache `80.0/60.0`,
  canceled before start on 2026-05-23 after the ja baseline was moved to Aries.
  The PSC local ja origin model directory
  `/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval/models/owaski/gigaspeech-ja-s_origin-bsz4`
  was deleted to recover space.

Two bad short submissions, `40968238` and `40968240`, were canceled after
inspection because they had corrupted `/jet/home/jluo7/4/...` paths and a
3-minute time limit.  The active jobs use a new `20260523T2200_psc_medicine_norag_tp4_*`
run stamp so failed 2xV100 outputs are not mixed with the rerun.

Slurm start estimates as of the 4xV100 resubmission:

- `40968295` `zh`: 2026-05-24 02:00:00 UTC
- `40968296` `de`: 2026-05-24 11:09:52 UTC
- `40968297` `ja`: 2026-05-24 11:09:52 UTC

`sbatch --test-only` alternatives were not better: 2xH100 estimated
2026-05-24 16:29:26 UTC and 2xL40S estimated 2026-05-25 06:09:52 UTC.  Keep
the 4xV100/TP=4 rerun unless the scheduler changes materially.

After user feedback that the actual runtime should be 4-5 hours, the active
jobs `40968578`-`40968581` were updated in place from 12 hours to 5 hours with
`scontrol update TimeLimit=05:00:00`.  The immediate start estimates were:

- `40968578` `zh_lm13`: 2026-05-23 23:16:38 UTC
- `40968579` `zh_lm4_no605000`: 2026-05-23 23:16:38 UTC
- `40968580` `de_lm1234`: 2026-05-24 04:00:00 UTC
- `40968581` `ja_lm1234`: 2026-05-24 04:00:00 UTC

Then `40968579` was shortened further from 5 hours to 1 hour because it only
runs `zh lm=4` on four samples and excludes the long `605000` sample already
completed on Taurus.  After the update, Slurm reported `Reason=None` and:

- `40968579` `zh_lm4_no605000`: 2026-05-23 23:16:38 UTC, ending 2026-05-24 00:16:38 UTC

On 2026-05-23, `40968581` was canceled before start because ja is now being run
on Aries.  Its PSC model cache was removed after verifying it was the only
matching `gigaspeech-ja-s_origin-bsz4` directory and occupied about 37G.

On 2026-05-24 UTC / 2026-05-23 EDT, the PSC zh jobs were checked live:

- `40968578` loaded the 4xV100 TP=4 vLLM model successfully, but failed before
  decoding with
  `soundfile.LibsndfileError: Error opening '/home/jiaxingxu/.../404_v2.wav'`.
  This proves the issue is not the V100 fit; the input source paths were not
  portable to PSC.
- `40968579` was still loading the same model and had the same `/home/jiaxingxu`
  source paths in its combined input file.  It was canceled at about 30 minutes
  elapsed to avoid wasting the allocation.
- `40968580` had the same invalid source paths and additionally logged
  `OSError: [Errno 122] Disk quota exceeded` from Triton's compile cache during
  vLLM startup.  It was canceled before reaching decoding.

User Taurus probe update: `MAX_CACHE_SECONDS=4.0` / `KEEP_CACHE_SECONDS=4.0`
materially changed quality on `zh lm=4 sample=605000`, so those values are not
a valid baseline default.  The PSC wrapper was synced to use:

- `VLLM_MAX_MODEL_LEN_OVERRIDE=8192`
- `VLLM_LIMIT_AUDIO_OVERRIDE=8`
- `VLLM_DISABLE_CUSTOM_ALL_REDUCE=1`
- `GPU_MEMORY_UTILIZATION_OVERRIDE=0.80`
- `MAX_CACHE_SECONDS_OVERRIDE=80.0`
- `KEEP_CACHE_SECONDS_OVERRIDE=60.0`

The existing Taurus `psc_limit8_keep80` result for `zh lm=4 sample=605000` was
copied to PSC for provenance instead of being rerun:

`/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval/outputs/medicine_norag_baseline_abbrev_restored/taurus_imports/20260523T2110_zh_lm4_sample605000/psc_limit8_keep80`

Fix and resubmit on 2026-05-24 UTC:

- Patched `prepare_medicine_one_talk_inputs.py` so it prefers the wav staged
  next to the selected `--eso-test-root` sample directory.  This prevents stale
  `metadata_v2.json` absolute paths from leaking into cross-cluster source
  lists.
- Added a pre-model-load source wav readability check in the batched medicine
  launcher, so missing PSC audio fails before vLLM loads.
- Patched the PSC wrapper to put HF, XDG, Triton, TorchInductor, vLLM, and
  Numba caches under `/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval/cache`.
- Staged the five medicine sample directories to
  `/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval/data/eso_medicine_abbrev_restored/test`.
  PSC prepare smoke confirmed the sample 404 source path is now
  `/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval/data/eso_medicine_abbrev_restored/test/sample_404_v2/404_v2.wav`.
- Resubmitted:
  - `40970222` `med_zh_lm13_fixaudio`, 4xV100-32, 5h, `zh lm=1 3`, all five samples.
  - `40970223` `med_zh_lm4_fixaudio`, 4xV100-32, 2h, `zh lm=4`, samples `404 545006 596001 606`; `605000` remains Taurus-imported.
  - `40970224` `med_de_lm1234_fixaudio`, 4xV100-32, 5h, `de lm=1 2 3 4`, all five samples.

At submission time PSC reported all three as pending with `Reason=Priority` and
`squeue --start` returned `N/A`.

Follow-up on 2026-05-24 UTC / 2026-05-23 EDT:

- `40970222` failed after the PSC source-path fix and before decoding.  The
  combined source list now contained readable `/ocean/projects/.../sample_*`
  wavs, so the stale `/home/jiaxingxu` path issue was fixed.
- `40970223` failed at the same vLLM initialization point.
- `40970224` was canceled by Codex after `de` was moved to Aries and to avoid
  wasting V100 allocation on the same environment issue.
- Root cause for `40970222`/`40970223`: PSC `vllm==0.13.0` imported the
  `_moe_C` namespace but did not register `_moe_C.topk_softmax`, causing
  Qwen3-Omni MoE routing to fail during vLLM profile/engine initialization.
- Added `documents/code/simuleval/tools/patch_vllm_moe_topk_softmax_fallback.py`.
  The patch keeps the native `_moe_C.topk_softmax` path when present and uses a
  torch `softmax + topk` fallback only when that op is missing.  It does not
  change `max_model_len`, audio limit, cache window, sampling, or latency
  settings.
- The PSC wrapper now applies this patch idempotently before model load.
- User canceled PSC `de`; `ja` and `de` are now being handled on Aries.
- Submitted zh-only smoke `40970552`: `zh lm=4 sample=545006`, 4xV100-32, 1h,
  `RUN_STAMP=20260524T0305_zh_smoke_topk`.  The goal is to verify vLLM engine
  initialization and one complete single-sample output before resubmitting the
  zh remaining grid.

Runtime strategy pivot on 2026-05-24 UTC:

- `40970552` was canceled because Slurm `--export` comma parsing left
  `GPU_PAIR=0` while `VLLM_TP_SIZE_OVERRIDE=4`; it was not used as a runtime
  verdict.
- Corrected TP=4 smoke `40970683` loaded all 15 model shards on 4xV100 but then
  failed during vLLM profile/kv-cache initialization with missing
  `_moe_C.moe_align_block_size`.  This showed that patching one missing MoE op
  at a time is not a robust path.
- PSC and Taurus vLLM native `.so` hashes match for `vllm==0.13.0`; the issue
  is the PSC host userspace not reliably registering `_moe_C` ops.  Earlier PSC
  container probing showed Ubuntu 22.04 Apptainer can load `vllm._moe_C` and
  register the MoE ops.
- The PSC medicine wrapper now defaults to `USE_APPTAINER=1` and disables the
  topk fallback patch by default (`ENABLE_VLLM_MOE_TOPK_PATCH=0`).
- Added no-model runtime smoke launcher
  `documents/code/simuleval/launchers/2026/05/20260524__psc_vllm_moe_ops_smoke.sh`.
  It checks `topk_softmax`, `moe_align_block_size`,
  `batched_moe_align_block_size`, and `moe_sum` without loading a model.
- Host smoke `40970818` was canceled.  Apptainer smoke `40970819` actually ran
  quickly but failed with `/var/spool/slurm/.../slurm_script: No such file or
  directory`; this was a wrapper bug, not a vLLM verdict.  Slurm executed a
  spool copy of the script, and that path was not visible inside Apptainer.
- Fixed both PSC wrappers to re-enter Apptainer via the stable repo-local
  script path instead of `$0`.
- Re-submitted Apptainer op smoke `40970887`.  It completed on V100 node `v002`
  in 2m30s with glibc 2.35, `torch_cuda_available=True`, GPU
  `Tesla V100-SXM2-32GB`, and all required MoE ops present:
  `topk_softmax`, `moe_align_block_size`, `batched_moe_align_block_size`, and
  `moe_sum`.
- Added `TARGET_SAMPLES` passthrough to the PSC wrapper so single-sample smokes
  do not accidentally run all five medicine samples.
- Submitted model-level zh smoke `40970896`: `zh lm=4 sample=545006`,
  4xV100-32/TP=4, 1h,
  `RUN_STAMP=20260524T0435_psc_medicine_zh_lm4_sample545006_appsmoke`.
  `squeue --start` estimated `2026-05-24T04:52:03Z` on node `v007`.
- Started detached local monitor PID `1543373` for `40970896`, polling every
  30 minutes and notifying through `~/bin/codex-notify` on state changes, first
  non-empty log, and terminal state.  Monitor state/logs:
  `/mnt/gemini/data1/jiaxuanluo/logs/psc_job_monitors/40970896_20260524T_monitor_v2`.
- User moved zh to Aries.  PSC model-level zh smoke `40970896` was canceled
  while still pending; `sacct` reported `CANCELLED by 98593`, elapsed `00:00:00`,
  so no PSC model output was produced.  The detached monitor PID `1543373` was
  stopped locally.
