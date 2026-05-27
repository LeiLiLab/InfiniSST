## Hypothesis

Speech LLM HF export is slowed by writing large merged shards directly to NFS. Staging the export on a local or near-node filesystem before syncing to `/mnt/gemini/data1` should reduce export wall time and avoid publishing partial HF directories.

## Background / Motivation

The German cap16 denoise short-tag run finished training quickly enough, but `swift export --to_hf` spent substantial time loading and writing 15 HF shards on Taurus. Future speech LLM runs need a reusable export path that does not repeatedly write intermediate shards to slow shared storage.

## What changed vs baseline

- Added `common/hf_export_staging.sh` with a shared `export_mcore_checkpoint_to_hf_staged` helper.
- Updated rank32/rank16 speech LLM training scripts and the standalone mcore-to-HF export script to use the helper.
- Added optional `HF_EXPORT_STAGE_ROOT` support: export to a staged local directory, validate, sync to a temporary final sibling, validate again, and atomically rename into place.
- Added optional `HF_EXPORT_LOCAL_CACHE_ROOT` and `HF_EXPORT_LOCAL_LATEST_LINK` support: keep a stable local HF cache and update a `latest-hf` symlink for fast local eval loading.
- Added optional `BASE_MODEL_STAGE_ROOT` support in the docker wrapper to cache the base HF model on host-local storage before mounting it into the container.
- Enabled local NVMe staging/cache defaults under `/mnt/data1/jiaxuanluo/...` in the German cap16 denoise short-tag 6-GPU launcher for future launches.

## Expected metrics

No model metric should change. The expected impact is operational: faster export and fewer partial HF directories when training completes.

## Verdict

Implemented and syntax-validated. The currently running export had already started with the old direct path, so this change applies to future speech LLM launches and standalone exports.
