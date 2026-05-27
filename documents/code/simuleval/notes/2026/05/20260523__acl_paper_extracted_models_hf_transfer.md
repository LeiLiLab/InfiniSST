# ACL Paper-Extracted Model Transfer To HF / PSC

## Hypothesis

Uploading the six paper-extracted main-result speech-LLM directories to private
HF model repos lets PSC pull them with resumable downloads and avoids relying on
one long rsync session for 396 GB of model shards.

## Background / Motivation

The paper-extracted ACL main result needs two model lines across three target
languages:

- no-TM-SFT origin-bsz4: zh/de/ja
- LLM-generated term-map RASST: zh/de/ja

Each local HF directory is about 66 GB and contains 15 safetensor shards.

## What changed vs baseline

This transfer workflow adds separate upload and PSC pull launchers.  The upload
launcher stages symlinks in a writable directory before calling
`hf upload-large-folder`.  The PSC pull launcher downloads each private model
repo into the exact directory layout expected by the eval launcher and validates
`config.json` plus 15 shards.

## Expected metrics

No eval metrics.  Success means all six PSC model directories exist and pass the
15-shard validation.

## Verdict

Planned.  Start after the PSC smoke eval succeeds.
