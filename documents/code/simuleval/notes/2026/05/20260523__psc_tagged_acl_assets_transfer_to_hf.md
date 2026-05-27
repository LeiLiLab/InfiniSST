# PSC Tagged ACL Eval Assets Transfer To HF

## Hypothesis

A small Hugging Face dataset repo with resumable upload/download is a safer PSC
asset transfer path than direct large `tar|ssh` copies.

## Background / Motivation

The PSC eval needs the already-uploaded HF speech LLM plus several non-model
assets: the `lh1b88kw` retriever checkpoint, the packed runtime environment,
ACL6060 data/audio, and `mwerSegmenter`.  PSC compute nodes can pull from HF
quickly, while login-node installs/downloads are fragile.

## What changed vs baseline

Create a reusable asset bundle:

- HF dataset repo: `gavinlaw/infinisst-psc-tagged-acl-newv5-r32-assets`
- retriever checkpoint: `lh1b88kw_best_eval_acl6060_recallat10.pt`
- env tar: `spaCyEnv_20260518.tar.gz`
- data tar: `acl6060_20260523.tar.gz`
- tool tar: `mwerSegmenter_20260523.tar.gz`

## Expected metrics

No model metric is expected.  Success means PSC has all files with byte sizes
matching the transfer manifest, and the env/data/tool archives are extracted to
the expected PSC workspace.

## Verdict

Success.  The HF dataset upload completed successfully:
`gavinlaw/infinisst-psc-tagged-acl-newv5-r32-assets`.

Uploaded asset manifest:

- `checkpoints/lh1b88kw_best_eval_acl6060_recallat10.pt`: `4278377821` bytes
- `data/acl6060_20260523.tar.gz`: `950017764` bytes
- `envs/spaCyEnv_20260518.tar.gz`: `6539540160` bytes
- `tools/mwerSegmenter_20260523.tar.gz`: `1123907` bytes

PSC pull/extract Slurm job `40963196` completed with exit code `0:0` in
`00:24:46`.  It pulled all assets, extracted env/data/tool directories, and
passed import validation for `torch`, `transformers`, `vllm`, `simuleval`,
`yaml`, `soundfile`, and `wandb`.

Downstream eval job `40963197` was submitted after the pull job and is queued on
`GPU-shared` / `gpu:v100-32:2`.
