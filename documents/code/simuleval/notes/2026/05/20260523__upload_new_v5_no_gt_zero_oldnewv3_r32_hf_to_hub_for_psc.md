# Upload New V5 no-GT-zero old-new_v3 R32 HF Model To Hub For PSC

## Hypothesis

Uploading the complete HF export to a private Hugging Face model repo gives a
more resumable and reusable transfer path for PSC than direct `tar|ssh` copies.

## Background / Motivation

PSC can schedule short V100 jobs under `cis260009p`, but large taurus-to-PSC
file copies are fragile because the PSC login environment does not provide
remote `rsync`.  The tagged ACL gs1k/gs10k eval needs the 66G HF export for the
current zh `lm=2` model.

## What changed vs baseline

Use Hugging Face Hub as the model distribution layer:

- local model: `/mnt/aries/data6/jiaxuanluo/slm/speech_llm_new_v5_no_gt_zero_llm_variant_aug_oldnewv3_zh_r32a64_tp2_aries2/keep1.0_r32/v0-20260523-050346-hf`
- writable upload stage: `/mnt/aries/data6/jiaxuanluo/hf_upload_staging/new_v5_no_gt_zero_oldnewv3_r32_hf`
- private repo: `gavinlaw/infinisst-new-v5-no-gt-zero-oldnewv3-r32a64-keep1p0-r32-zh`
- upload command: `hf upload-large-folder`

## Expected metrics

No model metric is expected from this maintenance event.  Success means the Hub
repo contains all 15 safetensor shards and can be pulled from PSC with
the repo-local resumable curl downloader.

## Verdict

Success.  The private HF repo was uploaded from taurus, then pulled on PSC with
Slurm job `40962330` on `GPU-shared` / `v100-32:1`.

PSC target:
`/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval/models/keep1.0_r32/v0-20260523-050346-hf`

Validation:

- Slurm state: `COMPLETED`, exit code `0:0`, elapsed `00:22:43`
- manifest expected files: `28`
- present files: `28`
- safetensor shards: `15`
- expected bytes: `70541952230`
- actual bytes: `70541952230`
- missing / wrong / `.part` files: none
