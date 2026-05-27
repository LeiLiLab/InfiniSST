# Multi-Scale Inference Ablation Dev Readout

Primary TSV:

```text
documents/code/train/term_train/reports/figures/20260525_multiscale_inference_ablation_devraw.tsv
```

## Scope

This is a retriever-only dev readout for the paper multi-scale inference
ablation.  It uses verified W&B metrics and event manifests, not chat-memory
metrics.

Primary comparison:

- `q2fus6f1`: main multi-scale MaxSim checkpoint `lh1b88kw`, full internal
  window set `2,3,4,5,6,7,8,10,12,16,20,24`, varctx dev fixed raw denominator.
- `y454004y`: same checkpoint/protocol as `q2fus6f1`, but inference forced to
  `MAXSIM_WINDOWS=24`.
- `740c7y40`: dense single-embedding checkpoint from historical training run
  `r5l4780c`, evaluated with `USE_MAXSIM=false`, `MFA_SUPERVISED=false`, and
  fixed 1.92s dev fixed raw denominator.

`g3iayem1` is retained only as an auxiliary diagnostic: it evaluates the main
checkpoint on fixed 5.76s context while still using the full MaxSim window set,
so it is not the inference-window ablation described in the paragraph.

## Verified Results

| Variant | Run | Protocol | R@10 | R@10 gs10k | R@10 gs100k | tau=0.75 recall gs100k |
|---|---:|---|---:|---:|---:|---:|
| Multi-scale MaxSim | `q2fus6f1` | varctx fixed raw | 0.9920 | 0.9897 | 0.9858 | 0.9842 |
| Only largest MaxSim window | `y454004y` | varctx fixed raw | 0.9821 | 0.9747 | 0.9607 | 0.9362 |
| Dense 1.92s trained | `740c7y40` | ctx1.92 fixed raw | 0.9760 | 0.9597 | 0.9265 | 0.2394 |

## Interpretation

For the pure inference-window ablation, keeping only the largest internal
MaxSim window drops gs100k recall from `0.9858` to `0.9607` under the same
varctx dev protocol.  The tau-filtered recall drops more sharply, from
`0.9842` to `0.9362`.

The dense 1.92s trained run is lower still at gs100k recall (`0.9265`) and has
very low tau-filtered recall (`0.2394`).  This source run predates manifest
management and crashed after the short sweep budget, so it is backfilled from
W&B/filesystem evidence and should be described as a historical dense baseline,
not a fully modern-schema training run.

## Provenance

- Multi-scale reference manifest:
  `documents/code/train/term_train/manifests/2026/05/20260525T013600__retriever_eval__context_ablation_varctx_perctxraw_100k.json`
- Only-24 MaxSim manifest:
  `documents/code/train/term_train/manifests/2026/05/20260525T204630__retriever_eval__maxsim_w24_lh1b88kw_varctx_devraw_100k.json`
- Dense 1.92s eval manifest:
  `documents/code/train/term_train/manifests/2026/05/20260525T205627__retriever_eval__dense_ctx192_r5l4780c_devraw_100k.json`
- Dense 1.92s training backfill manifest:
  `documents/code/train/term_train/manifests/2026/04/20260404T201553__retriever_train__dense_ctx192_r5l4780c_backfill.json`
