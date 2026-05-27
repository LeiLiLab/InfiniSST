# HN Depth Scout Protocol (`TCM off`)

Goal: freeze the current `smallest + dense + normAGGR` retriever recipe, turn
TCM completely off, and measure the sweet spot of `hard_neg_k_per_sample`
without mixing in a second boundary-shaping mechanism.

This is explicitly the stage **before** any new TCM sweep. The intent is:

1. fix the base retriever's hard-negative depth first,
2. then revisit TCM as the last-stage absolute-threshold calibration step.

## Historical anchors from WandB

Top historical anchors in family `sst_ood_hardneg` by
`eval_acl6060/topk10_filtered_recall@tau_0p80_gs10000`:

1. `ll5a6p9k` — `hnps_k512_lambda5_smallest_dense_normAGGR_6gpu`
   - URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/ll5a6p9k
   - Why not the baseline: TCM weight `lambda=5` is on, so it is not a clean
     HN-only reference.
2. `tys70s0y` — `hnps_k512_smallest_dense_normAGGR_6gpu`
   - URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/tys70s0y
   - Best secondary bundle: `recall@10_gs10000=0.8419`, fixed-probe
     `filt@tau0.80_gs10000=0.7798`, `noise@tau0.80_gs10000=2.78`
   - Why it matters: same base recipe and same per-sample HN policy, but legacy
     TCM is still on.
3. `zv28ve3q` — `variantE_smallest_dense_k64`
   - URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/zv28ve3q
   - Best secondary bundle: `recall@10_gs10000=0.8744`, fixed-probe
     `filt@tau0.80_gs10000=0.7736`, `noise@tau0.80_gs10000=2.07`
   - Why it matters: pool-HN control showing the old `k=64` branch.

These are **anchors**, not a fully clean non-TCM baseline. For the immediate
scout we intentionally reuse `tys70s0y` as a practical historical reference and
skip a fresh `k=512, TCM off` rerun to save Aries budget.

## Why a short scout is enough

Existing matched-step history already shows that the base recipe separates
meaningful HN regimes early:

- at step `160`, `tys70s0y` already reaches
  `recall@10_gs10000=0.7101`, `filt@tau0.80_gs10000=0.3194`
- at step `200`, it reaches
  `recall@10_gs10000=0.7039`, `filt@tau0.80_gs10000=0.4961`
- by step `520`, the same run reaches
  `recall@10_gs10000=0.8419`, `filt@tau0.80_gs10000=0.7798`

That is enough evidence that a scout to roughly `160-200` steps can rank
candidate HN depths without paying for a full training run.

## Comparison metrics

Use two layers of metrics:

### Selection metrics

Use these to pick the HN sweet spot:

- `eval_dev/recall@10_gs10000`
- `eval_dev/topk10_filtered_recall@tau_0p80_gs10000`
- `train/step_time_ms`

Rationale: this keeps the main selection logic on the in-domain dev side and
uses the fixed `tau=0.80` probe only as a stable reference cut.

### Diagnostic readout metrics

Track these on ACL during the scout, but treat them as readout rather than the
only selection criterion:

- `eval_acl6060/recall@10_gs10000`
- `eval_acl6060/topk10_filtered_recall@tau_0p80_gs10000`
- `eval_acl6060/noterm_noise@top10_tau_0p80_gs10000`

Important: with `TCM off`, `tau=0.80` is **not** the final claimed deployment
threshold. It is just a fixed probe used to compare score-shape changes across
HN depths.

## Sweet-spot rule

Pick the shallowest depth that lands on the Pareto frontier:

1. higher `eval_dev/recall@10_gs10000`,
2. then higher `eval_dev/topk10_filtered_recall@tau_0p80_gs10000`,
3. while keeping `train/step_time_ms` below roughly `2x` the `k=512` control.

Reject a deeper HN setting if it gains less than `+0.5pp` on the dev-side
selection metrics but costs more than `+50%` step time.

ACL numbers are still logged and reviewed, but a final paper-facing winner
should not be selected by ACL alone.

## Current scout budget

Single-node Aries budget target: about `10h` total wall-clock.

This round uses two main comparisons plus one staged feasibility wrapper:

### H1 — main comparison A

- `hard_neg_k_per_sample=1024`
- `tcm_loss_weight=0`, `tcm_pos_loss_weight=0`, `tcm_neg_loss_weight=0`
- `MAX_STEPS=200`
- `MAX_TRAIN_SECONDS=14400` (`4.0h`)

Purpose: first clean test of whether going beyond the historical `k=512`
reference still helps under the current `smallest + dense + normAGGR` recipe.

### H2 — main comparison B

- `hard_neg_k_per_sample=4096`
- `tcm_loss_weight=0`, `tcm_pos_loss_weight=0`, `tcm_neg_loss_weight=0`
- `MAX_STEPS=120`
- `MAX_TRAIN_SECONDS=12600` (`3.5h`)

Purpose: a clearly deeper HN regime that can answer whether the gain from
per-sample mining keeps growing after `1024`.

Promotion rule:

- no OOM in the first `40` steps,
- `train/step_time_ms <= 90000`,
- by step `80`, dev metrics are not obviously collapsing versus H1.

If all three hold, H2 becomes the leading candidate for a longer confirm run.

### H3 — staged only, not submitted initially

- `hard_neg_k_per_sample=8192`
- `tcm_loss_weight=0`, `tcm_pos_loss_weight=0`, `tcm_neg_loss_weight=0`
- `MAX_STEPS=40`
- `MAX_TRAIN_SECONDS=5400` (`1.5h`)

Purpose: check whether `8192` is even feasible on the current Aries setup after
seeing whether `4096` remains healthy. This wrapper is prepared now but only
submitted if H2 still looks viable.

## Launch order

1. submit `H1` (`k=1024`) now,
2. queue `H2` (`k=4096`) behind it,
3. keep `H3` (`k=8192`) staged but unsubmitted until `H2` looks healthy.

This matches the current priority: spend budget on the two informative depths
first, then only pay for `8192` if the deeper regime still works at all.

## Deferred until after HN freeze

After the HN sweet spot is frozen:

- re-run the best non-TCM depth longer,
- decide the final non-TCM base retriever,
- only then reopen TCM as the last-stage absolute-threshold exploration.
