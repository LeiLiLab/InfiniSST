## Hypothesis

Continuing the BGE-large-en-v1.5 varctx576 run from the epoch-0 checkpoint,
while raising GradCache chunk size from 128 to 256 and evaluating every 100
steps, should preserve the same training semantics while reducing step overhead
and giving denser convergence readouts than the 240-step scout run.

## Background / Motivation

The source run `ggeqpwie` / Slurm `45236` produced a usable epoch-0 checkpoint
but the run itself failed after inline eval exceeded the NCCL wait budget.  The
checkpoint at
`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_txt_bgel_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_dev100Tau1_eval240_taurus8_smoke2000_epoch_0.pt`
was inspected locally and contains `epoch=0`, `global_step=882`, optimizer,
scheduler, scaler, audio model, and text model state.

This continuation removes the 2000-step smoke cap so training can continue
through the planned 6-epoch schedule.  It keeps the same data, audio encoder,
BGE-large text encoder, hard-negative depth, MFA setting, TCM-off setup, and
held-out ACL/medicine readouts.

## What changed vs baseline

- Resume checkpoint: source run `ggeqpwie`, epoch-0 checkpoint, `global_step=882`.
- Baseline/control comparison remains `lh1b88kw`.
- GradCache chunk: `128` -> `256`.
- Inline eval interval: `240` -> `100`.
- Max steps: `2000` smoke cap -> `0` no explicit step cap.
- Epoch target: keep `EPOCHS=6`, so resume starts at epoch 1 and can run epochs
  1 through 5.
- W&B parent/baseline ids: `ggeqpwie lh1b88kw`.
- ACL remains held-out readout only and must not be used for model selection.

## Expected metrics

Primary checkpoint metric remains `eval_dev/recall@10_gs10000` on the fixed
100-sample dev smoke subset.  Secondary tracker remains
`eval_acl6060/recall@10` for held-out ACL readout continuity.  Cross-run
comparisons should use `wandb_tool.py compare ... --at-best-step` or
`db-compare --refresh`, not raw loop-summary keys.

## Verdict

PENDING: fill after the resumed run finishes and W&B is synced.
