## Hypothesis

The BGE-large-en-v1.5 epoch-0 checkpoint should be evaluated with the same full
dev / ACL6060 / medicine readout as the BGE-M3 epoch-0 control before making a
final text-encoder decision from the noisy inline training readouts.

## Background / Motivation

The Taurus BGE-large training run was canceled after early readouts looked weak.
The run still produced an epoch-0 checkpoint.  This eval-only run measures that
checkpoint directly against the matched BGE-M3 epoch-0 checkpoint using the
same data paths, glossary sizes, and tau diagnostic policy.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Related canceled BGE-large run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/mhukv2bi
- Evaluated checkpoint:
  `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_txt_bgel_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_dev100Tau1_eval240_taurus8_smoke2000_epoch_0.pt`
- Text encoder: `BAAI/bge-large-en-v1.5`
- Text input prefix: empty
- Text pooling: `cls`
- Eval-only readout: full dev, full ACL6060, full medicine
- Glossary sizes: base + `gs1000` + `gs10000`
- Tau diagnostics: `0.75` only
- Eval scoring: GPU chunked scoring (`eval_score_device=cuda`,
  `query_chunk=256`, `text_chunk=1024`)

## Expected metrics

Use W&B history from this eval-only run and the paired BGE-M3 eval-only run for
the final comparison.  Primary comparison is raw `recall@10` across dev /
ACL6060 / medicine at base, `gs1000`, and `gs10000`; tau `0.75` precision,
filtered recall, kept count, and no-term noise are secondary diagnostics.

## Verdict

Completed successfully.  In the paired full eval-only readout, the BGE-large
epoch-0 checkpoint underperforms the matched BGE-M3 epoch-0 control across dev,
ACL6060, and medicine, so BGE-large should not be promoted from this ablation.
