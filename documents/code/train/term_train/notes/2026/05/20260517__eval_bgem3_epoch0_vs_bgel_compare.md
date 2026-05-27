## Hypothesis

The original BGE-M3 varctx576 epoch-0 checkpoint remains a strong control for
the same audio/data setup and should outperform or contextualize the BGE-large
epoch-0 text-encoder ablation.

## Background / Motivation

The Taurus BGE-large training run was canceled after early readouts looked weak.
To avoid relying on inline smoke metrics, this eval-only run measures the
matched BGE-M3 epoch-0 checkpoint using the same full dev / ACL6060 / medicine
readout as the BGE-large epoch-0 checkpoint.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Related canceled BGE-large run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/mhukv2bi
- Evaluated checkpoint:
  `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_epoch_0.pt`
- Text encoder: `BAAI/bge-m3`
- Text input prefix: empty
- Text pooling: `cls`
- Eval-only readout: full dev, full ACL6060, full medicine
- Glossary sizes: base + `gs1000` + `gs10000`
- Tau diagnostics: `0.75` only
- Eval scoring: GPU chunked scoring (`eval_score_device=cuda`,
  `query_chunk=256`, `text_chunk=1024`)

## Expected metrics

Use W&B history from this eval-only run and the paired BGE-large eval-only run
for the final comparison.  Primary comparison is raw `recall@10` across dev /
ACL6060 / medicine at base, `gs1000`, and `gs10000`; tau `0.75` precision,
filtered recall, kept count, and no-term noise are secondary diagnostics.

## Verdict

Completed successfully.  In the paired full eval-only readout, the BGE-M3
epoch-0 checkpoint outperforms the BGE-large epoch-0 ablation across dev,
ACL6060, and medicine, so BGE-M3 remains the stronger text-encoder control.
