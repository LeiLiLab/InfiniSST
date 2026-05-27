## Hypothesis

Tagged term_map formatting should make relevant terms more salient to the Speech LLM and improve oracle/strict-term adoption compared with plain V3 real-retriever robust SFT.

## Background / Motivation

Medicine oracle examples show the model sometimes paraphrases or ignores a provided term_map entry.  This run keeps the same V3 robust mixture but wraps entries as `[TERM] source => target [/TERM]`.  The model architecture is unchanged, but inference must use the same tagged term_map format when evaluating this checkpoint.

## What changed vs baseline

- Baseline W&B candidate: `sst_omni/dvpmpqma` (`v2_srcm_r8a32`)
- Training data: `/mnt/gemini/data1/jiaxuanluo/speech_llm_v3_robust_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260520/train_s_zh_v3_tagged_termmap_lh1b88kw_tau073_srcmatch100k.jsonl`
- Dev data: `/mnt/gemini/data1/jiaxuanluo/speech_llm_v3_robust_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260520/dev_s_zh_v3_tagged_termmap_lh1b88kw_tau073_srcmatch100k.jsonl`
- Data manifest: `20260520T0000__data_prepare__v3_robust_termmap_zh`
- LoRA: rank 8, alpha 32
- Compute: taurus hold job `45269`, 2 GPUs
- Term-map format: tagged `[TERM] source => target [/TERM]`
- Inference code support: `agents/infinisst_omni_vllm_maxsim_rag.py --term-map-format tagged`

## Expected metrics

If salience is the bottleneck, tagged formatting should improve medicine oracle exact TERM_ACC and strict-term adoption without increasing false-copy or noisy term overuse.

## Verdict

Pending.
