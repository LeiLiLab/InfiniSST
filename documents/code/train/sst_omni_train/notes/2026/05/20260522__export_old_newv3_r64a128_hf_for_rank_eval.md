# Export old new_v3 r64/a128 checkpoints for rank eval

## Hypothesis

The old `new_v3` r64/a128 checkpoints can be evaluated with the same tagged ACL
fast pipeline after converting their MCore adapter directories to HF format.

## Background / Motivation

The r32/a64 old `new_v3` line already has a quick tagged ACL `zh lm=2 raw`
readout.  To isolate the LoRA-capacity effect, we also need the two r64/a128
old checkpoints:

- full `new_v3`: W&B `q159wce4`
- random `new_v3`: W&B `rj1v1p7r`

Only MCore checkpoint directories were found, so this event exports them to HF
under `/mnt/aries/data7/jiaxuanluo/slm`.

## What changed vs baseline

No model training is performed.  This is a format conversion only:

- input MCore full: `/mnt/gemini/data2/jiaxuanluo/speech_llm_tcmw100kgt_tau075_new_v3_r64a128_taurus4/keep1.0_r64/v1-20260508-135111`
- input MCore random: `/mnt/gemini/data2/jiaxuanluo/speech_llm_tcmw100kgt_tau075_new_v3_random_r64a128_aries8/keep1.0_r64/v1-20260508-143031`
- output HF root: `/mnt/aries/data7/jiaxuanluo/slm/old_newv3_rank_ablation`

## Expected metrics

No metrics for this event.  Success means each HF output has `config.json`,
`generation_config.json`, and model weights, and can be loaded by vLLM.

## Verdict

Pending export.
