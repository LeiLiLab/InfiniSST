## Hypothesis

New V10 sample50 no-GT-retrieved term-map SLM should recover de BLEU relative to the verified no-RAG streaming baseline while retaining a terminology advantage on tagged ACL raw.

## Background / Motivation

Clean NewV9 de had high terminology accuracy but weaker BLEU, plausibly because no-GT chunks trained with empty term maps made the SLM over-sensitive to term-map presence. New V10 sample50 keeps retriever-filtered term-map exposure on no-GT chunks but drops roughly half of no-GT term entries at term level to avoid overly dense prompts.

## What changed vs baseline

- Speech LLM: `/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_new_v10_no_gt_retrieved_termmap_sample50_de_r32a64_tp2_taurus8/keep1.0_r32/v0-20260525-083507-hf`
- Retriever: HN1024 ACL checkpoint.
- Runtime glossary: `acl6060_tagged_gt_raw_min_norm2`.
- Language: de.
- Latency multipliers: 1, 2, 3, 4.
- Batch protocol: same-lm batch, five ACL talks per lm, two Taurus GPUs per lm.
- Generation: `max_new_tokens=80`.
- RAG threshold: tau=0.79, from dev optimum 0.788 rounded for deployment.

## Expected metrics

The first gate is lm=2 BLEU no worse than the verified no-RAG de baseline while TERM_ACC remains clearly above no-RAG. lm=1,3,4 provide the full curve if the exported HF model loads cleanly.

## Verdict

Pending batch evaluation.
