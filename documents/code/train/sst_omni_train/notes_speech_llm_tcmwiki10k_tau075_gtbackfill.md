## Hypothesis

Training the speech LLM with deployment-like TCM retriever term maps should reduce mismatch between training and ACL SimulEval inference, improving term adoption on the weakest v4_ner baseline paper/latency case without changing the speech-LLM base recipe.

## Background / Motivation

The old v4_ner baseline uses a fixed NER-style term-map construction. The new data keeps the same Qwen3-Omni base/MCore and LoRA recipe, but replaces training term maps with candidates produced by the strongest current TCM RAG checkpoint over a larger general unseen P31 wiki glossary plus all GigaSpeech training GT terms.

## What changed vs baseline

- Baseline run URL: historical debt; the comparison baseline is the HF checkpoint `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4`, not a schema-compliant WandB run.
- Diff: term_map generation uses `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_ntcm_final_v3_n64_p1n4_s2000_aries-8gpu_smallest_dense_smoke4650_best.pt`, 100k translated P31 unseen wiki candidates merged with every GT term from the GigaSpeech training JSONL, `tau=0.75`, and `k=min(10, ceil(duration_sec * 5))`. Retrieved candidates passing tau are included for all chunks; missing GT terms are randomly inserted for with-term chunks.

## Expected metrics

Primary check is corrected ACL per-paper SimulEval on the weakest v4_ner `(paper, lm)` across raw/gs1k/gs10k glossaries. Expect TERM_ADOPTION/TERM_ADOPTION_MICRO to improve or stay flat while TERM_FCR does not increase materially.

## Verdict

Pending training and targeted SimulEval evaluation.
