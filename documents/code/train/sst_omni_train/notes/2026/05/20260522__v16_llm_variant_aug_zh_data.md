# V16 LLM-variant term translation augmentation data

## Hypothesis

V16 should preserve the adoption pressure of V15 while reducing the artificial distribution shift from marker suffixes.  If the Speech LLM ignores `term_map` because canonical translations are easy to infer, natural but uncommon target variants should make the map more necessary for minimizing SFT loss.

## Background / Motivation

V13 uses inference-aligned retriever timeline `term_map` entries.  V15 strengthens GT term supervision with explicit marker suffixes, but those markers are intentionally artificial.  V16 instead asks an OpenAI model to generate a short natural Chinese alternative for selected GT term translations, then atomically replaces both the current `term_map` value and the first exact future assistant occurrence.

## What changed vs baseline

- Baseline data: V13 lm1..6 retriever timeline data.
- Script: `documents/code/train/sst_omni_train/src/augment_term_translation_llm_variants.py`
- Launcher: `documents/code/train/sst_omni_train/launchers/2026/05/20260522__build_v16_llm_variant_aug_zh.sh`
- Train input:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_v13_lm1to6_retriever_timeline_zh_lh1b88kw_tau073_20260522/train_s_zh_v13_lm1to6_retriever_timeline_tau073_k10_minctx2p88.jsonl`
- Output directory:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_v16_llm_variant_aug_retriever_timeline_zh_lh1b88kw_tau073_20260522`
- Replacement is allowed only when:
  - the GT term is present in the current chunk's `term_map`;
  - the canonical target translation exact-matches assistant text from the current chunk onward;
  - the OpenAI-generated variant is non-empty, short, different from the canonical translation, and not marker-like.
- API key is supplied only through `OPENAI_API_KEY`; it is not written to files.

## Expected metrics

Downstream check should focus on tagged ACL `zh lm=2 raw`: `TERM_ACC`, `REAL_ADOPT`, `TERM_FCR`, BLEU, and StreamLAAL.  Compared with V15, V16 should be less likely to damage BLEU if the generated variants remain fluent.

## Verdict

Success.  Output is under
`/mnt/gemini/data1/jiaxuanluo/speech_llm_v16_llm_variant_aug_retriever_timeline_zh_lh1b88kw_tau073_20260522`.

Train split:

- rows: `6237`
- GT terms: `39468`
- GT terms in current term_map: `33544` (`84.99%`)
- selected terms: `16742`
- augmented terms: `16358`
- augmented / GT terms: `41.45%`
- augmented / GT-in-map terms: `48.77%`
- augmented / selected terms: `97.71%`

Variant cache:

- total: `5003`
- ok: `4990`
- invalid: `13`

Invalid variants were filtered rather than used for supervision, mostly identity outputs for names or already-English terms.
