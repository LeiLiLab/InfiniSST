# Enrich medicine strict glossary with target translations

## Hypothesis

Adding ESO v2 sentence-level target translations for strict GT terms, plus wiki-enriched translations for medicine filler terms, will make the same 10k medicine term set usable by SimulEval term_map injection without changing the retriever source-term bank.

## Background / Motivation

The strict medicine retriever readout uses MFA-exact source terms, but the generated glossary currently stores only English terms.  Speech LLM RAG evaluation needs `target_translations` for zh/de/ja term_map values.  ESO v2 sentence files contain translations for sentence-level GT terms; `documents/code/data_pre/glossary_scale/wiki_glossary_medicine_enriched.json` contains translations for the medicine wiki filler bank.

## What changed vs baseline

- **Baseline run URL**: N/A; data-prep enrichment derived from existing strict medicine data manifest.
- **Diff**:
  - input glossary: `/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_glossary_gt_plus_medicine_wiki_gs10000.json`
  - translation sources:
    - `/home/jiaxingxu/rag-sst/eso-dataset/outputs_v2/test/sample_*_v2/sentences_v2.json`
    - `documents/code/data_pre/glossary_scale/wiki_glossary_medicine_enriched.json`
  - output glossary: `/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_glossary_gt_plus_medicine_wiki_gs10000_translated.json`
  - matching rule: casefolded whitespace-normalized exact match on source term.

## Expected metrics

This is a provenance/data-prep step.  Expected output is a glossary with `target_translations` for the medicine GT terms present in ESO v2 sentence annotations and for wiki filler terms found in the enriched wiki medicine glossary, plus a stats JSON reporting coverage/conflicts.

## Verdict

SUCCESS: wrote the translated strict medicine glossary with `target_translations` for 10,000 of 10,000 entries.  Coverage came from 571 ESO sentence-term matches plus 9,429 wiki-enriched filler matches; untranslated entries = 0.
