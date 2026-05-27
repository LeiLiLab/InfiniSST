# ACL Paper-Extracted Union Glossaries zh

## Hypothesis

Merging the five ACL paper-extracted glossaries into one strict raw glossary will remove unnecessary per-paper runtime splitting while preserving the same extracted-term metric denominator.

## Background / Motivation

The previous ACL paper-extracted result used one glossary per paper.  The new zh rerun should evaluate all five papers together, using the union raw glossary for strict TERM metrics and separate raw/1k/10k runtime banks for RASST.

## What changed vs baseline

- Build a de-duplicated zh raw union from the five `extracted_glossary__2022.acl-long.*.json` files.
- Build `gs1000` and `gs10000` banks by preserving all raw union terms first, then appending deterministic wiki filler terms with zh translations.
- Keep TERM metrics fixed to the raw union glossary for all downstream runtime glossary sizes.

## Expected metrics

No model metrics are produced by this data-prep event.  Validation should report 253 raw terms, 1,000 gs1k terms, and 10,000 gs10k terms with no duplicate normalized terms and non-empty zh translations.

## Verdict

Completed.  Data prep produced 253 raw union terms, 1,000 gs1k terms, and 10,000 gs10k terms.  Independent validation confirmed no duplicate normalized terms, non-empty zh translations, and full raw-term inclusion in both expanded banks.
