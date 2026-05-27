## Hypothesis

Future retriever medicine readouts should default to the strict MFA-exact
medicine dataset, because non-MFA located positives are less defensible.

## Background / Motivation

The older medicine variable-context dataset includes `char_proportional` and
legacy `sentence_center_fallback` term rows. The strict
`clean_mfa_exact_only` dataset keeps only terms located by exact MFA word
interval matches and is the preferred medicine readout protocol.

## What changed vs baseline

The shared varctx HN/no-HN retriever launcher default medicine JSONL and
medicine eval glossary were changed to the `clean_mfa_exact_only` paths.

The previous non-strict medicine artifact directory was archived under its own
`archive/deprecated_non_strict_20260522T1015Z` directory. Compatibility
symlinks are retained at the original paths so historical manifests and running
jobs do not fail because a referenced file disappeared.

## Expected metrics

Future runs will report medicine metrics against the strict MFA-exact target
set by default. Existing running runs keep the paths already loaded at process
startup and should be interpreted with their recorded manifest/config.

## Verdict

Completed. Future launches through the shared varctx HN/no-HN retriever
launcher default to the strict MFA-exact medicine JSONL and glossary.

The legacy non-strict artifacts were moved to:

```text
/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76/archive/deprecated_non_strict_20260522T1015Z
```

The original top-level file paths are now compatibility symlinks, with a
`DEPRECATED_USE_clean_mfa_exact_only.txt` marker in the legacy directory.
