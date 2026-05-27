# A1: Voice-Pool (GigaSpeech v2, clean-only) retriever ablation

## Hypothesis

Replacing the retriever training audio (wiki_synth branch only) with CosyVoice
TTS synthesized from **9,989 unique-opus GigaSpeech voice prompts** (v2 pool)
instead of **13 VCTK accent voices / the v1 seg-biased GigaSpeech pool with
~6,206 effective speakers** will *significantly reduce audio-domain overfit*
and *close some of the ACL6060 OOD gap* without hurting in-distribution
(wiki dev) recall. We expect:

- `eval_acl6060/recall@10_gs10000` to improve by at least +2 pp at the
  best-checkpoint step vs the 43848 smallest+dense MFA baseline.
- `eval_acl6060/swp@tau_0.80` (deployment-critical filter recall) to improve
  by at least +1 pp (or at worst, flat) — a drop here would mean the broader
  speaker distribution is *hurting* absolute cosine scales.
- `eval_dev/recall@10` to stay within -0.5 pp of 43848 (no ID regression).

No noise is added at TTS time (clean-only); pre-rendered noisy WAVs in
previous runs dragged ACL performance down (5tlz8ikk). If this ablation
confirms the audio-domain hypothesis, A2 will re-introduce lightweight
on-the-fly augmentation (SpecAugment / room-IR) in the dataloader.

## Background / Motivation

Current best SST retriever baselines (43848 `smallest+dense + pool_k=64 +
normAGGR`, 43849 `ablation_A k=1024 + normAGGR`) all train on wiki_synth TTS
generated with **only 13 VCTK accent speakers** from 109 UK studio recordings.
This is a known audio-domain overfit risk: the retriever sees training
acoustic diversity roughly = 13 accents + gigaspeech-speech chunks, but
ACL6060 eval has a much wider distribution (conference speakers, non-native
English, varied mic conditions). The observed ACL6060 / wiki_dev delta at
tau=0.80 in 43850 (-12 pp) is consistent with the retriever having learned
a speaker-sensitive similarity scale.

An initial POC/smoke with a GigaSpeech voice pool was run earlier, but the
pool was built by sampling *segments* from the MFA manifest, which allowed
multiple segments from the same opus (= same speaker) to appear — v1 had
10,000 entries but only 6,206 unique opus (max 27 segs per single opus). A
v2 pool (this experiment) is built with `--one-per-opus`: for each valid
opus file, one random segment is chosen, yielding 9,989 genuinely distinct
long-form recordings. See
`/mnt/gemini/home/jiaxuanluo/gigaspeech_speaker_prompts/diversity_report.json`
for the v1-vs-v2 audit.

## What changed vs baseline

- **Baseline run URL**: 43848 `q3_mfa_smallest_dense_k1024_normAGGR`
  (to be filled with full WandB URL at launch time; queried via
  `documents/code/general/wandb_tool.py get-run 43848` or project
  `qwen3_rag`, tag `family:sst_density_ablation`).
- **Diff**:
  - data: `TRAIN_JSONL` switches from the VCTK-13-voice TTS chunks
    (`term_train_3variant_1m_mfa.jsonl`) to the new v2 clean-only GigaSpeech
    pool TTS chunks (`term_train_3variant_gsv2_clean_mfa.jsonl`). Same
    underlying 3M wiki_synth utterances, same MFA pipeline, same 1.92s
    chunk grid (smallest+dense) — only the upstream TTS speaker set
    differs.
  - hparam: NOTHING intentionally changed. `NOISY_RATIO=0.0` is kept
    (v2 does not produce a noisy branch anyway).
  - code: `build_gigaspeech_voice_pool.py` gained `--one-per-opus` dedup;
    `merge_tts_3variant.py` made `noisy_audio_path` optional; qwen3 train
    script gained a no-op `--profile_out_dir` hook used only by the perf
    smoke (no effect on A1 training).

## Expected metrics

- `eval_acl6060/recall@10_gs10000` : 43848 best ≈ 0.xx (fetch at launch) →
  expected >= baseline + 2 pp.
- `eval_acl6060/swp@tau_0.80`      : baseline → expected >= baseline + 1 pp
  (must not regress by more than 0.5 pp; regression would falsify the
  speaker-diversity hypothesis).
- `eval_dev/recall@10`             : baseline → expected within ±0.5 pp
  (no ID regression).
- `train/pos_sim_mean`             : ±0.02 is acceptable; a large drop
  would indicate the new TTS is systematically different from wiki eval.

## Verdict

<!-- Agent fills this after the run finishes + WandB status tag flipped. -->
