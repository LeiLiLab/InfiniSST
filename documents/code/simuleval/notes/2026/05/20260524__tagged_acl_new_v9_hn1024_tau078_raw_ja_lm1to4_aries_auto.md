# Tagged ACL ja raw New V9 HN1024 tau0.78

Runs the ja tagged ACL main result for `lm=1,2,3,4` with the New V9
assistant-side term-tag-delay Speech LLM.

- Speech LLM: `/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_ja_r32a64_tp2_taurus4/keep1.0_r32/v0-20260524-123624-hf`
- Retriever: HN1024 `lh1b88kw`
- Tau: `0.78`
- Runtime glossary: tagged ACL raw
- Metric denominator: fixed tagged ACL raw glossary
- Output cleanup: strip `<term>` / `</term>` before scoring
- Launcher: `documents/code/simuleval/launchers/2026/05/20260524__tagged_acl_new_v9_hn1024_tau078_raw_ja_lm1to4_aries_auto.sh`

The launcher first waits until the existing medicine ja lm4 waiter has
submitted lm4, then starts polling aries for an idle 2-GPU pair and runs one
latency setting at a time.
