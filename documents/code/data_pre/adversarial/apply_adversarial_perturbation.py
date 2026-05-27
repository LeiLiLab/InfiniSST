"""
Apply adversarial perturbation to speech-LLM training JSONL.

For each trivial GT term that has an adversarial alternative (from
generate_adversarial_translations.py), perturb a fraction of its training instances
so the canonical Chinese translation is replaced with a non-canonical alternative
BOTH in:
  - gt_terms_by_chunk[N][i]["zh"]  (which rebuild_termmap.py will later inject into term_map)
  - messages[2N + 2]["content"]    (the reference translation the model is trained to emit)

This creates a training signal that explicitly rewards copying from term_map and
punishes defaulting to zero-shot prior.

Perturbation atomicity: we ONLY apply a perturbation for a specific (chunk, term) pair
if the canonical_zh is found in the corresponding reference content. If not found,
we skip that pair entirely (no term_map change, no reference change) to preserve
signal consistency. We fail loudly if the post-replacement content does not contain
the adversarial_zh.

All user-facing strings are in English.
"""

# ======Configuration=====
import argparse
import json
import random
from pathlib import Path
from typing import Dict

DEFAULT_INPUT_JSONL = "/mnt/gemini/data1/jiaxuanluo/train_cleaned_with_retriever_results_varlen.jsonl"
DEFAULT_ADV_TRANS_JSON = "/mnt/gemini/data1/jiaxuanluo/adversarial/adversarial_translations.json"
DEFAULT_OUTPUT_JSONL = "/mnt/gemini/data1/jiaxuanluo/adversarial/train_cleaned_with_retriever_results_varlen_adv.jsonl"
DEFAULT_STATS_JSON = "/mnt/gemini/data1/jiaxuanluo/adversarial/perturbation_stats.json"
DEFAULT_PERTURB_PROB = 0.5
DEFAULT_SEED = 42
# Maximum tolerated fraction of target (chunk, term) pairs that could not be perturbed
# due to missing canonical_zh in reference content. If exceeded, we fail loudly.
MAX_MISS_FRACTION_SOFT = 0.4
# ======Configuration=====


def perturb_one_conversation(
    conversation: Dict,
    adversarial_map: Dict[str, str],
    perturb_prob: float,
    rng: random.Random,
    stats: Dict,
) -> Dict:
    """Apply adversarial perturbation in place and return the modified conversation.

    Modifies:
      - conversation["gt_terms_by_chunk"][N][i]["zh"] for perturbed terms
      - conversation["messages"][2N + 2]["content"] for those chunks
    """
    gt_terms_by_chunk = conversation.get("gt_terms_by_chunk", [])
    messages = conversation.get("messages", [])
    assert isinstance(gt_terms_by_chunk, list), "gt_terms_by_chunk must be list"
    assert isinstance(messages, list), "messages must be list"

    for chunk_idx, chunk_terms in enumerate(gt_terms_by_chunk):
        if not isinstance(chunk_terms, list) or not chunk_terms:
            continue
        assistant_msg_idx = 2 * chunk_idx + 2
        if assistant_msg_idx >= len(messages):
            # Dangling gt_terms chunk beyond messages; skip
            stats["chunks_missing_assistant_msg"] += 1
            continue
        assistant_msg = messages[assistant_msg_idx]
        assert isinstance(assistant_msg, dict), \
            f"messages[{assistant_msg_idx}] not a dict"
        assert assistant_msg.get("role") == "assistant", (
            f"Expected assistant at messages[{assistant_msg_idx}], "
            f"got role={assistant_msg.get('role')}"
        )
        content = assistant_msg.get("content", "")
        assert isinstance(content, str), \
            f"assistant content at {assistant_msg_idx} must be str"

        for term in chunk_terms:
            en = (term.get("term") or "").strip()
            canonical_zh = (term.get("zh") or "").strip()
            if not en or not canonical_zh:
                continue
            stats["candidate_terms_total"] += 1

            adv_zh = adversarial_map.get(en, "")
            if not adv_zh:
                stats["candidate_terms_not_in_adv_map"] += 1
                continue

            if adv_zh == canonical_zh:
                # adversarial_translations.json is keyed by en only. The same en may have
                # multiple canonical zh across the corpus; when this instance's canonical
                # happens to equal the adv (which was picked against a different canonical),
                # perturbation would be a no-op and break the adversarial signal. Skip.
                stats["adv_equals_canonical_in_instance"] += 1
                continue

            stats["candidate_terms_in_adv_map"] += 1

            if rng.random() >= perturb_prob:
                stats["skipped_by_prob"] += 1
                continue

            stats["targeted_for_perturbation"] += 1

            if canonical_zh not in content:
                stats["missed_canonical_in_reference"] += 1
                continue

            new_content = content.replace(canonical_zh, adv_zh)
            # Loud check: at least one replacement happened
            assert new_content != content, (
                f"Replacement produced identical content "
                f"(canonical={canonical_zh!r} adv={adv_zh!r})"
            )
            # Loud check: adversarial now present
            assert adv_zh in new_content, (
                f"Post-replacement reference does not contain adversarial "
                f"(adv={adv_zh!r} new_content[:50]={new_content[:50]!r})"
            )
            assistant_msg["content"] = new_content
            content = new_content
            term["zh"] = adv_zh
            stats["actually_perturbed"] += 1

    return conversation


def main():
    parser = argparse.ArgumentParser(description="Apply adversarial perturbation to training JSONL.")
    parser.add_argument("--input-jsonl", type=str, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--adversarial-json", type=str, default=DEFAULT_ADV_TRANS_JSON)
    parser.add_argument("--output-jsonl", type=str, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--stats-json", type=str, default=DEFAULT_STATS_JSON)
    parser.add_argument("--perturb-prob", type=float, default=DEFAULT_PERTURB_PROB)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke-n", type=int, default=0,
                        help="If > 0, only process first N conversations (for smoke testing)")
    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl)
    stats_json = Path(args.stats_json)
    adv_json = Path(args.adversarial_json)

    assert input_jsonl.is_file(), f"Input JSONL not found: {input_jsonl}"
    assert adv_json.is_file(), f"Adversarial JSON not found: {adv_json}"

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stats_json.parent.mkdir(parents=True, exist_ok=True)

    adversarial_map: Dict[str, str] = json.loads(adv_json.read_text(encoding="utf-8"))
    assert adversarial_map, f"Empty adversarial map: {adv_json}"
    print(f"[INFO] Loaded {len(adversarial_map)} adversarial translations.", flush=True)

    rng = random.Random(args.seed)
    stats = {
        "conversations_total": 0,
        "candidate_terms_total": 0,
        "candidate_terms_in_adv_map": 0,
        "candidate_terms_not_in_adv_map": 0,
        "targeted_for_perturbation": 0,
        "actually_perturbed": 0,
        "skipped_by_prob": 0,
        "missed_canonical_in_reference": 0,
        "chunks_missing_assistant_msg": 0,
        "adv_equals_canonical_in_instance": 0,
    }

    written = 0
    with input_jsonl.open("r", encoding="utf-8") as fin, \
         output_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if args.smoke_n > 0 and stats["conversations_total"] >= args.smoke_n:
                break
            conv = json.loads(line)
            perturb_one_conversation(conv, adversarial_map, args.perturb_prob, rng, stats)
            fout.write(json.dumps(conv, ensure_ascii=False) + "\n")
            stats["conversations_total"] += 1
            written += 1

    if stats["targeted_for_perturbation"] > 0:
        miss_fraction = stats["missed_canonical_in_reference"] / stats["targeted_for_perturbation"]
        stats["miss_fraction"] = miss_fraction
    else:
        stats["miss_fraction"] = 0.0

    stats_json.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"[INFO] Wrote {written} conversations to {output_jsonl}", flush=True)
    print(f"[INFO] Stats: {json.dumps(stats, ensure_ascii=False, indent=2)}", flush=True)

    # Loud check: miss fraction should not be excessively high, indicating a systemic
    # mismatch between gt_terms_by_chunk alignment and reference content.
    if args.smoke_n == 0 and stats["targeted_for_perturbation"] > 0:
        assert stats["miss_fraction"] <= MAX_MISS_FRACTION_SOFT, (
            f"Miss fraction {stats['miss_fraction']:.3f} exceeds threshold "
            f"{MAX_MISS_FRACTION_SOFT:.3f}. This suggests a systemic alignment issue "
            f"between gt_terms_by_chunk and reference content. "
            f"Investigate rebuild_termmap.py conventions before proceeding."
        )


if __name__ == "__main__":
    main()
