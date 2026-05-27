#!/usr/bin/env python3
"""Focused smoke tests for multi-term chunk masking.

Run with the training env, for example:

  /mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python \
    documents/code/train/term_train/test_multiterm_hn_mask_logic.py

The tests avoid model/audio loading and only exercise the mask semantics added
for per-sample hard negatives and MFA term-scoped positives.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import qwen3_glossary_neg_train as train_mod  # noqa: E402


def _tid(term: str) -> int:
    return train_mod.stable_term_id(term)


def _gid(sample: dict) -> int:
    return train_mod.stable_group_id(train_mod.build_speech_group_key(sample))


def test_attach_chunk_positive_term_ids() -> None:
    samples = [
        {"utter_id": "u1", "chunk_idx": 7, "term_key": "realm"},
        {"utter_id": "u1", "chunk_idx": 7, "term_key": "model"},
        {"utter_id": "u2", "chunk_idx": 1, "term_key": "tag"},
    ]

    stats = train_mod.attach_chunk_positive_term_ids(samples)
    assert int(stats["multi_term_groups"]) == 1
    assert int(stats["rows_with_multi_term_group"]) == 2

    expected = {_tid("realm"), _tid("model")}
    assert set(samples[0]["_chunk_positive_term_ids"]) == expected
    assert set(samples[1]["_chunk_positive_term_ids"]) == expected
    assert set(samples[2]["_chunk_positive_term_ids"]) == {_tid("tag")}


def test_per_sample_hn_masks_cochunk_terms() -> None:
    device = torch.device("cpu")
    bank = train_mod.NegativeTermBank(["realm", "model", "tag", "other"], device)
    # Row 0 is close to realm/model/tag; row 1 is close to model/realm/tag.
    # The miner must exclude both co-chunk positives (realm + model) for both rows.
    bank.embeddings = torch.tensor(
        [
            [1.00, 0.00],
            [0.96, 0.04],
            [0.80, 0.20],
            [0.00, 1.00],
        ],
        dtype=torch.float32,
    )
    speech = torch.tensor([[1.0, 0.0], [0.95, 0.05]], dtype=torch.float32)
    local_tids = torch.tensor([_tid("realm"), _tid("model")], dtype=torch.long)
    valid = torch.tensor([True, True])
    positive_ids = torch.tensor(
        [[_tid("realm"), _tid("model")], [_tid("realm"), _tid("model")]],
        dtype=torch.long,
    )
    positive_mask = torch.ones_like(positive_ids, dtype=torch.bool)

    _embs, hn_tids, count = bank.mine_hard_negatives_per_sample(
        speech,
        local_tids,
        valid,
        positive_ids,
        positive_mask,
        k_per_sample=2,
    )
    assert count == 4
    forbidden = {_tid("realm"), _tid("model")}
    for row in hn_tids.tolist():
        assert not (set(row) & forbidden), f"co-chunk GT term leaked into HN: {row}"


def test_loss_scopes_cochunk_terms_correctly() -> None:
    sample_realm = {"utter_id": "u1", "chunk_idx": 7, "term_key": "realm"}
    sample_model = {"utter_id": "u1", "chunk_idx": 7, "term_key": "model"}
    group_ids = torch.tensor([_gid(sample_realm), _gid(sample_model)], dtype=torch.long)
    term_ids = torch.tensor([_tid("realm"), _tid("model")], dtype=torch.long)
    positive_ids = torch.tensor(
        [[_tid("realm"), _tid("model")], [_tid("realm"), _tid("model")]],
        dtype=torch.long,
    )
    positive_mask = torch.ones_like(positive_ids, dtype=torch.bool)
    valid = torch.tensor([True, True])

    speech = torch.eye(2, dtype=torch.float32)
    text = torch.eye(2, dtype=torch.float32)
    logit_scale = torch.tensor(1.0)

    term_scope = train_mod.compute_masked_contrastive_loss(
        speech_embs=speech,
        text_embs=text,
        logit_scale=logit_scale,
        local_group_ids=group_ids,
        local_term_ids=term_ids,
        local_positive_term_ids=positive_ids,
        local_positive_term_mask=positive_mask,
        local_valid_mask=valid,
        mfa_positive_scope="term",
    )
    assert float(term_scope["pos_count_mean"].item()) == 1.0
    assert float(term_scope["cochunk_neutral_count"].item()) == 2.0

    chunk_scope = train_mod.compute_masked_contrastive_loss(
        speech_embs=speech,
        text_embs=text,
        logit_scale=logit_scale,
        local_group_ids=group_ids,
        local_term_ids=term_ids,
        local_positive_term_ids=positive_ids,
        local_positive_term_mask=positive_mask,
        local_valid_mask=valid,
        mfa_positive_scope="chunk",
    )
    assert float(chunk_scope["pos_count_mean"].item()) == 2.0
    assert float(chunk_scope["cochunk_neutral_count"].item()) == 0.0

    hn_scope = train_mod.compute_masked_contrastive_loss(
        speech_embs=speech,
        text_embs=text,
        logit_scale=logit_scale,
        local_group_ids=group_ids,
        local_term_ids=term_ids,
        local_positive_term_ids=positive_ids,
        local_positive_term_mask=positive_mask,
        local_valid_mask=valid,
        per_sample_neg_embs=torch.stack([text[1:2], text[0:1]], dim=0),
        per_sample_neg_term_ids=torch.tensor([[_tid("model")], [_tid("realm")]], dtype=torch.long),
        mfa_positive_scope="term",
    )
    assert float(hn_scope["hn_false_positive_masked_count"].item()) == 2.0


def main() -> None:
    train_mod.set_term_id_normalize_mode("lower_strip")
    test_attach_chunk_positive_term_ids()
    test_per_sample_hn_masks_cochunk_terms()
    test_loss_scopes_cochunk_terms_correctly()
    print("[OK] multi-term HN/MFA mask smoke tests passed")


if __name__ == "__main__":
    main()
