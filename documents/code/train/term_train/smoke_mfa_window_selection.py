"""Fast unit-smoke for the 3 MFA window selection modes.

Validates:
  1. Output shape is always [B, N] regardless of mode.
  2. hard_max picks the max-similarity covering window (baseline behavior).
  3. smallest picks the argmin-by-duration covering window (independent of sim).
  4. logsumexp aggregates all covering windows; softmax weights sum to 1
     over covering windows and are exactly 0 on non-covering windows.
  5. Gradient flows through the selected window(s).
  6. The "no covering window" per-row fallback correctly routes to the
     longest window (tightest feasible crop given the term).

Run with:
    python /mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/\
        smoke_mfa_window_selection.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

import torch

# ======Configuration=====
SCRIPT_PATH = (
    "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/"
    "qwen3_glossary_neg_train.py"
)
SMOKE_SEED = 20260418
SMOKE_BATCH = 6
SMOKE_WINDOWS = 6  # distinct (start, end) windows
SMOKE_NEG_COUNT = 5
SMOKE_DIM = 16
LSE_TEMPERATURE = 1.0
# ======Configuration=====


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "qwen3_glossary_neg_train", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec for {SCRIPT_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qwen3_glossary_neg_train"] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_fixture(seed: int):
    torch.manual_seed(seed)
    B, W, N, D = SMOKE_BATCH, SMOKE_WINDOWS, SMOKE_NEG_COUNT, SMOKE_DIM
    speech = torch.randn(B, W, D, requires_grad=True)
    text = torch.randn(N, D)

    # Windows: varying durations [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]s with varying
    # starts to create non-trivial coverage patterns.
    win_starts = torch.tensor([0.0, 0.2, 0.5, 0.3, 0.1, 0.0])
    win_ends = torch.tensor([0.5, 1.0, 1.5, 1.8, 2.1, 3.0])
    assert win_starts.shape[0] == W and win_ends.shape[0] == W

    # Terms: mix of covered-by-some, covered-by-many, covered-by-none, and no-MFA.
    mfa_starts = torch.tensor([
        0.3, 0.4, 0.1, 0.7, 2.8, -1.0,   # last entry: no MFA -> fallback "all"
    ])
    mfa_ends = torch.tensor([
        0.9, 0.9, 0.3, 1.2, 2.95, -1.0,  # row 4: term 2.8-2.95, only w=5 (0-3s) covers
    ])
    return speech, text, mfa_starts, mfa_ends, win_starts, win_ends


def _covering(ws, we, ts, te):
    B = ts.shape[0]
    W = ws.shape[0]
    out = torch.zeros(B, W, dtype=torch.bool)
    for b in range(B):
        if ts[b] < 0:
            out[b] = True  # fallback "all" for no-MFA rows
            continue
        for w in range(W):
            if ws[w] <= ts[b] and we[w] >= te[b]:
                out[b, w] = True
    return out


def test_hard_max(mod):
    speech, text, ts, te, ws, we = _build_fixture(SMOKE_SEED)
    cov = _covering(ws, we, ts, te)

    out, debug = mod._maxsim_score_mfa(
        speech, text, ts, te, ws, we,
        selection_mode="hard_max", return_debug=True,
    )
    B, W, D = speech.shape
    N = text.shape[0]
    assert out.shape == (B, N), f"hard_max shape={out.shape}"

    # Reference: max over covering windows.
    sim_all = torch.matmul(speech, text.T)  # [B, W, N]
    ref = torch.full((B, N), -1e9)
    for b in range(B):
        valid_w = cov[b].nonzero(as_tuple=False).flatten()
        if valid_w.numel() == 0:
            # per-row fallback: longest window only
            longest_idx = int((we - ws).argmax().item())
            ref[b] = sim_all[b, longest_idx]
        else:
            ref[b] = sim_all[b, valid_w].max(dim=0).values
    max_abs_diff = (out - ref).abs().max().item()
    assert max_abs_diff < 1e-4, f"hard_max diff={max_abs_diff}"
    print(f"[OK] hard_max: shape={tuple(out.shape)} max_abs_diff={max_abs_diff:.2e}")
    print(f"     covering counts per row = {debug['covering_counts'].tolist()}")
    print(f"     fallback rows = {debug['fallback_rows'].tolist()}")


def test_smallest(mod):
    speech, text, ts, te, ws, we = _build_fixture(SMOKE_SEED)
    cov = _covering(ws, we, ts, te)

    out, debug = mod._maxsim_score_mfa(
        speech, text, ts, te, ws, we,
        selection_mode="smallest", return_debug=True,
    )
    B, W, D = speech.shape
    N = text.shape[0]
    assert out.shape == (B, N), f"smallest shape={out.shape}"

    # Reference: smallest-duration covering window.
    sim_all = torch.matmul(speech, text.T)
    win_dur = we - ws  # [W]
    ref = torch.zeros(B, N)
    expected_idx = torch.zeros(B, dtype=torch.long)
    for b in range(B):
        valid_w = cov[b].nonzero(as_tuple=False).flatten()
        if valid_w.numel() == 0:
            longest_idx = int(win_dur.argmax().item())
            ref[b] = sim_all[b, longest_idx]
            expected_idx[b] = longest_idx
        else:
            durs = win_dur[valid_w]
            pick = valid_w[int(durs.argmin().item())]
            ref[b] = sim_all[b, pick]
            expected_idx[b] = pick
    max_abs_diff = (out - ref).abs().max().item()
    assert max_abs_diff < 1e-4, f"smallest diff={max_abs_diff}"
    # Sanity: selected_win_idx must match reference.
    assert torch.equal(debug["selected_win_idx"], expected_idx), (
        f"smallest idx mismatch: got {debug['selected_win_idx'].tolist()}, "
        f"expected {expected_idx.tolist()}"
    )
    print(f"[OK] smallest: shape={tuple(out.shape)} max_abs_diff={max_abs_diff:.2e}")
    print(f"     selected_win_idx per row = {debug['selected_win_idx'].tolist()}")
    print(f"     selected_win_dur per row = {debug['selected_win_dur'].tolist()}")
    print(f"     fallback rows = {debug['fallback_rows'].tolist()}")


def test_logsumexp(mod):
    speech, text, ts, te, ws, we = _build_fixture(SMOKE_SEED)
    cov = _covering(ws, we, ts, te)

    out, debug = mod._maxsim_score_mfa(
        speech, text, ts, te, ws, we,
        selection_mode="logsumexp",
        lse_temperature=LSE_TEMPERATURE,
        return_debug=True,
    )
    B, W, D = speech.shape
    N = text.shape[0]
    assert out.shape == (B, N), f"logsumexp shape={out.shape}"

    # Reference: tau * logsumexp(sim/tau) over covering windows.
    sim_all = torch.matmul(speech, text.T)
    ref = torch.zeros(B, N)
    for b in range(B):
        cov_b = cov[b].clone()
        if cov_b.sum() == 0:
            longest_idx = int((we - ws).argmax().item())
            cov_b[longest_idx] = True
        masked = sim_all[b].clone()
        masked[~cov_b] = float("-inf")
        ref[b] = LSE_TEMPERATURE * torch.logsumexp(masked / LSE_TEMPERATURE, dim=0)
    max_abs_diff = (out - ref).abs().max().item()
    assert max_abs_diff < 1e-4, f"logsumexp diff={max_abs_diff}"

    # Sanity: softmax weights at column 0 must sum to 1 over covering windows
    # and be exactly 0 on non-covering windows.
    weights = debug["softmax_weights_col0"]  # [B, W]
    for b in range(B):
        cov_b = cov[b].clone()
        if cov_b.sum() == 0:
            longest_idx = int((we - ws).argmax().item())
            cov_b[longest_idx] = True
        wsum = weights[b, cov_b].sum().item()
        assert abs(wsum - 1.0) < 1e-4, f"row {b} covering weight sum={wsum}"
        non_cov_wsum = weights[b, ~cov_b].sum().item()
        assert abs(non_cov_wsum) < 1e-6, (
            f"row {b} non-covering weight sum={non_cov_wsum} (should be 0)"
        )
    print(f"[OK] logsumexp: shape={tuple(out.shape)} max_abs_diff={max_abs_diff:.2e}")
    print(f"     covering counts = {debug['covering_counts'].tolist()}")
    print(f"     softmax weights row 0 (col 0) = {weights[0].tolist()}")
    print(f"     softmax weights row 5 (no-MFA, col 0) = {weights[5].tolist()}")


def test_gradient_flow(mod):
    for mode in ("hard_max", "smallest", "logsumexp"):
        speech, text, ts, te, ws, we = _build_fixture(SMOKE_SEED + 1)
        out = mod._maxsim_score_mfa(
            speech, text, ts, te, ws, we,
            selection_mode=mode, lse_temperature=LSE_TEMPERATURE,
        )
        loss = out.sum()
        loss.backward()
        grad_norm = speech.grad.norm().item()
        assert grad_norm > 0.0, f"mode={mode} zero gradient on speech_embs"
        print(f"[OK] grad flow mode={mode}: ||speech.grad||={grad_norm:.4f}")


def test_invalid_mode(mod):
    speech, text, ts, te, ws, we = _build_fixture(SMOKE_SEED)
    try:
        mod._maxsim_score_mfa(
            speech, text, ts, te, ws, we, selection_mode="not_a_mode",
        )
    except AssertionError as exc:
        print(f"[OK] invalid mode rejected: {exc}")
        return
    raise RuntimeError("Expected AssertionError for invalid selection_mode")


def main():
    mod = _load_module()
    print("=" * 60)
    print("MFA window selection smoke tests")
    print("=" * 60)
    test_hard_max(mod)
    print("-" * 60)
    test_smallest(mod)
    print("-" * 60)
    test_logsumexp(mod)
    print("-" * 60)
    test_gradient_flow(mod)
    print("-" * 60)
    test_invalid_mode(mod)
    print("=" * 60)
    print("ALL MFA SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
