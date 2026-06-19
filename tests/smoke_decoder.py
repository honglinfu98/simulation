"""Standardized decoder smoke test — every decoder must pass this before training.

Synthetic data only (no cluster / no real data needed). Checks the interface contract:
state shapes, the anti-leakage rule, intensity positivity + finiteness, gradient flow to
all parameters, and the branching certificate (if exposed).

    ./setup_repo.sh  &&  . venv/bin/activate  &&  pytest tests/smoke_decoder.py
    # or, without the venv:  PYTHONPATH=. python3 tests/smoke_decoder.py
"""
import sys
import torch
import torch.nn as nn

from volume_set_mtpp.models.nmh_decoder import NMHDecoder
from volume_set_mtpp.models.ptp_s2p2_decoder import PerTypeS2P2Decoder
from volume_set_mtpp.models.lgm_decoder import LGMDecoder
from volume_set_mtpp.models.gmh_decoder import GMHDecoder

K, B, N = 62, 4, 50


def synth():
    torch.manual_seed(0)
    marks = torch.zeros(B, N, K)
    for b in range(B):
        for i in range(N):
            if torch.rand(1).item() > 0.33:
                marks[b, i, torch.randint(0, K, (1,)).item()] = 1.0
    dt = torch.rand(B, N) * 0.3
    ts = torch.cumsum(dt, dim=1)
    return marks, ts


def intensity_from_state(dec, h):
    """Per-type intensity from a head-facing state, handling both decoder styles."""
    if hasattr(dec, "is_lgm") and dec.is_lgm:
        gd = dec.ground_dim
        lam = dec.ground_intensity(h[..., :gd]).unsqueeze(-1) * torch.softmax(dec.mark_score(h[..., gd:]), -1)
        return lam
    return dec.type_intensities(h)


def check(name, dec):
    marks, ts = synth()
    right, left = dec.get_states_and_event_left_states(marks, ts)
    D = dec.recurrent_hidden_size
    # `right` may be a decoder-internal PACKED state (dim >= D) that is consumed only by
    # get_hidden_h; `left` and get_hidden_h outputs are the head-facing dim D.
    assert right.shape[:2] == (B, N + 1), f"{name}: right shape {tuple(right.shape)}"
    assert left.shape == (B, N, D), f"{name}: left shape {tuple(left.shape)} != {(B, N, D)}"
    assert torch.allclose(left[:, 0], torch.zeros_like(left[:, 0])), f"{name}: anti-leakage violated (left[:,0]!=0)"

    lam_ev = intensity_from_state(dec, left)
    assert lam_ev.shape == (B, N, K), f"{name}: lambda(events) shape {tuple(lam_ev.shape)}"
    assert torch.isfinite(lam_ev).all(), f"{name}: non-finite intensity"
    assert (lam_ev > 0).all(), f"{name}: non-positive intensity"

    q = ts[:, -1:] + torch.tensor([[0.02, 0.2]])
    h = dec.get_hidden_h(right, ts, q)
    assert h.shape == (B, 2, D), f"{name}: get_hidden_h shape {tuple(h.shape)}"
    lam_q = intensity_from_state(dec, h)
    assert torch.isfinite(lam_q).all()

    # gradient flow to every trainable parameter
    loss = -(torch.log(lam_ev.sum(-1) + 1e-8)).mean() + lam_q.sum(-1).mean()
    loss.backward()
    n_grad = sum(1 for p in dec.parameters() if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
    n_par = sum(1 for p in dec.parameters() if p.requires_grad)
    assert n_grad >= 1, f"{name}: no parameter received gradient"

    rho = dec.closed_form_rho() if hasattr(dec, "closed_form_rho") else None
    rho_s = f"rho={rho:.3f}" if rho is not None else "rho=n/a"
    print(f"  PASS  {name:9s} | state_dim={D:4d} params={n_par:3d} grad={n_grad:3d} | "
          f"lambda[{lam_ev.min():.3f},{lam_ev.max():.2f}] | {rho_s}")


def build():
    emb = nn.Embedding(K, 64)
    return [
        ("nmh", NMHDecoder(num_channels=K, num_timescales=4)),
        ("ptp", PerTypeS2P2Decoder(channel_embedding=emb, num_channels=K, per_type_dim=8)),
        ("gmh", GMHDecoder(channel_embedding=emb, time_embedding=None, num_channels=K, num_timescales=4)),
        ("lgm", LGMDecoder(channel_embedding=emb, num_channels=K, per_type_dim=8, num_timescales=4, target_rate=2.0)),
    ]


if __name__ == "__main__":
    print("Decoder smoke test (synthetic data):")
    fails = 0
    for name, dec in build():
        try:
            check(name, dec)
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {name}: {e}")
    print("ALL PASS" if fails == 0 else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
