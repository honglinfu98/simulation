"""Exact Ogata-thinning simulation of a trained NMH model.

NMH is an exact multivariate Hawkes: lambda_k(t) = softplus(mu_k + sum_{m,j}
A_{k,(m,j)} S^m_j(t)), S^m_j the per-timescale decayed type counts.  The neural
harness (tfow_stylized_facts) simulates every model through an autoregressive
grid-based dt sampler whose quantization inflates the rate; Compound Hawkes by
contrast is simulated by exact thinning.  To compare NMH to Compound Hawkes on
equal footing -- and to see NMH's TRUE generative behaviour unconfounded by the
grid sampler -- we extract (mu, A, delta) from the checkpoint and thinning-
simulate exactly, then score the same Cont stylized-facts battery.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
import torch

from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp
from volume_set_mtpp.training.bfnx_data_loader import _fixed_bfnx_event_names
from .tfow_world_model_diagnostics import save_json
from .tfow_price_facts_v2 import parse_v2_file
from .tfow_stylized_facts import bucketize, all_facts, build_sign_vectors


def softplus(x):
    return np.logaddexp(0.0, x)


def load_real(files, name_to_idx, K, max_per_file, max_events):
    types, dts = [], []
    for fp in files:
        s = parse_v2_file(fp, name_to_idx, K, max_per_file)
        m = s["marks"]; ev = m.sum(1) > 0
        types.append(m.argmax(1)[ev]); dts.append(s["dt"][ev])
    t = np.concatenate(types); d = np.concatenate(dts).astype(np.float64)
    return t[:max_events], d[:max_events]


def simulate_thinning(mu, A, delta, K, M, duration, n_seq, seed, max_events=60000, ub_safety=1.2):
    """Ogata thinning for lambda_k = softplus(mu_k + sum_{m,j} A[k,m,j] S^m_j)."""
    rng = np.random.default_rng(seed)
    A = A.reshape(K, M, K)                      # [target k, timescale m, source j]
    streams = []
    for _ in range(n_seq):
        S = np.zeros((M, K))
        t = 0.0; last = 0.0
        ev_types, ev_dt = [], []
        # intensity helper
        def lam():
            z = mu + np.einsum("mkj,mj->k", A.transpose(1, 0, 2), S)  # sum_{m,j} A[k,m,j] S[m,j]
            return softplus(z)
        while t < duration and len(ev_types) < max_events:
            l = lam(); L = l.sum()
            if L <= 0:
                break
            lam_bar = L * ub_safety                 # upper bound over the next (decaying) interval
            w = rng.exponential(1.0 / lam_bar)
            t += w
            S *= np.exp(-delta[:, None] * w)        # decay all timescales
            l2 = lam(); L2 = l2.sum()
            if rng.random() <= L2 / lam_bar:
                k = rng.choice(K, p=l2 / L2)
                ev_types.append(k); ev_dt.append(t - last); last = t
                S[:, k] += 1.0                      # kick every timescale at type k
        streams.append((np.asarray(ev_types), np.asarray(ev_dt)))
    return streams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--v2-dir", required=True)
    ap.add_argument("--pattern", default="events_gmni_ethusdt_*.jsonl.gz")
    ap.add_argument("--max-events-per-file", type=int, default=150000)
    ap.add_argument("--real-events", type=int, default=150000)
    ap.add_argument("--rollout-duration", type=float, default=600.0)
    ap.add_argument("--rollout-sequences", type=int, default=32)
    ap.add_argument("--rollout-seed", type=int, default=1)
    ap.add_argument("--bucket-seconds", type=float, default=1.0)
    ap.add_argument("--label", default="nmh_thinning")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    names = _fixed_bfnx_event_names(); K = len(names)
    name_to_idx = {n: i for i, n in enumerate(names)}
    idx_to_event = {str(i): n for i, n in enumerate(names)}
    sign, moving = build_sign_vectors(idx_to_event, K)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = create_volume_set_mtpp(cfg.get("num_channels", K), cfg, torch.device("cpu"),
                                   use_volume=cfg.get("use_volume", False), intensity_type="dynamic")
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    dec = model.decoder
    M = dec.num_timescales
    mu = dec.mu.detach().numpy().astype(np.float64)
    A = dec.A.weight.detach().numpy().astype(np.float64)      # [K, M*K]
    delta = dec._deltas().detach().numpy().astype(np.float64) # [M]
    rho = dec.closed_form_rho()
    print(f"NMH params: K={K} M={M} rho={rho:.4f} deltas={[round(x,3) for x in delta.tolist()]}", flush=True)

    files = sorted(sum((glob.glob(str(Path(args.v2_dir) / p)) for p in args.pattern.split(",")), []))
    rtypes, rdt = load_real(files, name_to_idx, K, args.max_events_per_file, args.real_events)
    rmk = np.zeros((len(rtypes), K), dtype=bool); rmk[np.arange(len(rtypes)), rtypes] = True
    r_real, a_real = bucketize(rmk, rdt.astype(np.float32), sign, moving, args.bucket_seconds)
    facts_real = all_facts(r_real, a_real)

    streams = simulate_thinning(mu, A, delta, K, M, args.rollout_duration,
                                args.rollout_sequences, args.rollout_seed)
    rs, As = [], []
    tot_ev = 0
    for ctypes, cdt in streams:
        tot_ev += len(ctypes)
        if len(ctypes) < 10:
            continue
        mk = np.zeros((len(ctypes), K), dtype=bool); mk[np.arange(len(ctypes)), ctypes] = True
        r, a = bucketize(mk, cdt.astype(np.float32), sign, moving, args.bucket_seconds)
        rs.append(r); As.append(a)
    r_sim = np.concatenate(rs); a_sim = np.concatenate(As)
    facts_sim = all_facts(r_sim, a_sim)
    print(f"SIM total_events={tot_ev} mean_rate={tot_ev/(args.rollout_sequences*args.rollout_duration):.1f}/s", flush=True)
    print("REAL Fano", [round(x, 2) for x in facts_real["f5_fano_vs_scale"]], flush=True)
    print("NMHt Fano", [round(x, 2) for x in facts_sim["f5_fano_vs_scale"]], flush=True)
    print("NMHt f6", round(facts_sim["f6_mean_acf_abs_1_10"], 3), "f8", round(facts_sim["f8_powerlaw_exponent"], 2),
          "kurt", round(facts_sim["f2_excess_kurtosis"], 1), "skew", round(facts_sim["f3_skewness"], 2), flush=True)

    save_json(out / f"stylized_facts_{args.label}.json", {
        "label": args.label, "method": "exact Ogata thinning of trained NMH (softplus multivariate Hawkes)",
        "branching_ratio_rho": rho, "deltas": delta.tolist(),
        "genuine_mark_accuracy": ck.get("genuine_acc"),
        "rollout_duration": args.rollout_duration, "bucket_seconds": args.bucket_seconds,
        "facts_real": facts_real, "facts_model": facts_sim,
    })
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
