"""Compound Hawkes Process baseline (Jain et al. 2024, "Limit Order Book
dynamics and order size modelling using Compound Hawkes Process").

A multivariate exponential-kernel Hawkes process over the 62 order-flow event
types, compounded with per-type log-normal order sizes:

    lambda_k(t) = mu_k + sum_{t_i < t} alpha_{k, c_i} exp(-beta (t - t_i))
    size | type k  ~  LogNormal(m_k, s_k)

- alpha >= 0 (excitatory; enables Ogata thinning and a well-defined branching
  ratio).  Fit by MLE on the exact Hawkes log-likelihood.  beta is a shared
  scalar selected by profile likelihood over a small grid.
- branching ratio rho = spectral_radius(alpha / beta) -- the *classical*
  stationarity quantity our neural stability work generalizes; reported here.
- Simulated by Ogata thinning, then scored on the same Cont stylized-facts
  battery as the neural models for an apples-to-apples event-driven comparison.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from volume_set_mtpp.training_evaluation.bfnx_data_loader import _fixed_bfnx_event_names
from tfow_world_model_diagnostics import save_json
from tfow_price_facts_v2 import parse_v2_file
from tfow_stylized_facts import bucketize, all_facts, build_sign_vectors


def load_events(files, name_to_idx, k, max_per_file):
    """Concatenate single-mark event streams -> (types [N], dt [N], logvol [N])."""
    types, dts, logv = [], [], []
    for fp in files:
        s = parse_v2_file(fp, name_to_idx, k, max_per_file)
        m = s["marks"]
        ev = m.sum(1) > 0
        c = m.argmax(1)[ev]
        types.append(c)
        dts.append(s["dt"][ev])
        vr = s["vols_raw"][ev]
        logv.append(np.log1p(vr[np.arange(len(c)), c]))
    return (np.concatenate(types), np.concatenate(dts).astype(np.float64),
            np.concatenate(logv))


def decayed_state(types, dt, beta, K):
    """S[i,j] = sum_{i'<i, c_{i'}=j} exp(-beta (t_i - t_{i'})), recursively."""
    N = len(types)
    S = np.zeros((N, K), dtype=np.float64)
    s = np.zeros(K, dtype=np.float64)
    for i in range(N):
        s *= np.exp(-beta * dt[i])      # decay to t_i (dt[i] = t_i - t_{i-1})
        S[i] = s
        s[types[i]] += 1.0              # add this event for the future
    return S


def fit_mu_alpha(types, dt, S, T, K, steps=400, lr=0.05, device="cpu"):
    """Concave MLE in (mu, alpha) for fixed beta.  alpha>=0 via softplus."""
    Sd = torch.tensor(S, device=device)
    c = torch.tensor(types, device=device, dtype=torch.long)
    Tt = torch.tensor(T, device=device)
    raw_mu = torch.zeros(K, device=device, requires_grad=True)
    raw_al = torch.full((K, K), -3.0, device=device, requires_grad=True)
    # integral term weights: per event i, (1 - exp(-beta (T - t_i)))/beta summed
    # over targets k of alpha_{k,c_i}; here we use the simple T-scaled bound that
    # ignores edge truncation (T >> 1/beta), i.e. integral ~ sum_i alpha_{.,c_i}/beta.
    opt = torch.optim.Adam([raw_mu, raw_al], lr=lr)
    onehot = torch.zeros(len(types), K, device=device)
    onehot[torch.arange(len(types)), c] = 1.0
    src_counts = onehot.sum(0)  # number of events per source type
    for _ in range(steps):
        opt.zero_grad()
        mu = torch.nn.functional.softplus(raw_mu)
        al = torch.nn.functional.softplus(raw_al)            # [K(target),K(source)]
        lam = mu[c] + (Sd * al[c]).sum(1)                    # lambda at each event
        ll_events = torch.log(lam.clamp_min(1e-12)).sum()
        integral = mu.sum() * Tt + (al.sum(0) * src_counts).sum() / BETA_T
        loss = -(ll_events - integral)
        loss.backward()
        opt.step()
    with torch.no_grad():
        mu = torch.nn.functional.softplus(raw_mu)
        al = torch.nn.functional.softplus(raw_al)
        lam = mu[c] + (Sd * al[c]).sum(1)
        ll = (torch.log(lam.clamp_min(1e-12)).sum()
              - (mu.sum() * Tt + (al.sum(0) * src_counts).sum() / BETA_T)).item()
    return mu.detach().cpu().numpy(), al.detach().cpu().numpy(), ll


def simulate_thinning(mu, alpha, beta, duration, n_seq, seed, max_events=60000):
    """Ogata thinning for an excitatory multivariate exp-Hawkes."""
    rng = np.random.default_rng(seed)
    K = len(mu)
    streams = []
    for _ in range(n_seq):
        s = np.zeros(K)
        t = 0.0
        ev_types, ev_dt = [], []
        last = 0.0
        while t < duration and len(ev_types) < max_events:
            lam = mu + alpha @ s
            lam_sum = lam.sum()
            if lam_sum <= 0:
                break
            w = rng.exponential(1.0 / lam_sum)   # upper bound = current total (decreasing)
            t += w
            s *= np.exp(-beta * w)
            lam2 = mu + alpha @ s
            if rng.random() <= lam2.sum() / lam_sum:
                p = lam2 / lam2.sum()
                k = rng.choice(K, p=p)
                ev_types.append(k); ev_dt.append(t - last); last = t
                s[k] += 1.0
        streams.append((np.asarray(ev_types), np.asarray(ev_dt)))
    return streams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-dir", required=True)
    ap.add_argument("--pattern", default="events_gmni_ethusdt_*.jsonl.gz")
    ap.add_argument("--max-events-per-file", type=int, default=60000)
    ap.add_argument("--fit-events", type=int, default=150000)
    ap.add_argument("--rollout-duration", type=float, default=600.0)
    ap.add_argument("--rollout-sequences", type=int, default=32)
    ap.add_argument("--rollout-seed", type=int, default=1)
    ap.add_argument("--bucket-seconds", type=float, default=1.0)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    names = _fixed_bfnx_event_names()
    K = len(names)
    name_to_idx = {n: i for i, n in enumerate(names)}
    idx_to_event = {str(i): n for i, n in enumerate(names)}
    sign, moving = build_sign_vectors(idx_to_event, K)

    files = sorted(sum((glob.glob(str(Path(args.v2_dir) / p)) for p in args.pattern.split(",")), []))
    print("FILES", len(files), flush=True)
    types, dt, logv = load_events(files, name_to_idx, K, args.max_events_per_file)
    types, dt, logv = types[:args.fit_events], dt[:args.fit_events], logv[:args.fit_events]
    T = float(dt.sum())
    print(f"FIT events={len(types)} T={T:.0f}s rate={len(types)/T:.1f}/s", flush=True)

    # per-type log-normal sizes (the "compound" marks)
    vm = np.zeros(K); vs = np.ones(K)
    for k in range(K):
        lk = logv[types == k]
        if len(lk) > 5:
            vm[k] = lk.mean(); vs[k] = lk.std() + 1e-3

    # profile-likelihood over a small shared-beta grid
    global BETA_T
    best = None
    for beta in [2.0, 5.0, 10.0, 20.0, 50.0]:
        BETA_T = beta
        S = decayed_state(types, dt, beta, K)
        mu, al, ll = fit_mu_alpha(types, dt, S, T, K, device=args.device)
        rho = float(np.max(np.abs(np.linalg.eigvals(al / beta))))
        print(f"beta={beta:>5} ll={ll:>12.0f} rho={rho:.3f}", flush=True)
        if best is None or ll > best[0]:
            best = (ll, beta, mu, al, rho)
    ll, beta, mu, al, rho = best
    print(f"BEST beta={beta} branching_ratio_rho={rho:.3f}", flush=True)

    # next-mark prediction accuracy + perplexity (argmax of fitted intensity at
    # each event; in-stream, comparable in spirit to the neural genuine-event
    # accuracy though over the raw stream rather than the windowed loader).
    import math as _m
    S = decayed_state(types, dt, beta, K)
    lam = mu[None, :] + S @ al.T                     # [N,K] intensity per type at each event
    pred = lam.argmax(1)
    acc = float((pred == types).mean())
    p = lam / lam.sum(1, keepdims=True)
    ppl = float(_m.exp(-np.log(p[np.arange(len(types)), types] + 1e-12).mean()))
    print(f"CHP_ACCURACY genuine_acc={acc:.4f} perplexity={ppl:.2f} n={len(types)}", flush=True)

    # simulate + score on the stylized-facts battery
    cum = np.cumsum(dt)
    rmk = np.zeros((len(types), K), dtype=bool); rmk[np.arange(len(types)), types] = True
    r_real, a_real = bucketize(rmk, dt.astype(np.float32), sign, moving, args.bucket_seconds)
    facts_real = all_facts(r_real, a_real)

    streams = simulate_thinning(mu, al, beta, args.rollout_duration,
                                args.rollout_sequences, args.rollout_seed)
    rs, As = [], []
    for ctypes, cdt in streams:
        if len(ctypes) < 10:
            continue
        mk = np.zeros((len(ctypes), K), dtype=bool); mk[np.arange(len(ctypes)), ctypes] = True
        r, a = bucketize(mk, cdt.astype(np.float32), sign, moving, args.bucket_seconds)
        rs.append(r); As.append(a)
    r_sim = np.concatenate(rs); a_sim = np.concatenate(As)
    facts_sim = all_facts(r_sim, a_sim)

    save_json(out / "stylized_facts_compound_hawkes.json", {
        "label": "compound_hawkes", "method": "Jain et al. 2024 multivariate exp-Hawkes + lognormal sizes",
        "beta": beta, "branching_ratio_rho": rho, "fit_loglik": ll,
        "n_fit_events": int(len(types)), "rollout_duration": args.rollout_duration,
        "bucket_seconds": args.bucket_seconds,
        "facts_real": facts_real, "facts_model": facts_sim,
    })
    print("DONE rho=%.3f sim_buckets=%d" % (rho, len(r_sim)), flush=True)


if __name__ == "__main__":
    main()
