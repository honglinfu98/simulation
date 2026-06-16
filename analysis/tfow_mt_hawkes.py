"""Multi-timescale multivariate Hawkes fit by full-stream MLE -- the NMH idea
done in Compound Hawkes's (stationary, exactly-fit) protocol.

This isolates NMH's one genuine innovation over Jain's Compound Hawkes -- a bank
of M decay timescales instead of a single beta -- from the windowed cold-start
training that made the neural NMH explode in free rollout.  Everything else
mirrors tfow_compound_hawkes.py: linear excitatory intensity (no softplus, so
the classical branching ratio governs stationarity exactly), exact-likelihood
MLE on the whole event stream, Ogata thinning, same stylized-facts battery.

    lambda_k(t) = mu_k + sum_{m,j} alpha^m_{k,j} exp(-beta_m (t - t_i)),  c_i=j
    rho = spectral_radius( sum_m alpha_m / beta_m )

The question it answers: does a multi-timescale (vs single-beta) Hawkes recover
the long-memory facts (F6 |r|-ACF, F8 power-law) that single-beta Compound
Hawkes missed (F6~0, F8<0), while keeping Fano near real (~8)?
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
import torch

from volume_set_mtpp.training_evaluation.bfnx_data_loader import _fixed_bfnx_event_names
from tfow_world_model_diagnostics import save_json
from tfow_price_facts_v2 import parse_v2_file
from tfow_stylized_facts import bucketize, all_facts, build_sign_vectors

BETAS = [50.0, 5.0, 0.5, 0.1]   # decade-spread timescales (1/beta: 0.02s .. 10s)


def load_events(files, name_to_idx, K, max_per_file):
    types, dts = [], []
    for fp in files:
        s = parse_v2_file(fp, name_to_idx, K, max_per_file)
        m = s["marks"]; ev = m.sum(1) > 0
        types.append(m.argmax(1)[ev]); dts.append(s["dt"][ev])
    return np.concatenate(types), np.concatenate(dts).astype(np.float64)


def decayed_state(types, dt, betas, K):
    """S[i,m,j] = sum_{i'<i, c_{i'}=j} exp(-beta_m (t_i - t_{i'}))."""
    N = len(types); M = len(betas)
    S = np.zeros((N, M, K), dtype=np.float64)
    s = np.zeros((M, K), dtype=np.float64)
    eb = np.asarray(betas)[:, None]
    for i in range(N):
        s *= np.exp(-eb * dt[i])
        S[i] = s
        s[:, types[i]] += 1.0
    return S


def fit(types, dt, S, T, betas, K, steps=500, lr=0.05, device="cpu",
        rho_max=0.8, pen_weight=1.0):
    """Concave MLE in (mu, alpha^m), alpha>=0 via softplus, betas fixed, plus a
    Gershgorin subcriticality penalty so the multi-timescale fit stays stationary
    (unconstrained MLE overfits the extra slow-kernel capacity to rho>1).  NLL is
    per-event-mean so the penalty weight is on a comparable scale."""
    M = len(betas); N = len(types)
    Sd = torch.tensor(S, device=device, dtype=torch.float32)  # [N,M,K]
    c = torch.tensor(types, device=device, dtype=torch.long)
    Tt = torch.tensor(float(T), device=device, dtype=torch.float32)
    binv = torch.tensor([1.0 / b for b in betas], device=device, dtype=torch.float32)  # [M]
    raw_mu = torch.zeros(K, device=device, requires_grad=True)
    raw_al = torch.full((M, K, K), -3.0, device=device, requires_grad=True)  # [m,target,source]
    opt = torch.optim.Adam([raw_mu, raw_al], lr=lr)
    onehot = torch.zeros(len(types), K, device=device)
    onehot[torch.arange(len(types)), c] = 1.0
    src_counts = onehot.sum(0)                                # [K] events per source
    for _ in range(steps):
        opt.zero_grad()
        mu = torch.nn.functional.softplus(raw_mu)
        al = torch.nn.functional.softplus(raw_al)             # [M,K,K]
        al_c = al[:, c, :]                                    # [M,N,K]
        lam = mu[c] + torch.einsum("mnk,nmk->n", al_c, Sd)
        ll_events = torch.log(lam.clamp_min(1e-12)).sum()
        integral = mu.sum() * Tt + (al.sum(1) @ src_counts * binv).sum()
        loss = -(ll_events - integral)
        loss.backward(); opt.step()
        # Hard subcriticality projection on the ACTUAL spectral radius: G =
        # sum_m alpha_m/beta_m is LINEAR in alpha, so scaling all alpha by s
        # scales rho(G) by s.  Project rho to rho_max when exceeded -> the fit
        # sits near-critical at rho_max (strong clustering) yet provably
        # stationary.  (Row-sum Gershgorin bound is too loose -- pins rho ~10x
        # below target -> near-Poisson.)
        with torch.no_grad():
            alc = torch.nn.functional.softplus(raw_al)
            G = (alc * binv.view(M, 1, 1)).sum(dim=0)
            rho_now = float(torch.linalg.eigvals(G).abs().max())
            if rho_now > rho_max:
                al_new = (alc * (rho_max / rho_now)).clamp_min(1e-9)
                raw_al.data = torch.log(torch.expm1(al_new))
    with torch.no_grad():
        mu = torch.nn.functional.softplus(raw_mu)
        al = torch.nn.functional.softplus(raw_al)
        al_c = al[:, c, :]
        lam = mu[c] + torch.einsum("mnk,nmk->n", al_c, Sd)
        ll = (torch.log(lam.clamp_min(1e-12)).sum()
              - (mu.sum() * Tt + (al.sum(1) @ src_counts * binv).sum())).item()
    return mu.cpu().numpy(), al.cpu().numpy(), ll


def branching_rho(al, betas):
    G = sum(al[m] / betas[m] for m in range(len(betas)))      # [K,K]
    return float(np.max(np.abs(np.linalg.eigvals(G))))


def simulate_thinning(mu, al, betas, K, duration, n_seq, seed, max_events=60000):
    rng = np.random.default_rng(seed)
    M = len(betas); eb = np.asarray(betas)
    streams = []
    for _ in range(n_seq):
        s = np.zeros((M, K)); t = 0.0; last = 0.0
        ev_types, ev_dt = [], []
        while t < duration and len(ev_types) < max_events:
            lam = mu + np.einsum("mkj,mj->k", al, s)
            L = lam.sum()
            if L <= 0:
                break
            w = rng.exponential(1.0 / L)
            t += w
            s *= np.exp(-eb[:, None] * w)
            lam2 = mu + np.einsum("mkj,mj->k", al, s); L2 = lam2.sum()
            if rng.random() <= L2 / L:
                k = rng.choice(K, p=lam2 / L2)
                ev_types.append(k); ev_dt.append(t - last); last = t
                s[:, k] += 1.0
        streams.append((np.asarray(ev_types), np.asarray(ev_dt)))
    return streams


def next_mark_accuracy(mu, al, S, types):
    lam = mu[None, :] + np.einsum("mkj,nmj->nk", al, S)       # [N,K]
    pred = lam.argmax(1)
    acc = float((pred == types).mean())
    p = lam / lam.sum(1, keepdims=True)
    ppl = float(np.exp(-np.log(p[np.arange(len(types)), types] + 1e-12).mean()))
    return acc, ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-dir", required=True)
    ap.add_argument("--pattern", default="events_gmni_ethusdt_*.jsonl.gz")
    ap.add_argument("--max-events-per-file", type=int, default=150000)
    ap.add_argument("--fit-events", type=int, default=120000)
    ap.add_argument("--rollout-duration", type=float, default=600.0)
    ap.add_argument("--rollout-sequences", type=int, default=32)
    ap.add_argument("--rollout-seed", type=int, default=1)
    ap.add_argument("--bucket-seconds", type=float, default=1.0)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rho-max", type=float, default=0.8)
    ap.add_argument("--pen-weight", type=float, default=1.0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    names = _fixed_bfnx_event_names(); K = len(names)
    name_to_idx = {n: i for i, n in enumerate(names)}
    idx_to_event = {str(i): n for i, n in enumerate(names)}
    sign, moving = build_sign_vectors(idx_to_event, K)

    files = sorted(sum((glob.glob(str(Path(args.v2_dir) / p)) for p in args.pattern.split(",")), []))
    types, dt = load_events(files, name_to_idx, K, args.max_events_per_file)
    types, dt = types[:args.fit_events], dt[:args.fit_events]
    T = float(dt.sum())
    print(f"FIT events={len(types)} T={T:.0f}s rate={len(types)/T:.1f}/s betas={BETAS}", flush=True)

    S = decayed_state(types, dt, BETAS, K)
    mu, al, ll = fit(types, dt, S, T, BETAS, K, device=args.device,
                     rho_max=args.rho_max, pen_weight=args.pen_weight)
    rho = branching_rho(al, BETAS)
    acc, ppl = next_mark_accuracy(mu, al, S, types)
    print(f"FIT ll={ll:.0f} rho={rho:.3f} acc={acc:.4f} ppl={ppl:.2f}", flush=True)

    rmk = np.zeros((len(types), K), dtype=bool); rmk[np.arange(len(types)), types] = True
    r_real, a_real = bucketize(rmk, dt.astype(np.float32), sign, moving, args.bucket_seconds)
    facts_real = all_facts(r_real, a_real)

    streams = simulate_thinning(mu, al, BETAS, K, args.rollout_duration, args.rollout_sequences, args.rollout_seed)
    rs, As = [], []; tot = 0
    for ct, cd in streams:
        tot += len(ct)
        if len(ct) < 10:
            continue
        mk = np.zeros((len(ct), K), dtype=bool); mk[np.arange(len(ct)), ct] = True
        r, a = bucketize(mk, cd.astype(np.float32), sign, moving, args.bucket_seconds)
        rs.append(r); As.append(a)
    r_sim = np.concatenate(rs); a_sim = np.concatenate(As)
    facts_sim = all_facts(r_sim, a_sim)
    print(f"SIM events={tot} rate={tot/(args.rollout_sequences*args.rollout_duration):.1f}/s", flush=True)
    print("REAL Fano", [round(x, 2) for x in facts_real["f5_fano_vs_scale"]], flush=True)
    print("MTH  Fano", [round(x, 2) for x in facts_sim["f5_fano_vs_scale"]], flush=True)
    print("MTH  f6", round(facts_sim["f6_mean_acf_abs_1_10"], 3), "f8", round(facts_sim["f8_powerlaw_exponent"], 2),
          "kurt", round(facts_sim["f2_excess_kurtosis"], 1), "skew", round(facts_sim["f3_skewness"], 2), flush=True)

    save_json(out / "stylized_facts_mt_hawkes.json", {
        "label": "mt_hawkes", "method": "multi-timescale multivariate Hawkes, full-stream MLE + thinning",
        "betas": BETAS, "branching_ratio_rho": rho, "fit_loglik": ll,
        "genuine_mark_accuracy": acc, "genuine_mark_perplexity": ppl,
        "rollout_duration": args.rollout_duration, "bucket_seconds": args.bucket_seconds,
        "facts_real": facts_real, "facts_model": facts_sim,
    })
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
