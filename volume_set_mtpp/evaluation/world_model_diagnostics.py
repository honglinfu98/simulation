#!/usr/bin/env python3
"""TFOW/VolumeSetMTPP world-model diagnostics and conference-style plots.

Produces Event Stream Model diagnostics for a selected checkpoint:
- event-time calibration / compensator-based PIT
- set-size distribution (target-free predicted sets)
- event co-occurrence matrix: real vs predicted sets
- volume distribution (when the checkpoint provides a volume prediction)
- intensity / survival calibration from the model's own survival function
- FREE-RUNNING autoregressive rollout vs real event streams + horizon decay

Methodology notes (v2, 2026-06-10):
- PIT is computed from the compensator: u_i = 1 - exp(-Lambda(dt_i)) with
  Lambda(dt) = int_0^dt lambda(s) ds evaluated by trapezoid quadrature.  The
  previous version used 1 - exp(-lambda(t_true) * dt_true), which is not a PIT.
- pred_dt is the target-free E[dt] = int_0^inf S(t) dt via survival quadrature.
  The previous version used 1/lambda(t_true), which leaks the answer.
- Predicted sets are thresholded at the model-predicted time (target-free),
  using the validation-calibrated threshold passed via --threshold.
- Plot 06 is a true closed-loop rollout: the model's own sampled events are
  fed back as history.  The previous version cumulated teacher-forced
  one-step predictions.

The script intentionally avoids pandas/seaborn so it works on the UCL shared stack.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from volume_set_mtpp.training.data_loader import create_bfnx_dataloaders
from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp


def get_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def move_batch(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def limit_batches(loader: Iterable, max_batches: int | None):
    for i, batch in enumerate(loader):
        if max_batches is not None and max_batches > 0 and i >= max_batches:
            break
        yield batch


def qstats(x: np.ndarray) -> Dict[str, float | int | None]:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p01": float(np.quantile(x, 0.01)),
        "p05": float(np.quantile(x, 0.05)),
        "p25": float(np.quantile(x, 0.25)),
        "p50": float(np.quantile(x, 0.50)),
        "p75": float(np.quantile(x, 0.75)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
    }


def save_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def save_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def style_ax(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linewidth=0.6)


def savefig(out: Path, name: str):
    for ext in ("png", "svg"):
        plt.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=220)
    plt.close()


def get_target_volumes(batch: Dict) -> torch.Tensor | None:
    for k in ("target_volumes", "target_volume", "volumes"):
        if k in batch and torch.is_tensor(batch[k]):
            v = batch[k]
            if v.ndim >= 2:
                return v.float()
    return None


# ---------------------------------------------------------------------------
# Model query helpers (compensator quadrature)
# ---------------------------------------------------------------------------

def _distribution_at_dts(model, states: torch.Tensor, timestamps: torch.Tensor, query_dts: torch.Tensor, state_feats=None, pot_feats=None) -> Dict[str, torch.Tensor]:
    """Evaluate the model at history-relative offsets query_dts [B,Q].

    Honors `model._sim_rate_k` (simulation-time rate calibration): total and
    per-channel intensities are scaled by k, leaving the mark distribution
    (item_probability / item_logits) untouched -- uniform intensity scaling is
    mark-preserving for every decoder in this harness. Evaluation code that
    never sets the attribute is unaffected (k defaults to 1)."""
    query_timestamps = timestamps[:, -1:] + query_dts.float().clamp_min(0.0)
    h_t = model.decoder.get_hidden_h(state_values=states, state_times=timestamps, timestamps=query_timestamps)
    pf = None
    if pot_feats is not None:
        pf = pot_feats if pot_feats.shape[1] == h_t.shape[1] else pot_feats.expand(-1, h_t.shape[1], -1)
    try:
        d = model.get_total_intensity_and_items(h_t, None, state_features=state_feats, potential_feats=pf)
    except TypeError:
        d = model.get_total_intensity_and_items(h_t)
    k = float(getattr(model, "_sim_rate_k", 1.0))
    if k != 1.0:
        d = dict(d)
        d["total_intensity"] = d["total_intensity"] * k
        if "channel_intensity" in d:
            d["channel_intensity"] = d["channel_intensity"] * k
    return d


def _total_intensity_at_dts(model, states, timestamps, query_dts, state_feats=None, pot_feats=None) -> torch.Tensor:
    d = _distribution_at_dts(model, states, timestamps, query_dts, state_feats=state_feats, pot_feats=pot_feats)
    return d["total_intensity"].squeeze(-1).clamp_min(1e-8)


def _survival_quadrature(model, states, timestamps, horizon: float, n_grid: int, state_feats=None, pot_feats=None):
    """Return (grid [Q], lambda [B,Q], Lambda [B,Q], S [B,Q], E[dt] [B])."""
    device = timestamps.device
    b = timestamps.shape[0]
    grid = torch.logspace(-4, math.log10(horizon), n_grid, device=device)
    lam = _total_intensity_at_dts(model, states, timestamps, grid.unsqueeze(0).expand(b, -1), state_feats=state_feats, pot_feats=pot_feats)
    widths = (grid[1:] - grid[:-1]).unsqueeze(0)
    seg = 0.5 * (lam[:, 1:] + lam[:, :-1]) * widths
    big_lambda = torch.cat([torch.zeros(b, 1, device=device), torch.cumsum(seg, dim=1)], dim=1)
    surv = torch.exp(-big_lambda)
    expected = grid[0] + (0.5 * (surv[:, 1:] + surv[:, :-1]) * widths).sum(dim=1)
    expected = (expected + surv[:, -1] / lam[:, -1]).clamp(min=1e-6, max=2.0 * horizon)
    return grid, lam, big_lambda, surv, expected


def _median_from_quadrature(grid: torch.Tensor, lam: torch.Tensor, big_lambda: torch.Tensor) -> torch.Tensor:
    """Predictive median: dt with Lambda(dt) = ln 2, linear interp on the grid."""
    b = big_lambda.shape[0]
    ln2 = math.log(2.0)
    tgt = torch.full((b, 1), ln2, device=big_lambda.device)
    idx = torch.searchsorted(big_lambda.contiguous(), tgt).squeeze(1).clamp(1, big_lambda.shape[1] - 1)
    lo, hi = idx - 1, idx
    l_lo = big_lambda.gather(1, lo.unsqueeze(1)).squeeze(1)
    l_hi = big_lambda.gather(1, hi.unsqueeze(1)).squeeze(1)
    frac = ((ln2 - l_lo) / (l_hi - l_lo).clamp_min(1e-12)).clamp(0.0, 1.0)
    median = grid[lo] + frac * (grid[hi] - grid[lo])
    overflow = big_lambda[:, -1] < ln2
    if overflow.any():
        median = torch.where(overflow, grid[-1] + (ln2 - big_lambda[:, -1]) / lam[:, -1].clamp_min(1e-8), median)
    return median.clamp(min=1e-6)


def _compensator_at(model, states, timestamps, dt_limits: torch.Tensor, n_points: int = 33) -> torch.Tensor:
    """Lambda(dt_i) = int_0^{dt_i} lambda(s) ds per sample, trapezoid on a linear grid."""
    device = timestamps.device
    fracs = torch.linspace(0.0, 1.0, n_points, device=device).unsqueeze(0)
    dts = dt_limits.clamp_min(1e-8).unsqueeze(1) * fracs  # [B,P]
    lam = _total_intensity_at_dts(model, states, timestamps, dts)
    widths = dts[:, 1:] - dts[:, :-1]
    return (0.5 * (lam[:, 1:] + lam[:, :-1]) * widths).sum(dim=1)


# ---------------------------------------------------------------------------
# One-step (teacher-forced) diagnostics collection — target-free predictions
# ---------------------------------------------------------------------------

def collect_predictions(model, loader, device, threshold: float, max_batches: int | None,
                        horizon: float, n_grid: int) -> Dict[str, np.ndarray]:
    true_dt=[]; pred_dt=[]; lam0_all=[]; pit=[]; true_sets=[]; pred_sets=[]
    true_vol=[]; pred_vol=[]; active_vol_true=[]; active_vol_pred=[]
    surv_rows=[]
    grid_np = None
    with torch.no_grad():
        for batch in limit_batches(loader, max_batches):
            batch = move_batch(batch, device)
            input_times = batch["input_times"].float()
            input_marks = batch["input_marks"].float()
            timestamps = torch.cumsum(input_times, dim=1)
            states = model.decoder.get_states(input_marks, timestamps)
            target_dt = batch["target_time"].float().reshape(-1)

            # Target-free E[dt] + predictive median + survival curves from the
            # model's own intensity.  pred_dt records the MEDIAN (the readout
            # comparable to the median of realized dts; the mean would carry a
            # mean-vs-median skew artifact into the dt-ratio diagnostic).
            grid, lam, big_lambda, surv, expected_dt = _survival_quadrature(model, states, timestamps, horizon, n_grid)
            median_dt = _median_from_quadrature(grid, lam, big_lambda)
            if grid_np is None:
                grid_np = grid.detach().cpu().numpy()

            # Compensator-based PIT: u = 1 - exp(-Lambda(true dt)).
            comp = _compensator_at(model, states, timestamps, target_dt)
            pit_b = 1.0 - torch.exp(-comp)

            # Target-free set prediction at the model-predicted time.
            d = _distribution_at_dts(model, states, timestamps, expected_dt.unsqueeze(1))
            probs = d["item_probability"].squeeze(1).float()
            pred_set = probs > threshold
            true_set = batch["target_marks"].float() > 0

            vol_t = get_target_volumes(batch)
            vol_p = None
            for k in ("volume_mean", "volume_prediction"):
                if k in d:
                    vol_p = d[k]
                    break
            if vol_p is not None:
                vol_p = vol_p.squeeze(1).float()
                if vol_p.shape == true_set.shape:
                    active_vol_pred.append(vol_p[true_set].detach().cpu().numpy())
                    pred_vol.append(vol_p.detach().cpu().numpy())
            if vol_t is not None and vol_t.shape == true_set.shape:
                active_vol_true.append(vol_t[true_set].detach().cpu().numpy())
                true_vol.append(vol_t.detach().cpu().numpy())

            true_dt.append(target_dt.detach().cpu().numpy())
            pred_dt.append(median_dt.detach().cpu().numpy())
            lam0_all.append(lam[:, 0].detach().cpu().numpy())
            pit.append(pit_b.detach().cpu().numpy())
            true_sets.append(true_set.detach().cpu().numpy().astype(bool))
            pred_sets.append(pred_set.detach().cpu().numpy().astype(bool))
            surv_rows.append(surv.detach().cpu().numpy())

    def cat(xs, axis=0):
        xs = [x for x in xs if x is not None and np.size(x)]
        return np.concatenate(xs, axis=axis) if xs else np.array([])
    return {
        "true_dt": cat(true_dt),
        "pred_dt": cat(pred_dt),
        "total_intensity": cat(lam0_all),
        "pit": cat(pit),
        "true_sets": cat(true_sets),
        "pred_sets": cat(pred_sets),
        "true_volumes": cat(true_vol),
        "pred_volumes": cat(pred_vol),
        "active_true_volumes": cat(active_vol_true),
        "active_pred_volumes": cat(active_vol_pred),
        "survival": cat(surv_rows),
        "survival_grid": grid_np if grid_np is not None else np.array([]),
    }


# ---------------------------------------------------------------------------
# Free-running autoregressive rollout
# ---------------------------------------------------------------------------

def free_running_rollout(model, batch, device, threshold: float, steps: int, n_seq: int,
                         horizon: float, n_grid: int, seed: int = 0) -> Dict[str, np.ndarray]:
    """Closed-loop simulation: sample (dt, event set), feed back, iterate.

    History is kept as a sliding window of the most recent ``seq_length``
    events, matching the context length the model was trained on.  Event times
    are sampled exactly from the ground process by inverting the compensator
    (Lambda(dt) = u with u ~ Exp(1)); sets are sampled from the Bernoulli
    marginals at the sampled time (empty draws fall back to the argmax type).
    """
    torch.manual_seed(seed)
    n = min(n_seq, batch["input_marks"].shape[0])
    marks = batch["input_marks"][:n].float().to(device).clone()
    dts = batch["input_times"][:n].float().clamp_min(0.0).to(device).clone()
    seq_len = marks.shape[1]
    sim_dt = []
    sim_set_size = []
    type_counts = torch.zeros(marks.shape[-1], device=device)
    with torch.no_grad():
        for _ in range(steps):
            timestamps = torch.cumsum(dts, dim=1)
            states = model.decoder.get_states(marks, timestamps)
            grid, lam, big_lambda, _, _ = _survival_quadrature(model, states, timestamps, horizon, n_grid)
            u = -torch.log(torch.rand(n, device=device).clamp_min(1e-12))
            # Invert Lambda(dt) = u by linear interpolation on the quadrature grid.
            idx = torch.searchsorted(big_lambda.contiguous(), u.unsqueeze(1)).squeeze(1).clamp(1, big_lambda.shape[1] - 1)
            lo, hi = idx - 1, idx
            lam_lo = big_lambda.gather(1, lo.unsqueeze(1)).squeeze(1)
            lam_hi = big_lambda.gather(1, hi.unsqueeze(1)).squeeze(1)
            g_lo = grid[lo]
            g_hi = grid[hi]
            frac = ((u - lam_lo) / (lam_hi - lam_lo).clamp_min(1e-12)).clamp(0.0, 1.0)
            new_dt = g_lo + frac * (g_hi - g_lo)
            # Tail: if u exceeds Lambda(horizon), extend with the terminal rate.
            overflow = u > big_lambda[:, -1]
            if overflow.any():
                tail = (u - big_lambda[:, -1]) / lam[:, -1].clamp_min(1e-8)
                new_dt = torch.where(overflow, grid[-1] + tail, new_dt)
            new_dt = new_dt.clamp(min=1e-6, max=4.0 * horizon)

            d = _distribution_at_dts(model, states, timestamps, new_dt.unsqueeze(1))
            probs = d["item_probability"].squeeze(1).float()
            new_set = torch.bernoulli(probs.clamp(0.0, 1.0))
            empty = new_set.sum(dim=1) == 0
            if empty.any():
                top1 = probs.argmax(dim=1)
                new_set[empty] = 0.0
                new_set[empty, top1[empty]] = 1.0

            sim_dt.append(new_dt.detach().cpu().numpy())
            sim_set_size.append(new_set.sum(dim=1).detach().cpu().numpy())
            type_counts += new_set.sum(dim=0)

            # Slide the history window.
            marks = torch.cat([marks[:, 1:, :], new_set.unsqueeze(1)], dim=1)
            dts = torch.cat([dts[:, 1:], new_dt.unsqueeze(1)], dim=1)
            assert marks.shape[1] == seq_len
    return {
        "sim_dt": np.stack(sim_dt, axis=1),               # [n_seq, steps]
        "sim_set_size": np.stack(sim_set_size, axis=1),   # [n_seq, steps]
        "sim_type_marginal": (type_counts / type_counts.sum().clamp_min(1.0)).detach().cpu().numpy(),
    }


def horizon_decay(sim_dt: np.ndarray, sim_sz: np.ndarray, real_dt: np.ndarray, real_sz: np.ndarray,
                  bucket: int = 50) -> Dict[str, List[float]]:
    """L1 histogram distance between rollout and real marginals per step bucket."""
    lo = 0.0
    hi = max(float(np.quantile(np.log1p(real_dt), 0.999)), 1e-3)
    dt_bins = np.linspace(lo, hi, 31)
    ref_dt, _ = np.histogram(np.log1p(real_dt), bins=dt_bins, density=True)
    max_sz = int(max(real_sz.max(initial=1), sim_sz.max(initial=1)))
    sz_bins = np.arange(max_sz + 2) - 0.5
    ref_sz, _ = np.histogram(real_sz, bins=sz_bins, density=True)
    steps = sim_dt.shape[1]
    centers, dt_l1, sz_l1 = [], [], []
    for s in range(0, steps, bucket):
        chunk_dt = np.log1p(sim_dt[:, s:s + bucket].reshape(-1))
        chunk_sz = sim_sz[:, s:s + bucket].reshape(-1)
        if len(chunk_dt) == 0:
            continue
        h_dt, _ = np.histogram(chunk_dt, bins=dt_bins, density=True)
        h_sz, _ = np.histogram(chunk_sz, bins=sz_bins, density=True)
        bw_dt = dt_bins[1] - dt_bins[0]
        centers.append(s + min(bucket, steps - s) / 2)
        dt_l1.append(float(np.sum(np.abs(h_dt - ref_dt)) * bw_dt))
        sz_l1.append(float(np.sum(np.abs(h_sz - ref_sz)) * 1.0))
    return {"step": centers, "dt_hist_l1": dt_l1, "set_size_hist_l1": sz_l1}


# ---------------------------------------------------------------------------
# Metrics + plots
# ---------------------------------------------------------------------------

def set_metrics(true: np.ndarray, pred: np.ndarray) -> Dict:
    inter = (true & pred).sum(axis=1).astype(float)
    union = (true | pred).sum(axis=1).astype(float)
    ps = pred.sum(axis=1).astype(float); ts = true.sum(axis=1).astype(float)
    prec = inter / np.maximum(ps, 1.0); rec = inter / np.maximum(ts, 1.0)
    f1 = (2*prec*rec) / np.maximum(prec+rec, 1e-12)
    return {
        "set_hit_rate": float(np.mean(inter > 0)),
        "exact_set_accuracy": float(np.mean((true == pred).all(axis=1))),
        "sample_precision": float(np.mean(prec)),
        "sample_recall": float(np.mean(rec)),
        "sample_f1": float(np.mean(f1)),
        "jaccard": float(np.mean(inter / np.maximum(union, 1.0))),
        "avg_true_set_size": float(np.mean(ts)),
        "avg_pred_set_size": float(np.mean(ps)),
    }


def plot_time_calibration(data: Dict, out: Path):
    true_dt = data["true_dt"]; pred_dt = data["pred_dt"]; pit = np.clip(data["pit"], 0, 1)
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.6))
    axs[0].scatter(true_dt, pred_dt, s=5, alpha=0.22)
    lim = np.nanpercentile(np.concatenate([true_dt, pred_dt]), 99) if len(true_dt) else 1
    axs[0].plot([0, lim], [0, lim], color="black", lw=1)
    style_ax(axs[0], "Next-event time calibration (target-free)", "true Δt", "model median Δt")
    axs[1].hist(np.log1p(true_dt), bins=60, density=True, alpha=0.55, label="real")
    axs[1].hist(np.log1p(pred_dt), bins=60, density=True, alpha=0.55, label="model")
    axs[1].legend(frameon=False); style_ax(axs[1], "Event-time distribution", "log(1+Δt)", "density")
    axs[2].hist(pit, bins=np.linspace(0,1,21), density=True, alpha=0.8, color="#4C78A8")
    axs[2].axhline(1.0, color="black", lw=1, ls="--")
    style_ax(axs[2], "Compensator PIT: 1-exp(-Λ(Δt))", "PIT", "density")
    savefig(out, "01_event_time_calibration")


def plot_set_size_and_cooccurrence(data: Dict, out: Path, top_k: int = 24):
    true = data["true_sets"]; pred = data["pred_sets"]
    ts = true.sum(axis=1); ps = pred.sum(axis=1)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
    max_size = int(max(ts.max() if len(ts) else 1, ps.max() if len(ps) else 1))
    bins = np.arange(max_size+2)-0.5
    axs[0].hist(ts, bins=bins, density=True, alpha=0.6, label="real")
    axs[0].hist(ps, bins=bins, density=True, alpha=0.6, label="model")
    axs[0].legend(frameon=False); style_ax(axs[0], "Event-set size distribution (target-free)", "|S|", "probability")
    axs[1].scatter(ts, ps, s=5, alpha=0.22)
    axs[1].plot([0, max_size], [0, max_size], color="black", lw=1)
    style_ax(axs[1], "Per-sample set-size calibration", "real |S|", "predicted |S|")
    savefig(out, "02_set_size_distribution")

    freq = true.mean(axis=0)
    idx = np.argsort(-freq)[:min(top_k, true.shape[1])]
    co_true = (true[:, idx].astype(float).T @ true[:, idx].astype(float)) / max(len(true), 1)
    co_pred = (pred[:, idx].astype(float).T @ pred[:, idx].astype(float)) / max(len(pred), 1)
    diff = co_pred - co_true
    fig, axs = plt.subplots(1, 3, figsize=(14.0, 4.0))
    vmax = max(float(co_true.max(initial=0)), float(co_pred.max(initial=0)), 1e-8)
    for ax, mat, title, cmap, vmin, vmax_i in [
        (axs[0], co_true, "Real co-occurrence", "magma", 0, vmax),
        (axs[1], co_pred, "Model co-occurrence", "magma", 0, vmax),
        (axs[2], diff, "Model - real", "coolwarm", -max(abs(diff).max(initial=0),1e-8), max(abs(diff).max(initial=0),1e-8)),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax_i)
        ax.set_title(title, fontsize=11, weight="bold"); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(out, "03_event_cooccurrence_matrix")


def plot_volume_distribution(data: Dict, out: Path):
    tv = np.asarray(data["active_true_volumes"], dtype=float).reshape(-1)
    pv = np.asarray(data["active_pred_volumes"], dtype=float).reshape(-1)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
    if len(tv): axs[0].hist(np.log1p(tv), bins=80, density=True, alpha=0.6, label="real active")
    if len(pv): axs[0].hist(np.log1p(pv), bins=80, density=True, alpha=0.6, label="model active")
    if len(tv) or len(pv): axs[0].legend(frameon=False)
    style_ax(axs[0], "Volume distribution", "log(1+volume)", "density")
    if len(tv) and len(pv):
        n = min(len(tv), len(pv), 5000)
        q = np.linspace(0.01, 0.99, n)
        axs[1].scatter(np.quantile(tv, q), np.quantile(pv, q), s=4, alpha=0.35)
        lim = np.nanpercentile(np.concatenate([tv, pv]), 99)
        axs[1].plot([0, lim], [0, lim], color="black", lw=1)
    style_ax(axs[1], "Volume Q-Q", "real quantile", "model quantile")
    savefig(out, "04_volume_distribution")


def plot_intensity_and_survival(data: Dict, out: Path):
    true_dt = data["true_dt"]; lam = data["total_intensity"]
    surv = data["survival"]; grid = data["survival_grid"]
    fig, axs = plt.subplots(1, 3, figsize=(14.0, 3.8))
    inv_dt = 1.0 / np.clip(true_dt, 1e-8, None)
    axs[0].scatter(inv_dt, lam, s=5, alpha=0.20)
    lim = np.nanpercentile(np.concatenate([inv_dt, lam]), 99) if len(lam) else 1
    axs[0].plot([0, lim], [0, lim], color="black", lw=1)
    style_ax(axs[0], "λ(t⁺) vs observed 1/Δt (target-free)", "observed 1/Δt", "model λ at last event")

    if len(lam) and surv.size and len(grid):
        qs = np.quantile(lam, np.linspace(0, 1, 6))
        gmax = float(np.quantile(true_dt, 0.98))
        gmask = grid <= max(gmax, grid[0] * 10)
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = (lam >= lo) & (lam <= hi)
            if m.sum() < 5:
                continue
            model_surv = surv[m].mean(axis=0)
            emp_surv = np.array([(true_dt[m] > t).mean() for t in grid[gmask]])
            axs[1].plot(grid[gmask], emp_surv, alpha=0.8, lw=1.4)
            axs[1].plot(grid[gmask], model_surv[gmask], alpha=0.8, lw=1.4, ls="--")
            axs[2].plot(grid[gmask], model_surv[gmask] - emp_surv, alpha=0.8, lw=1.4)
    style_ax(axs[1], "Survival by λ bin (solid=real, dashed=model)", "horizon τ", "P(Δt > τ)")
    style_ax(axs[2], "Survival residual (model - empirical)", "horizon τ", "Δ survival")
    savefig(out, "05_intensity_survival_calibration")


def plot_rollout(roll: Dict, data: Dict, out: Path):
    sim_dt = roll["sim_dt"]; sim_sz = roll["sim_set_size"]
    real_dt = data["true_dt"]; real_sz = data["true_sets"].sum(axis=1)
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    n_steps = sim_dt.shape[1]
    n_real = min(len(real_dt), n_steps)
    axs[0, 0].plot(np.cumsum(real_dt[:n_real]), np.arange(n_real), label="real stream", lw=1.8, color="black")
    for i in range(sim_dt.shape[0]):
        axs[0, 0].plot(np.cumsum(sim_dt[i]), np.arange(n_steps), lw=1.0, alpha=0.6)
    axs[0, 0].legend(frameon=False)
    style_ax(axs[0, 0], "Free-running rollout: cumulative event count", "time", "events")
    axs[0, 1].hist(np.log1p(real_dt), bins=50, density=True, alpha=0.55, label="real")
    axs[0, 1].hist(np.log1p(sim_dt.reshape(-1)), bins=50, density=True, alpha=0.55, label="rollout")
    axs[0, 1].legend(frameon=False)
    style_ax(axs[0, 1], "Inter-event time: real vs rollout", "log(1+Δt)", "density")
    max_sz = int(max(real_sz.max(initial=1), sim_sz.max(initial=1)))
    bins = np.arange(max_sz + 2) - 0.5
    axs[1, 0].hist(real_sz, bins=bins, density=True, alpha=0.55, label="real")
    axs[1, 0].hist(sim_sz.reshape(-1), bins=bins, density=True, alpha=0.55, label="rollout")
    axs[1, 0].legend(frameon=False)
    style_ax(axs[1, 0], "Set size: real vs rollout", "|S|", "probability")
    for arr, label in [(real_dt, "real"), (sim_dt[0], "rollout")]:
        arr = np.log1p(arr[:min(2000, len(arr))])
        lags = np.arange(1, min(60, len(arr) // 2))
        ac = []
        for lag in lags:
            if np.std(arr[:-lag]) > 0 and np.std(arr[lag:]) > 0:
                ac.append(float(np.corrcoef(arr[:-lag], arr[lag:])[0, 1]))
            else:
                ac.append(0.0)
        axs[1, 1].plot(lags, ac, lw=1.6, label=label)
    axs[1, 1].legend(frameon=False)
    style_ax(axs[1, 1], "Inter-event autocorrelation", "lag", "corr(log Δt)")
    savefig(out, "06_free_running_rollout_vs_real")


def plot_horizon_decay(decay: Dict, out: Path):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6))
    axs[0].plot(decay["step"], decay["dt_hist_l1"], marker="o", lw=1.6)
    style_ax(axs[0], "Rollout horizon decay: Δt marginal", "rollout step", "L1 hist distance vs real")
    axs[1].plot(decay["step"], decay["set_size_hist_l1"], marker="o", lw=1.6, color="#E45756")
    style_ax(axs[1], "Rollout horizon decay: set size", "rollout step", "L1 hist distance vs real")
    savefig(out, "07_rollout_horizon_decay")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-files", type=int, default=21)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--seq-length", type=int, default=50)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-test-batches", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--dt-horizon", type=float, default=10.0)
    ap.add_argument("--dt-grid-points", type=int, default=64)
    ap.add_argument("--rollout-steps", type=int, default=300)
    ap.add_argument("--rollout-sequences", type=int, default=8)
    ap.add_argument("--rollout-seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    dl_kwargs = dict(num_workers=args.num_workers)
    if args.cache_dir:
        dl_kwargs["cache_dir"] = args.cache_dir
    train_loader, val_loader, test_loader, event_mapping = create_bfnx_dataloaders(
        args.data_dir, args.batch_size, args.seq_length, args.stride, args.max_files, **dl_kwargs
    )
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ck["config"]; em = ck.get("event_mapping", event_mapping)
    model = create_volume_set_mtpp(em.num_events, cfg, device, use_volume=cfg.get("use_volume", True), intensity_type=cfg.get("intensity_type", "dynamic"))
    model.load_state_dict(ck["model_state_dict"]); model.to(device); model.eval()

    data = collect_predictions(model, test_loader, device, args.threshold, args.max_test_batches,
                               horizon=args.dt_horizon, n_grid=args.dt_grid_points)

    first_batch = move_batch(next(iter(test_loader)), device)
    roll = free_running_rollout(model, first_batch, device, args.threshold,
                                steps=args.rollout_steps, n_seq=args.rollout_sequences,
                                horizon=args.dt_horizon, n_grid=args.dt_grid_points,
                                seed=args.rollout_seed)
    real_sz = data["true_sets"].sum(axis=1) if data["true_sets"].size else np.array([1.0])
    decay = horizon_decay(roll["sim_dt"], roll["sim_set_size"], data["true_dt"], real_sz)

    summary = {
        "metadata": vars(args) | {"device": str(device), "num_event_types": int(em.num_events), "params": int(sum(p.numel() for p in model.parameters()))},
        "methodology": {
            "pit_method": "compensator_trapezoid_quadrature_33pts",
            "pred_dt_method": "survival_quadrature_predictive_median",
            "set_prediction_protocol": "unknown_time_threshold_at_model_predicted_dt",
            "rollout": "free_running_closed_loop_sliding_window",
        },
        "time_true_dt": qstats(data["true_dt"]),
        "time_pred_dt": qstats(data["pred_dt"]),
        "total_intensity_at_last_event": qstats(data["total_intensity"]),
        "pit": qstats(data["pit"]),
        "set_metrics": set_metrics(data["true_sets"], data["pred_sets"]) if data["true_sets"].size else {},
        "active_true_volume": qstats(data["active_true_volumes"]),
        "active_pred_volume": qstats(data["active_pred_volumes"]),
        "rollout_sim_dt": qstats(roll["sim_dt"]),
        "rollout_sim_set_size": qstats(roll["sim_set_size"]),
        "rollout_horizon_decay": decay,
        "artifacts": [
            "01_event_time_calibration.png/svg",
            "02_set_size_distribution.png/svg",
            "03_event_cooccurrence_matrix.png/svg",
            "04_volume_distribution.png/svg",
            "05_intensity_survival_calibration.png/svg",
            "06_free_running_rollout_vs_real.png/svg",
            "07_rollout_horizon_decay.png/svg",
        ],
    }
    if len(data["true_dt"]) and len(data["pred_dt"]) and np.std(data["true_dt"]) and np.std(data["pred_dt"]):
        summary["time_pred_true_corr"] = float(np.corrcoef(data["true_dt"], data["pred_dt"])[0,1])
    if len(data["pit"]):
        hist, edges = np.histogram(np.clip(data["pit"], 0, 1), bins=np.linspace(0,1,11), density=True)
        summary["pit_uniform_l1_error_10bins"] = float(np.mean(np.abs(hist - 1.0)))
    if len(data["true_dt"]) and len(data["pred_dt"]):
        summary["dt_scale_ratio_median_pred_over_true"] = float(
            np.median(data["pred_dt"]) / max(np.median(data["true_dt"]), 1e-8))

    save_json(out / "world_model_diagnostics_summary.json", summary)
    rows = []
    for i in range(len(data["true_dt"])):
        rows.append({
            "i": i,
            "true_dt": float(data["true_dt"][i]),
            "pred_dt": float(data["pred_dt"][i]),
            "total_intensity": float(data["total_intensity"][i]),
            "pit": float(data["pit"][i]),
            "true_set_size": int(data["true_sets"][i].sum()),
            "pred_set_size": int(data["pred_sets"][i].sum()),
        })
    save_csv(out / "world_model_prediction_pairs.csv", rows)

    plot_time_calibration(data, out)
    plot_set_size_and_cooccurrence(data, out)
    plot_volume_distribution(data, out)
    plot_intensity_and_survival(data, out)
    plot_rollout(roll, data, out)
    plot_horizon_decay(decay, out)
    summary_print = {k: v for k, v in summary.items() if k != "metadata"}
    print(json.dumps(summary_print, indent=2))


if __name__ == "__main__":
    main()
