"""Genuine-event prediction metrics for any VolumeSetMTPP checkpoint, computed
CONSISTENTLY so models with different mark heads are comparable.

The gmni-marks data is one-mark-or-empty per slot (~33% empty, 0% multi).  The
comparable mark metric is therefore over GENUINE events only:
  - top-1 mark accuracy: argmax(mark logits) == true mark, on non-empty targets
  - mark perplexity: exp(mean -log p(true mark)) over non-empty targets,
    where p is softmax(logits) for BOTH bernoulli and categorical heads (the
    mark *ranking* is what we score, head-agnostic).
Empty-target positions are excluded identically for every model, so the number
is head-agnostic and fair across the whole comparison.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from volume_set_mtpp.training.data_loader import create_bfnx_dataloaders
from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp
from volume_set_mtpp.evaluation.world_model_diagnostics import (
    _total_intensity_at_dts, _compensator_at, _survival_quadrature)


def _ks_vs_exp1(u: np.ndarray) -> float:
    """KS statistic between samples u and Exp(1).

    Time-rescaling theorem: the compensator masses u_i = int_{t_{i-1}}^{t_i} lambda
    should be i.i.d. Exp(1) iff the model's intensity is correct. KS = max gap
    between the empirical CDF of u and the Exp(1) CDF 1-e^{-u}; 0 = perfect.
    """
    u = np.sort(np.asarray(u, dtype=np.float64))
    n = u.size
    if n == 0:
        return float("nan")
    f_emp_hi = np.arange(1, n + 1) / n
    f_emp_lo = np.arange(0, n) / n
    f_exp = 1.0 - np.exp(-u)
    return float(np.maximum(np.abs(f_emp_hi - f_exp), np.abs(f_exp - f_emp_lo)).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--max-files", type=int, default=7)
    ap.add_argument("--seq-length", type=int, default=50)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-batches", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--label", default="model")
    ap.add_argument("--output", default=None)
    ap.add_argument("--dt-horizon", type=float, default=60.0,
                    help="integration horizon (s) for the time compensator / E[dt]")
    ap.add_argument("--dt-grid-points", type=int, default=128,
                    help="quadrature points for E[dt]")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dl = dict(cache_dir=args.cache_dir) if args.cache_dir else {}
    _, _, test_loader, em = create_bfnx_dataloaders(
        args.data_dir, args.batch_size, args.seq_length, args.stride, args.max_files, num_workers=0, **dl)

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = create_volume_set_mtpp(em.num_events, cfg, device,
                                   use_volume=cfg.get("use_volume", True),
                                   intensity_type=cfg.get("intensity_type", "dynamic"))
    model.load_state_dict(ck["model_state_dict"]); model.to(device); model.eval()

    n_ev = 0
    correct = 0
    nll_sum = 0.0          # mark NLL (cross-entropy) sum over genuine events
    time_nll_sum = 0.0     # time NLL sum: -log lambda(dt) + Lambda(dt)
    dt_err_sum = 0.0       # |E[dt] - dt| (time prediction MAE numerator)
    logdt_err_sum = 0.0    # |log E[dt] - log dt|
    u_all = []             # compensator masses Lambda(dt) for the time-rescaling KS
    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            if bi >= args.max_batches:
                break
            im = batch["input_marks"].to(device).float()
            it = batch["input_times"].to(device).float()
            tm = batch["target_marks"].to(device).float()
            tt = batch["target_time"].to(device).float()
            ts = torch.cumsum(it, dim=1)
            states = model.decoder.get_states(im, ts)
            q = ts[:, -1:] + tt.unsqueeze(1).clamp_min(0.0)
            h = model.decoder.get_hidden_h(state_values=states, state_times=ts, timestamps=q)
            d = model.get_total_intensity_and_items(h)
            logits = d["item_logits"].squeeze(1)              # [B,K]
            ev = tm.sum(dim=1) > 0                            # genuine-event mask
            if ev.sum() == 0:
                continue
            tgt = tm[ev].argmax(dim=1)
            lg = logits[ev]
            pred = lg.argmax(dim=1)
            correct += (pred == tgt).sum().item()
            nll = F.cross_entropy(lg, tgt, reduction="sum").item()   # ranking-based, head-agnostic
            nll_sum += nll

            # ---- timing metrics (head-agnostic; on the same genuine events) ----
            dt = tt.clamp_min(1e-8)
            lam_dt = _total_intensity_at_dts(model, states, ts, dt.unsqueeze(1))[:, 0]   # lambda(dt) [B]
            Lam_dt = _compensator_at(model, states, ts, dt)                              # Lambda(dt)=int_0^dt lambda [B]
            _, _, _, _, e_dt = _survival_quadrature(model, states, ts, args.dt_horizon, args.dt_grid_points)
            tnll = (-torch.log(lam_dt.clamp_min(1e-8)) + Lam_dt)        # one-step time NLL [B]
            m = ev
            time_nll_sum += tnll[m].sum().item()
            u_all.append(Lam_dt[m].detach().cpu())
            dt_err_sum += (e_dt[m] - dt[m]).abs().sum().item()
            logdt_err_sum += (torch.log(e_dt[m].clamp_min(1e-6)) - torch.log(dt[m])).abs().sum().item()
            n_ev += int(ev.sum().item())

    ne = max(n_ev, 1)
    acc = correct / ne
    mark_nll = nll_sum / ne
    ppl = math.exp(mark_nll)
    time_nll = time_nll_sum / ne
    overall_nll = (nll_sum + time_nll_sum) / ne                 # total per-event NLL (the OVERALL fit)
    u = torch.cat(u_all).numpy() if u_all else np.array([])
    ks = _ks_vs_exp1(u)
    mean_u = float(u.mean()) if u.size else float("nan")        # should be ~1.0 if calibrated
    var_u = float(u.var()) if u.size else float("nan")          # should be ~1.0 (Exp(1))
    res = {"label": args.label, "checkpoint": args.checkpoint, "n_genuine_events": n_ev,
           # OVERALL FIT (check this first):
           "overall_nll_per_event": overall_nll,
           "time_nll_per_event": time_nll,
           "mark_nll_per_event": mark_nll,
           # timing calibration / accuracy:
           "time_rescaling_ks": ks, "compensator_mean_u": mean_u, "compensator_var_u": var_u,
           "time_mae_seconds": dt_err_sum / ne, "time_mae_log": logdt_err_sum / ne,
           # mark / direction:
           "genuine_mark_accuracy": acc, "genuine_mark_perplexity": ppl,
           "mark_head": cfg.get("mark_head", "bernoulli")}
    print(json.dumps(res, indent=2), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
