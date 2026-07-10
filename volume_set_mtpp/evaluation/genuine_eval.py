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

from volume_set_mtpp.training.data_loader import create_bfnx_dataloaders, StatefulBFNXLoader
from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp
from volume_set_mtpp.evaluation.world_model_diagnostics import (
    _total_intensity_at_dts, _compensator_at, _survival_quadrature)


def _stream_old_states(decoder, right_last: torch.Tensor):
    """Convert the last packed right state into the decoder's old_states form.

    S2P2-family 'output' readout packs [L layer states | L-1 held anchors]:
    restore layer states [B, L, H]. NHP's packed 6H state passes through.
    Decoders whose get_states ignores old_states (LSTM/SAHP) return None --
    they are evaluated per fresh window (fixed context = their training and
    simulation regime), but still on EVERY event.
    """
    if hasattr(decoder, "_initial_layer_states"):
        L, H = decoder.num_layers, decoder.recurrent_hidden_size
        if right_last.shape[-1] == (2 * L - 1) * H:
            return right_last[:, :L * H].reshape(right_last.shape[0], L, H)
        return right_last
    if getattr(decoder, "is_ptp", False):
        return right_last
    if hasattr(decoder, "decay") and hasattr(decoder, "recurrence"):   # NHP packed 6H
        return right_last
    return None


def streaming_metrics(model, dataset, batch_size: int, device,
                      dt_horizon: float, dt_grid: int, max_windows: int = 0):
    """Score EVERY event of the (test) stream with carried decoder state.

    Lanes walk contiguous stretches of the stream in order (StatefulBFNXLoader,
    stride == seq); for state-carrying decoders the state is handed across
    windows, so event i is scored with the FULL stream history behind it.
    Per event: mark CE/acc at the anti-leakage left state, one-step time NLL
    -log lambda(t_i^-) + Lambda(gap_i) (endpoint rule per gap), compensator
    masses u_i for KS/mean_u, and E[dt] time-MAE via per-event survival
    quadrature. Returns aggregate dict + number of events scored.
    """
    loader = StatefulBFNXLoader(dataset, batch_size)
    n_ev = correct = 0
    nll_sum = time_nll_sum = dt_err_sum = logdt_err_sum = 0.0
    u_all = []
    carried = None
    n_win = 0
    with torch.no_grad():
        for batch in loader:
            n_win += 1
            if max_windows and n_win > max_windows:
                break
            reset = batch.pop("reset_mask").to(device)
            im = batch["input_marks"].to(device).float()
            it = batch["input_times"].to(device).float()
            ts = torch.cumsum(it, dim=1)
            old = None
            if carried is not None:
                old = carried
                if torch.is_tensor(old) and reset.any():   # zone/file boundaries restart cold
                    old = old.clone()
                    if hasattr(model.decoder, "init_state") and old.dim() == 3:
                        old[reset] = model.decoder.init_state.detach().to(old.device, old.dtype)
                    else:
                        old[reset] = 0.0
            states, left = model.decoder.get_states_and_event_left_states(im, ts, old_states=old)
            d = model.get_total_intensity_and_items(left)
            lam_ev = d["total_intensity"].squeeze(-1).clamp_min(1e-8)     # [B,N]
            logits = d["item_logits"]                                     # [B,N,K]

            ev = im.sum(dim=-1) > 0                                       # genuine events [B,N]
            tgt = im.argmax(dim=-1)
            lg = logits[ev]
            tg = tgt[ev]
            correct += (lg.argmax(dim=1) == tg).sum().item()
            nll_sum += F.cross_entropy(lg, tg, reduction="sum").item()

            # per-gap compensator: quadrature (below, alongside E[dt]) when
            # dt_grid > 0, endpoint fallback otherwise
            u_i = (it * lam_ev)                                           # [B,N] endpoint fallback

            # E[dt] per event: survival quadrature per position, anchored WITHOUT
            # leakage -- for event j the states/times are sliced to history
            # 0..j-1, so queries beyond t_j cannot resolve to later states
            # (matching the windowed evaluator's last-event semantics exactly).
            B, N = it.shape
            if dt_grid > 0:
                e_dt = torch.zeros(B, N, device=device)
                grid = torch.logspace(-4, math.log10(dt_horizon), dt_grid, device=device)
                widths = (grid[1:] - grid[:-1]).view(1, -1)
                t_prev = torch.cat([torch.zeros(B, 1, device=device), ts[:, :-1]], dim=1)
                zt = torch.zeros(B, 1, device=device)
                Lam_dt = torch.zeros(B, N, device=device)                 # quadrature Λ(gap_j)
                for j in range(N):
                    q = t_prev[:, j:j + 1] + grid.view(1, -1)             # [B,G]
                    h = model.decoder.get_hidden_h(
                        state_values=states[:, :j + 1], state_times=ts[:, :j] if j > 0 else zt,
                        timestamps=q)
                    lam_q = model.get_total_intensity_and_items(h)["total_intensity"]
                    lam_q = lam_q.squeeze(-1).clamp_min(1e-8)             # [B,G]
                    seg = 0.5 * (lam_q[:, 1:] + lam_q[:, :-1]) * widths
                    Lam = torch.cat([torch.zeros(B, 1, device=device),
                                     torch.cumsum(seg, dim=-1)], dim=-1)
                    surv = torch.exp(-Lam)
                    e = grid[0] + (0.5 * (surv[:, 1:] + surv[:, :-1]) * widths).sum(dim=-1)
                    e = (e + surv[:, -1] / lam_q[:, -1]).clamp(min=1e-6, max=2.0 * dt_horizon)
                    e_dt[:, j] = e
                    # Λ at the realized gap: interpolate the cumulative on the grid
                    dtj = it[:, j]
                    idx = torch.searchsorted(grid, dtj.contiguous()).clamp(1, dt_grid - 1)
                    l_lo = Lam.gather(1, (idx - 1).unsqueeze(1)).squeeze(1)
                    l_hi = Lam.gather(1, idx.unsqueeze(1)).squeeze(1)
                    g_lo, g_hi = grid[idx - 1], grid[idx]
                    frac = ((dtj - g_lo) / (g_hi - g_lo).clamp_min(1e-12)).clamp(0.0, 1.0)
                    lam_j = l_lo + frac * (l_hi - l_lo)
                    over = dtj > grid[-1]                                 # beyond horizon: extend flat
                    if bool(over.any()):
                        lam_j = torch.where(over, Lam[:, -1] + lam_q[:, -1] * (dtj - grid[-1]), lam_j)
                    under = dtj < grid[0]
                    if bool(under.any()):
                        lam_j = torch.where(under, lam_q[:, 0] * dtj, lam_j)
                    Lam_dt[:, j] = lam_j
                u_i = Lam_dt                                              # quadrature per-gap masses
                dt_c = it.clamp_min(1e-8)
                dt_err_sum += (e_dt - dt_c).abs()[ev].sum().item()
                logdt_err_sum += (torch.log(e_dt.clamp_min(1e-6)) - torch.log(dt_c)).abs()[ev].sum().item()
            time_nll = (-torch.log(lam_ev) + u_i)
            time_nll_sum += time_nll[ev].sum().item()
            u_all.append(u_i[ev].detach().cpu())
            n_ev += int(ev.sum().item())

            nxt = _stream_old_states(model.decoder, states[:, -1].detach())
            carried = nxt
    return dict(n_ev=n_ev, correct=correct, nll_sum=nll_sum, time_nll_sum=time_nll_sum,
                dt_err_sum=dt_err_sum, logdt_err_sum=logdt_err_sum,
                u=torch.cat(u_all).numpy() if u_all else np.array([]))


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
                    help="quadrature points for E[dt] (streaming: 0 skips tMAE)")
    ap.add_argument("--streaming", action="store_true",
                    help="score EVERY test event with carried state across non-overlapping "
                         "windows (stream order; requires stride == seq-length). Decoders "
                         "without an old_states path (LSTM/SAHP) are scored per fresh "
                         "window -- still every event, fixed context.")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dl = dict(cache_dir=args.cache_dir) if args.cache_dir else {}
    stride = args.seq_length if args.streaming else args.stride
    if args.streaming and args.stride != args.seq_length:
        print(f"STREAMING: forcing stride {args.stride} -> {args.seq_length} (non-overlapping)", flush=True)
    _, _, test_loader, em = create_bfnx_dataloaders(
        args.data_dir, args.batch_size, args.seq_length, stride, args.max_files, num_workers=0, **dl)

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = create_volume_set_mtpp(em.num_events, cfg, device,
                                   use_volume=cfg.get("use_volume", True),
                                   intensity_type=cfg.get("intensity_type", "dynamic"))
    model.load_state_dict(ck["model_state_dict"]); model.to(device); model.eval()

    if args.streaming:
        dec = model.decoder
        state_carried = (hasattr(dec, "_initial_layer_states") or getattr(dec, "is_ptp", False)
                         or (hasattr(dec, "decay") and hasattr(dec, "recurrence")))
        print("STREAMING evaluator: every test event scored; state "
              + ("CARRIED across windows" if state_carried else "per fresh window (no old_states path)"),
              flush=True)
        agg = streaming_metrics(model, test_loader.dataset, args.batch_size, device,
                                args.dt_horizon, args.dt_grid_points)
        ne = max(agg["n_ev"], 1)
        u = agg["u"]
        res = {"label": args.label, "checkpoint": args.checkpoint, "evaluator": "streaming",
               "state_carried": bool(state_carried),
               "n_genuine_events": agg["n_ev"],
               "overall_nll_per_event": (agg["nll_sum"] + agg["time_nll_sum"]) / ne,
               "time_nll_per_event": agg["time_nll_sum"] / ne,
               "mark_nll_per_event": agg["nll_sum"] / ne,
               "time_rescaling_ks": _ks_vs_exp1(u),
               "compensator_mean_u": float(u.mean()) if u.size else float("nan"),
               "compensator_var_u": float(u.var()) if u.size else float("nan"),
               "time_mae_seconds": agg["dt_err_sum"] / ne, "time_mae_log": agg["logdt_err_sum"] / ne,
               "genuine_mark_accuracy": agg["correct"] / ne,
               "genuine_mark_perplexity": math.exp(agg["nll_sum"] / ne),
               "mark_head": cfg.get("mark_head", "bernoulli")}
        print(json.dumps(res, indent=2), flush=True)
        if args.output:
            Path(args.output).write_text(json.dumps(res, indent=2))
        return

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
    res = {"label": args.label, "checkpoint": args.checkpoint, "evaluator": "windowed",
           "n_genuine_events": n_ev,
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
