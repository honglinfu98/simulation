#!/usr/bin/env python3
"""Hazard-profile probe: model intensity at gap age t vs empirical hazard.

Produces the data behind the quiet-regime figure: for each checkpoint, the mean
intensity lambda(t) at gap age t among windows whose TRUE gap survives past t,
plus the empirical hazard of the real gap distribution, plus the per-event
timeNLL decomposition (event term -log lambda vs compensator mass).

Run ON THE CLUSTER from the repo root (CPU is fine, ~10 min):

    PYTHONPATH=. python3 paper/scripts/probe_hazard.py \
        --root experiments/final_v2 --tags nhp-s1 s2p2-s3 ss2p2-full-s3 \
        --out paper/data/hazard_profiles.json

Then scp the JSON into paper/data/ locally. make_plots.py renders it.
"""
import argparse
import json
import os

import numpy as np
import torch

from volume_set_mtpp.training.data_loader import create_bfnx_dataloaders
from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp
from volume_set_mtpp.evaluation.world_model_diagnostics import (
    _total_intensity_at_dts, _compensator_at)

DATA = "/SAN/medic/TFOW/data/events/gmni_eth_7_v2_marks"
CACHE = DATA + "/.tensor_cache_eval"
SEQ, STRIDE, BS, MAXB = 512, 128, 32, 30
GRID = [0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0]
GAP_EDGES = [0.0, 0.02, 0.1, 0.5, 2.0, float("inf")]
GAP_LABELS = ["<20ms", "20-100ms", "0.1-0.5s", "0.5-2s", ">2s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/final_v2")
    ap.add_argument("--tags", nargs="+",
                    default=["nhp-s1", "s2p2-s3", "ss2p2-full-s3"])
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--cache-dir", default=CACHE)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    _, _, test_loader, em = create_bfnx_dataloaders(
        args.data_dir, BS, SEQ, STRIDE, 7, num_workers=0,
        cache_dir=args.cache_dir)
    batches = []
    for bi, batch in enumerate(test_loader):
        if bi >= MAXB:
            break
        batches.append(batch)

    grid = torch.tensor(GRID)
    dts_all = torch.cat([b["target_time"][b["target_marks"].sum(dim=1) > 0]
                         .clamp_min(1e-8) for b in batches]).numpy()
    g = np.array(GRID)
    haz = []
    for i in range(len(g) - 1):
        at_risk = float((dts_all >= g[i]).sum())
        died = float(((dts_all >= g[i]) & (dts_all < g[i + 1])).sum())
        haz.append(died / max(at_risk, 1.0) / (g[i + 1] - g[i]))
    out = {"grid": GRID, "empirical_hazard": haz + [None],
           "n_gaps": int(dts_all.size), "gap_labels": GAP_LABELS,
           "models": {}}

    for tag in args.tags:
        ck = torch.load(f"{args.root}/{tag}/train/best_model.pt",
                        map_location="cpu", weights_only=False)
        cfg = ck["config"]
        model = create_volume_set_mtpp(
            em.num_events, cfg, "cpu",
            use_volume=cfg.get("use_volume", True),
            intensity_type=cfg.get("intensity_type", "dynamic"))
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        DT, LAM, U = [], [], []
        lam_sum = torch.zeros(len(GRID))
        lam_n = torch.zeros(len(GRID))
        with torch.no_grad():
            for batch in batches:
                im = batch["input_marks"].float()
                it = batch["input_times"].float()
                tm = batch["target_marks"].float()
                tt = batch["target_time"].float()
                ts = torch.cumsum(it, dim=1)
                states = model.decoder.get_states(im, ts)
                dt = tt.clamp_min(1e-8)
                ev = tm.sum(dim=1) > 0
                lam = _total_intensity_at_dts(model, states, ts,
                                              dt.unsqueeze(1))[:, 0]
                u = _compensator_at(model, states, ts, dt)
                DT.append(dt[ev]); LAM.append(lam[ev]); U.append(u[ev])
                lg = _total_intensity_at_dts(
                    model, states, ts, grid.unsqueeze(0).expand(dt.size(0), -1))
                for gi in range(len(GRID)):
                    m = ev & (dt >= grid[gi])
                    lam_sum[gi] += lg[m, gi].sum()
                    lam_n[gi] += m.sum()
        dt = torch.cat(DT).numpy()
        lam = torch.cat(LAM).numpy()
        u = torch.cat(U).numpy()
        evt = -np.log(np.clip(lam, 1e-8, None))
        bins = {}
        for lo, hi, lb in zip(GAP_EDGES[:-1], GAP_EDGES[1:], GAP_LABELS):
            m = (dt >= lo) & (dt < hi)
            bins[lb] = {"n": int(m.sum()),
                        "event_term": float(evt[m].mean()),
                        "comp_term": float(u[m].mean()),
                        "time_nll": float((evt + u)[m].mean())}
        out["models"][tag] = {
            "profile": (lam_sum / lam_n.clamp_min(1)).tolist(),
            "time_nll": float((evt + u).mean()),
            "event_term": float(evt.mean()),
            "comp_term": float(u.mean()),
            "lam_max": float(lam.max()),
            "frac_at_ceiling": float((lam > 0.95 * lam.max()).mean()),
            "gap_bins": bins,
        }
        print(tag, "done")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
