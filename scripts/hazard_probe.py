#!/usr/bin/env python3
"""Hazard-profile probe: mean model intensity at gap age vs. empirical hazard.

For each asset and checkpoint: traverse real test windows, and for a log grid
of gap ages delta compute the mean model intensity at age delta among gaps
that survive past delta (teacher-forced; calibration not required). Also
estimate the empirical hazard of the real gap distribution on the same grid:
h_j = deaths_j / (at-risk_j * bin width). Feeds fig_hazard_profile.

    PYTHONPATH=. python3 scripts/hazard_probe.py \
        --data-dir /SAN/.../cbse_btc_7d --cache-dir .../.tensor_cache_eval \
        --root experiments/ma_cbse/btc --tags nhp-s1 s2p2-s3 ss2p2-full-s1 \
        --out /tmp/hazard_btc.json
"""
import argparse
import json

import numpy as np
import torch

from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp
from volume_set_mtpp.training.data_loader import create_bfnx_dataloaders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-files", type=int, default=7)
    ap.add_argument("--seq-length", type=int, default=1024)
    ap.add_argument("--max-windows", type=int, default=256)
    ap.add_argument("--grid-min", type=float, default=1e-3)
    ap.add_argument("--grid-max", type=float, default=10.0)
    ap.add_argument("--grid-points", type=int, default=25)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, em = create_bfnx_dataloaders(
        data_dir=args.data_dir, batch_size=32, sequence_length=args.seq_length,
        stride=args.seq_length, max_files=args.max_files, num_workers=0,
        cache_dir=args.cache_dir)
    grid = np.logspace(np.log10(args.grid_min), np.log10(args.grid_max),
                       args.grid_points)

    # empirical hazard from pooled real gaps
    gaps = []
    n_win = 0
    batches = []
    for b in test_loader:
        batches.append(b)
        gaps.append(b["input_times"].reshape(-1).numpy())
        n_win += b["input_times"].shape[0]
        if n_win >= args.max_windows:
            break
    g = np.concatenate(gaps)
    g = g[g > 0]
    deaths, _ = np.histogram(g, bins=grid)
    at_risk = np.array([(g >= lo).sum() for lo in grid[:-1]])
    width = np.diff(grid)
    emp = (deaths / np.maximum(at_risk, 1) / width).tolist() + [None]

    out = {"grid": grid.tolist(), "empirical_hazard": emp,
           "n_gaps": int(len(g)), "models": {}}

    for tag in args.tags:
        ck_path = f"{args.root}/{tag}/train/best_model.pt"
        try:
            ck = torch.load(ck_path, map_location=device, weights_only=False)
        except FileNotFoundError:
            print(f"SKIP {tag}: no checkpoint")
            continue
        cfg = ck["config"]
        model = create_volume_set_mtpp(em.num_events, cfg, device,
                                       use_volume=cfg.get("use_volume", True),
                                       intensity_type=cfg.get("intensity_type", "dynamic"))
        model.load_state_dict(ck["model_state_dict"]); model.to(device); model.eval()

        s_sum = np.zeros(len(grid))
        s_cnt = np.zeros(len(grid))
        with torch.no_grad():
            for b in batches:
                marks = b["input_marks"].float().to(device)
                dts = b["input_times"].float().clamp_min(0).to(device)
                ts = torch.cumsum(dts, dim=1)
                states = model.decoder.get_states(marks, ts)
                nxt = torch.cat([dts[:, 1:], torch.full_like(dts[:, :1], np.inf)], dim=1)
                for j, age in enumerate(grid):
                    valid = nxt > age                       # gap survives past age
                    if valid.sum() == 0:
                        continue
                    q = ts + age
                    # evaluate at absolute query times via get_hidden_h
                    h = model.decoder.get_hidden_h(state_values=states,
                                                   state_times=ts, timestamps=q)
                    d = model.get_total_intensity_and_items(h)
                    lam = d["total_intensity"].squeeze(-1)
                    s_sum[j] += float(lam[valid].sum())
                    s_cnt[j] += int(valid.sum())
        prof = (s_sum / np.maximum(s_cnt, 1)).tolist()
        out["models"][tag] = {"profile": prof}
        print(f"{tag}: profile computed over {int(s_cnt[0])} gaps at age {grid[0]:.3g}")

    json.dump(out, open(args.out, "w"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
