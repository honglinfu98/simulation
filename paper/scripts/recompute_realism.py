#!/usr/bin/env python3
"""Recompute realism_<tag>.json from banked realism_streams_<tag>.npz.

CPU-only: no model, no rollout -- the streams were dumped by the harness under
--realism, so metric/replay iterations never need the GPU again.

    PYTHONPATH=. python3 paper/scripts/recompute_realism.py \
        --roots experiments/ma_cbse/btc experiments/ma_cbse/eth \
                experiments/ma_cbse/sol \
        --schema paper/data/idx_to_event.json
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from volume_set_mtpp.evaluation.realism import compute_realism


def load_streams(npz_path, k):
    z = np.load(npz_path)
    eye = np.eye(k, dtype=bool)
    def seqs(prefix):
        idxs = sorted({int(re.match(rf"{prefix}_k_(\d+)", key).group(1))
                       for key in z.files if key.startswith(f"{prefix}_k_")})
        return [(eye[z[f"{prefix}_k_{i}"]], z[f"{prefix}_dt_{i}"].astype(float))
                for i in idxs]
    return seqs("sim"), seqs("real")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--duration", type=float, default=600.0)
    args = ap.parse_args()
    idx_to_event = json.load(open(args.schema))
    k = len(idx_to_event)
    n = 0
    for root in args.roots:
        for npz in sorted(glob.glob(os.path.join(root, "*", "sf_r*",
                                                 "realism_streams_*.npz"))):
            out = npz.replace("realism_streams_", "realism_").replace(".npz", ".json")
            meta = {}
            if os.path.exists(out):
                old = json.load(open(out))
                meta = {kk: old.get(kk) for kk in
                        ["label", "rate_scale_k", "rollout_seed"]}
            sim_seqs, real_segs = load_streams(npz, k)
            if not sim_seqs or not real_segs:
                print("SKIP", npz)
                continue
            res = compute_realism(sim_seqs, real_segs, idx_to_event,
                                  duration=args.duration)
            res.update(meta)
            json.dump(res, open(out, "w"))
            n += 1
            print(f"[{n}] {out}", flush=True)
    print(f"recomputed {n} artifacts")


if __name__ == "__main__":
    main()
