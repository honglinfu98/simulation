#!/usr/bin/env python3
"""Collect realism_<tag>.json artifacts into one snapshot for the paper.

Run ON THE CLUSTER from the repo root:

    python3 paper/scripts/collect_realism.py \
        --root btc=experiments/ma_cbse/btc \
        --root eth=experiments/ma_cbse/eth \
        --root sol=experiments/ma_cbse/sol \
        --out /tmp/realism.json

then scp to paper/data/realism.json.  make_realism_tables.py and
make_realism_plots.py consume the snapshot.
"""
import argparse
import json
import os

MODELS = ["nhp", "lstm", "sahp", "pct-lstm", "s2p2", "ss2p2-full"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    snap = {}
    n = 0
    for spec in args.root:
        name, path = spec.split("=", 1)
        snap[name] = {}
        for mdl in MODELS:
            for s in [1, 2, 3]:
                tag = f"{mdl}-s{s}"
                for r in [1, 2, 3]:
                    p = os.path.join(path, tag, f"sf_r{r}", f"realism_{tag}.json")
                    if os.path.exists(p):
                        snap[name].setdefault(tag, {})[str(r)] = json.load(open(p))
                        n += 1
    json.dump(snap, open(args.out, "w"))
    print(f"wrote {args.out}: {n} realism artifacts")


if __name__ == "__main__":
    main()
