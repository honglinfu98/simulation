#!/usr/bin/env python3
"""Train a Volume-Set MTPP model (training stage).

    python scripts/train.py --decoder-type ss2p2 --data-dir <events> \
        --s2p2-layers 2 --ss2p2-wnorm-cap 6.0 --target-rate 3.77 --mark-head categorical --epochs 40

Run `python scripts/train.py --help` for the full flag list.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from volume_set_mtpp.training.train import main

if __name__ == "__main__":
    main()
