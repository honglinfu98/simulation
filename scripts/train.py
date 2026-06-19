#!/usr/bin/env python3
"""Train a Volume-Set MTPP model (training stage).

    python scripts/train.py --decoder-type lgm --data-dir <events> \
        --lgm-target-rate 2.381 --nmh-project-rho 0.86 --mark-head categorical --epochs 40

Run `python scripts/train.py --help` for the full flag list.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from volume_set_mtpp.training.train import main

if __name__ == "__main__":
    main()
