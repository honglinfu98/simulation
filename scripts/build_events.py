#!/usr/bin/env python3
"""Build 62-channel event JSONL from raw LOB/trades (process stage).

    python scripts/build_events.py        # see the module for --exchange/--workers flags
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from volume_set_mtpp.process.process_all_events_chunked import main

if __name__ == "__main__":
    main()
