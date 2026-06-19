#!/usr/bin/env python3
"""Fetch raw LOB / trade data (extraction stage; cluster-only, needs credentials).

    python scripts/fetch_data.py orderbook --crypto wif --parallel 4
    python scripts/fetch_data.py trades    --crypto wif --parallel 4

See volume_set_mtpp/extract/README.md for the credential setup.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("orderbook", "trades"):
        sys.exit("usage: fetch_data.py {orderbook|trades} [--crypto ... --parallel ...]")
    kind = sys.argv.pop(1)  # remove the subcommand so the target sees its own args
    if kind == "orderbook":
        from volume_set_mtpp.extract.download_orderbook import main as run
    else:
        from volume_set_mtpp.extract.download_trades import main as run
    run()


if __name__ == "__main__":
    main()
