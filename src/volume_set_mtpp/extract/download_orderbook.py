"""Download raw order-book snapshots (cluster-only entry point).

Thin, credential-checking stub. The full downloader runs on the UCL HPC
cluster against the data provider; see extract/README.md. With credentials
present this is where the real GCS/Kaiko fetch implementation goes (keep the
same CLI). Without them it exits with guidance rather than failing obscurely.
"""

import argparse

from ._creds import require_credentials


def main() -> None:
    ap = argparse.ArgumentParser(description="Download raw order-book snapshots.")
    ap.add_argument("--crypto", required=True, help="symbol, e.g. wif")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--out-dir", default="data/orderbook")
    args = ap.parse_args()

    cred = require_credentials()
    raise SystemExit(
        f"extract.download_orderbook: credentials OK ({cred}), but the real "
        f"downloader implementation is cluster-only and not vendored into this "
        f"open-source repo. Run on the UCL HPC cluster, or drop the production "
        f"implementation into this module. Requested: crypto={args.crypto} "
        f"parallel={args.parallel} out_dir={args.out_dir}."
    )


if __name__ == "__main__":
    main()
