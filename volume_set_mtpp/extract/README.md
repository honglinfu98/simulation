# `extract/` — raw market-data extraction (cluster-only)

This stage downloads raw limit-order-book snapshots and trades from the data
provider (Kaiko via Google Cloud Storage) into the local/cluster `data/` tree.
The downstream `process/` stage turns those raw files into the 62-channel event
JSONL the models train on.

## Why these are stubs

The production extractors run **only on the UCL HPC cluster**, against
`/SAN/medic/TFOW/...`, and require provider credentials that must **never** be
committed:

- `GOOGLE_APPLICATION_CREDENTIALS` — path to a GCS service-account JSON, or
- `KAIKO_API_KEY` — Kaiko REST API key.

The modules here (`download_orderbook.py`, `download_trades.py`) are thin,
credential-checking entry points. Run on a machine with the credentials and the
real downloader available; absent credentials they exit with a clear message
rather than failing obscurely. Drop the full cluster implementation into these
files (same CLI) when extracting on the cluster — no structural change needed.

## Usage (on the cluster)

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json   # or KAIKO_API_KEY=...
python -m volume_set_mtpp.extract.download_orderbook --crypto wif --parallel 4
python -m volume_set_mtpp.extract.download_trades    --crypto wif --parallel 4
```

Then build events with the `process/` stage (see `docs/RUNBOOK.md`).
