# src/volume_set_mtpp/process/process_all_events_chunked.py
#!/usr/bin/env python3
"""
Process all your order book and trade data to create events - CHUNKED HPC VERSION.
Features:
- Chunked reading to prevent OOM on HPC
- JSONL streaming for partial output recovery
- Worker-level memory protection
"""

import os
import sys
import glob
import time
from collections import defaultdict
from multiprocessing import Pool
from typing import Tuple, Optional

from volume_set_mtpp.settings import PROJECT_ROOT

# Add project to path
project_root = PROJECT_ROOT
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

# Optional: dump stack traces on SIGUSR1 (great for hangs)
import faulthandler
import signal
faulthandler.register(signal.SIGUSR1)

from volume_set_mtpp.process.event_construction_chunked import process_data_files_chunked


def check_data_alignment() -> bool:
    print("Checking data alignment...", flush=True)

    orderbook_dir = f"{project_root}/data/orderbook"
    trades_dir = f"{project_root}/data/trades"

    orderbook_files = glob.glob(
        os.path.join(orderbook_dir, "**/full_order_book_*.csv.gz"), recursive=True
    )
    ob_inventory = defaultdict(lambda: defaultdict(set))
    for filepath in orderbook_files:
        filename = os.path.basename(filepath)
        parts = filename.replace("full_order_book_", "").replace(".csv.gz", "").split("_")
        if len(parts) >= 4:
            exchange = parts[0]
            pair = parts[2]
            date = parts[3]
            ob_inventory[exchange][pair].add(date)

    trade_files = glob.glob(
        os.path.join(trades_dir, "**/trades_*.csv.gz"), recursive=True
    )
    trade_inventory = defaultdict(lambda: defaultdict(set))
    for filepath in trade_files:
        filename = os.path.basename(filepath)
        parts = filename.replace("trades_", "").replace(".csv.gz", "").split("_")
        if len(parts) >= 4:
            exchange = parts[0]
            pair = parts[2]
            date = parts[3]
            trade_inventory[exchange][pair].add(date)

    missing_trades = []
    for exchange, pairs in ob_inventory.items():
        for pair, dates in pairs.items():
            for date in dates:
                if date not in trade_inventory[exchange][pair]:
                    missing_trades.append((exchange, pair, date))

    if missing_trades:
        print(f"⚠️ WARNING: Missing {len(missing_trades)} trade files:", flush=True)
        for exc, pair, date in missing_trades[:5]:
            print(f" - {exc}/{pair}/{date}", flush=True)
        if len(missing_trades) > 5:
            print(f" ... and {len(missing_trades)-5} more", flush=True)
        return False

    print(
        f"✅ Data aligned! Found {len(orderbook_files)} orderbook and {len(trade_files)} trade files",
        flush=True,
    )
    return True


def get_n_workers(default: int = 8, cap: int = 4) -> int:
    """
    Determine worker count from SGE NSLOTS with optional cap
    to prevent memory explosion.
    """
    try:
        n = int(os.environ.get("NSLOTS", default))
    except Exception:
        n = default

    n = max(1, n)
    return min(n, cap)


def process_single_file_worker(args: Tuple) -> Tuple[str, bool, str]:
    """
    Worker function for processing a single orderbook/trade pair with chunked reading.

    Args:
        args: Tuple of (orderbook_file, trades_file, output_file, coin, chunksize)

    Returns:
        Tuple of (output_file, success, message)
    """
    orderbook_file, trades_file, output_file, coin, chunksize = args

    start = time.time()
    print(f"[WORKER START] {coin}", flush=True)

    try:
        process_data_files_chunked(
            orderbook_file=orderbook_file,
            trades_file=trades_file,
            k_levels=10,
            chunksize=chunksize,
            output_file=output_file,
            jsonl_format=True
        )

        dt = time.time() - start
        print(f"[WORKER DONE] {coin} in {dt/60:.1f} min", flush=True)
        return (output_file, True, f"Success in {dt/60:.1f} min")

    except Exception as e:
        dt = time.time() - start
        print(f"[WORKER FAIL] {coin} after {dt/60:.1f} min: {e}", flush=True)
        return (output_file, False, str(e))


def batch_process_exchange_data_chunked(
    exchange: str,
    base_data_dir: str,
    output_dir: str,
    k_levels: int = 10,
    n_workers: int = 4,
    chunksize: int = 10000
):
    """
    Process all data for a single exchange using chunked reading.

    Args:
        exchange: Exchange name (e.g., 'binc', 'gmni')
        base_data_dir: Base data directory containing orderbook/ and trades/
        output_dir: Output directory for events
        k_levels: Number of orderbook levels to track
        n_workers: Number of parallel workers
        chunksize: Rows to process per chunk
    """
    print(f"\n{'='*60}", flush=True)
    print(f"Processing {exchange.upper()} with chunked reading", flush=True)
    print(f"  Workers: {n_workers}, Chunk size: {chunksize} rows", flush=True)
    print(f"{'='*60}", flush=True)

    # Find all orderbook files for this exchange
    orderbook_dir = os.path.join(base_data_dir, "orderbook", exchange)
    orderbook_files = glob.glob(
        os.path.join(orderbook_dir, "**/full_order_book_*.csv.gz"),
        recursive=True
    )

    if not orderbook_files:
        print(f"No orderbook files found for {exchange}", flush=True)
        return

    # Build task list
    tasks = []
    for ob_file in orderbook_files:
        # Extract coin/date info from filename
        filename = os.path.basename(ob_file)
        parts = filename.replace("full_order_book_", "").replace(".csv.gz", "").split("_")

        if len(parts) >= 4:
            coin = parts[2]
            date = parts[3]

            # Find matching trades file
            trades_file = ob_file.replace("orderbook", "trades").replace("full_order_book_", "trades_")

            if os.path.exists(trades_file):
                # Create output file path (gzipped — see process_data_files_chunked)
                output_file = os.path.join(
                    output_dir,
                    f"events_{exchange}_{coin}_{date}.jsonl.gz"
                )

                tasks.append((ob_file, trades_file, output_file, f"{exchange}_{coin}_{date}", chunksize))
            else:
                print(f"⚠️ Missing trades file for {coin}/{date}", flush=True)

    if not tasks:
        print(f"No valid file pairs found for {exchange}", flush=True)
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(tasks)} file pairs to process", flush=True)

    # Process with workers
    start_time = time.time()
    successful = 0
    failed = 0

    if n_workers == 1:
        # Single worker mode
        for task in tasks:
            result = process_single_file_worker(task)
            if result[1]:
                successful += 1
            else:
                failed += 1
    else:
        # Multi-worker mode
        with Pool(n_workers) as pool:
            results = pool.map(process_single_file_worker, tasks, chunksize=1)

            for output_file, success, message in results:
                if success:
                    successful += 1
                else:
                    failed += 1
                    print(f"  Failed: {output_file} - {message}", flush=True)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}", flush=True)
    print(f"Completed {exchange.upper()} processing in {elapsed/60:.1f} minutes", flush=True)
    print(f"  Successful: {successful}/{len(tasks)}", flush=True)
    if failed > 0:
        print(f"  Failed: {failed}/{len(tasks)}", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"{'='*60}", flush=True)


def batch_process_all_chunked() -> None:
    print("\n" + "=" * 70, flush=True)
    print("BATCH PROCESSING ALL DATA (CHUNKED MODE)", flush=True)
    print("=" * 70, flush=True)

    data_dir = f"{project_root}/data"
    output_dir = "/SAN/medic/TFOW/data/events"

    orderbook_dir = os.path.join(data_dir, "orderbook")
    exchanges = [
        d for d in os.listdir(orderbook_dir)
        if os.path.isdir(os.path.join(orderbook_dir, d))
    ]

    print(f"\nFound exchanges: {', '.join(exchanges)}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)

    # Use NSLOTS but cap to protect memory
    n_workers = get_n_workers(default=8, cap=4)
    print(f"\nUsing {n_workers} workers (NSLOTS={os.environ.get('NSLOTS', 'N/A')})", flush=True)

    # Determine chunk size based on available memory
    # Larger chunks = more memory but faster processing
    # Smaller chunks = less memory but slower processing
    chunksize = 10000  # Conservative default for HPC
    print(f"Using chunk size: {chunksize} rows per chunk", flush=True)

    for exchange in exchanges:
        print(f"\n📊 Processing {exchange.upper()}...", flush=True)
        exchange_output = os.path.join(output_dir, exchange)
        try:
            batch_process_exchange_data_chunked(
                exchange=exchange,
                base_data_dir=data_dir,
                output_dir=exchange_output,
                k_levels=10,
                n_workers=n_workers,
                chunksize=chunksize
            )
            print(f"✅ Completed {exchange}", flush=True)
        except Exception as e:
            print(f"❌ Error processing {exchange}: {e}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("✅ BATCH PROCESSING COMPLETE!", flush=True)
    print(f"Events saved to: {output_dir}", flush=True)
    print("=" * 70, flush=True)


def main() -> None:
    print("\n" + "=" * 70, flush=True)
    print("EVENT CONSTRUCTION FROM ORDER BOOK & TRADE DATA", flush=True)
    print("CHUNKED VERSION FOR HPC - PREVENTS OOM", flush=True)
    print("=" * 70, flush=True)

    if not check_data_alignment():
        print("\n❌ Cannot proceed without complete trade data!", flush=True)
        sys.exit(1)

    batch_process_all_chunked()


if __name__ == "__main__":
    main()