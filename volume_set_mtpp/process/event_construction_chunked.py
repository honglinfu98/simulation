"""
Event construction from LOB and trade data - CHUNKED HPC VERSION.

This module processes limit order book (LOB) snapshots and trade prints to construct
event sequences for market microstructure analysis.

KEY FEATURES:
1. Chunked reading to prevent OOM on HPC
2. Prevents "revealed depth" bug by tracking full orderbook state
3. Correct tick-based IS level calculation
4. Parse cache for performance
5. JSONL streaming to prevent memory issues and handle partial outputs
"""

import ast
import bisect
import gzip
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, TextIO
import pandas as pd


class EventType(Enum):
    """Types of order book events."""
    MO = "MO"  # Market Order
    CO = "CO"  # Cancel Order
    LO = "LO"  # Limit Order
    IS = "IS"  # Inside-Spread Order


class Side(Enum):
    """Order side."""
    BID = "b"
    ASK = "a"


@dataclass(frozen=True)
class Event:
    """Represents a single order book event."""
    event_type: EventType
    side: Side
    level: int  # depth index (1 for best quote, 2 for second best, etc.)
    volume: float
    price: float
    timestamp: int

    def __lt__(self, other):
        """Define ordering for events based on market impact priority."""
        type_priority = {EventType.MO: 0, EventType.CO: 1, EventType.LO: 2, EventType.IS: 3}

        if type_priority[self.event_type] != type_priority[other.event_type]:
            return type_priority[self.event_type] < type_priority[other.event_type]

        if self.side != other.side:
            return self.side == Side.BID

        return self.level < other.level


class LOBSnapshot:
    """Represents a limit order book snapshot at a given time."""

    def __init__(self, timestamp: int, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]):
        """
        Initialize LOB snapshot.

        Args:
            timestamp: Timestamp in nanoseconds
            bids: List of (price, volume) tuples sorted by price descending
            asks: List of (price, volume) tuples sorted by price ascending
        """
        self.timestamp = timestamp
        self.bids = bids
        self.asks = asks

    def get_best_bid(self) -> Optional[Tuple[float, float]]:
        """Get best bid (price, volume) or None if no bids."""
        return self.bids[0] if self.bids else None

    def get_best_ask(self) -> Optional[Tuple[float, float]]:
        """Get best ask (price, volume) or None if no asks."""
        return self.asks[0] if self.asks else None

    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread or None if either side is empty."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return best_ask[0] - best_bid[0]
        return None

    def get_top_k_levels(self, side: Side, k: int) -> Dict[float, float]:
        """Get top k price levels for a given side as price->volume dict."""
        if side == Side.BID:
            levels = self.bids[:k]
        else:
            levels = self.asks[:k]
        return {price: volume for price, volume in levels}


class EventConstructor:
    """Constructs events from LOB updates and trades with chunked processing."""

    def __init__(self, k_levels: int = 10, fast_mode: bool = False, tick_size: Optional[float] = None):
        self.k_levels = k_levels
        self.fast_mode = fast_mode
        self.tick_size = tick_size

    @staticmethod
    def detect_tick_size(trade_prices: List[float]) -> float:
        if not trade_prices:
            return 0.00001
        tick_candidates = []
        for price in trade_prices:
            price_str = f"{price:.10f}".rstrip("0")
            if "." in price_str:
                dec = price_str.split(".")[1]
                if dec:
                    tick_candidates.append(10 ** (-len(dec)))
        return min(tick_candidates) if tick_candidates else 1.0

    def parse_trades_data(self, df: pd.DataFrame) -> Dict[int, List[Dict]]:
        trades_by_time = defaultdict(list)
        dates = df["date"].values
        prices = df["price"].values
        amounts = df["amount"].values
        sells = df["sell"].values

        for i in range(len(df)):
            ts_ns = int(dates[i]) * 1_000_000  # ms -> ns
            trades_by_time[ts_ns].append(
                {
                    "price": float(prices[i]),
                    "volume": float(amounts[i]),
                    "side": Side.ASK if sells[i] else Side.BID,  # sell=True means hitting bids
                    "timestamp": ts_ns,
                }
            )
        return dict(trades_by_time)

    def _apply_side_updates_return_deltas(
        self,
        book: Dict[float, float],
        updates: List[Tuple],
    ) -> Dict[float, float]:
        """
        Apply updates (price, volume) to a full-book dict and return exact deltas per price.
        This prevents the "revealed depth" bug (e.g., level 12 -> level 9) being misread as a new LO.
        """
        deltas: Dict[float, float] = {}
        for p_str, v_str in updates:
            price = float(p_str)
            new_vol = float(v_str)
            old_vol = float(book.get(price, 0.0))

            if new_vol == 0.0:
                if price in book:
                    book.pop(price, None)
                # delta is -old_vol (if it existed), else 0
                if old_vol != 0.0:
                    deltas[price] = deltas.get(price, 0.0) - old_vol
            else:
                book[price] = new_vol
                dv = new_vol - old_vol
                if abs(dv) > 1e-12:
                    deltas[price] = deltas.get(price, 0.0) + dv

        return deltas

    def _top_k_snapshot_from_full_books(
        self,
        timestamp: int,
        bids_full: Dict[float, float],
        asks_full: Dict[float, float],
    ) -> LOBSnapshot:
        """
        Build a snapshot that only contains TOP-K levels on each side, derived from full dict state.
        Full dict state is maintained separately for correctness.
        """
        # sort prices only (values fetched from dict) and slice top-k
        bid_prices = sorted(bids_full.keys(), reverse=True)[: self.k_levels]
        ask_prices = sorted(asks_full.keys())[: self.k_levels]

        bids = [(p, float(bids_full[p])) for p in bid_prices]
        asks = [(p, float(asks_full[p])) for p in ask_prices]
        return LOBSnapshot(timestamp, bids, asks)

    def classify_events_from_deltas(
        self,
        prev_snapshot: LOBSnapshot,
        curr_snapshot: LOBSnapshot,
        trades: List[Dict],
        bid_deltas: Dict[float, float],
        ask_deltas: Dict[float, float],
    ) -> List[Event]:
        """
        Same intent as your classify_events(), but LO/CO/IS come from deltas of UPDATED PRICES only.
        FIXED: Now uses correct tick-based IS level calculation.
        """
        if prev_snapshot.timestamp == curr_snapshot.timestamp:
            return []

        events: List[Event] = []
        timestamp = curr_snapshot.timestamp

        # ---- Step 1: MO directly from trades (your approach) ----
        trade_volumes_by_price_and_side = defaultdict(float)

        for trade in trades:
            price = float(trade["price"])
            volume = float(trade["volume"])

            # sell=True => trade.side==ASK => hit bids
            if trade["side"] == Side.ASK:
                hit_side = Side.BID
            else:
                hit_side = Side.ASK

            events.append(
                Event(
                    event_type=EventType.MO,
                    side=hit_side,
                    level=1,
                    volume=volume,
                    price=price,
                    timestamp=timestamp,
                )
            )
            trade_volumes_by_price_and_side[(price, hit_side)] += volume

        # ---- Spread info for IS detection ----
        prev_best_bid = prev_snapshot.get_best_bid()
        prev_best_ask = prev_snapshot.get_best_ask()
        has_prev_spread = bool(prev_best_bid and prev_best_ask)
        prev_bid_price = prev_best_bid[0] if prev_best_bid else 0.0
        prev_ask_price = prev_best_ask[0] if prev_best_ask else float("inf")

        # ---- Step 2: LO/CO/IS from UPDATED prices only ----
        for side, deltas in [(Side.BID, bid_deltas), (Side.ASK, ask_deltas)]:
            if not deltas:
                continue

            # Build price->level mapping from *current* TOP-K snapshot only.
            curr_levels = curr_snapshot.get_top_k_levels(side, self.k_levels)
            if side == Side.BID:
                sorted_prices = sorted(curr_levels.keys(), reverse=True)
            else:
                sorted_prices = sorted(curr_levels.keys())
            price_to_level = {p: i + 1 for i, p in enumerate(sorted_prices)}

            for price, dv in deltas.items():
                # IMPORTANT: only emit events if this updated price is actually inside TOP-K now
                level = price_to_level.get(price)
                if level is None:
                    continue  # updated, but outside top-k -> you drop it by design

                if dv < 0:
                    # reduction: trade-part vs cancel-part
                    abs_reduction = -dv
                    trade_vol_at_price = trade_volumes_by_price_and_side.get((price, side), 0.0)
                    remaining = abs_reduction - trade_vol_at_price

                    if remaining > 1e-12:
                        events.append(
                            Event(
                                event_type=EventType.CO,
                                side=side,
                                level=level,
                                volume=float(remaining),
                                price=float(price),
                                timestamp=timestamp,
                            )
                        )

                elif dv > 0:
                    # addition: IS vs LO
                    if (not self.fast_mode) and has_prev_spread and (prev_bid_price < price < prev_ask_price):
                        # FIXED: Use tick-based IS level calculation
                        if self.tick_size and self.tick_size > 0:
                            if side == Side.BID:
                                # For bids: measure ticks from best bid
                                ticks_from_best = (price - prev_bid_price) / self.tick_size
                            else:
                                # For asks: measure ticks from best ask
                                ticks_from_best = (prev_ask_price - price) / self.tick_size

                            # IS level is the number of ticks from best price
                            is_level = max(1, int(round(ticks_from_best)))

                            # Validate tick alignment (use relative tolerance for large tick counts)
                            tick_frac = abs(ticks_from_best - round(ticks_from_best))
                            if tick_frac > 0.4:
                                raise ValueError(
                                    f"Price {price} not multiple of tick {self.tick_size}. ticks={ticks_from_best}"
                                )
                        else:
                            # Fallback if no tick size
                            is_level = 1

                        events.append(
                            Event(
                                event_type=EventType.IS,
                                side=side,
                                level=is_level,
                                volume=float(dv),
                                price=float(price),
                                timestamp=timestamp,
                            )
                        )
                    else:
                        events.append(
                            Event(
                                event_type=EventType.LO,
                                side=side,
                                level=level,
                                volume=float(dv),
                                price=float(price),
                                timestamp=timestamp,
                            )
                        )

        return sorted(events)

    def process_orderbook_chunk(
        self,
        chunk: pd.DataFrame,
        trades_by_time: Dict[int, List[Dict]],
        current_bids: Dict[float, float],
        current_asks: Dict[float, float],
        prev_snapshot: Optional[LOBSnapshot],
        first_update: bool,
        parse_cache: Dict,
        stream_output: Optional[TextIO],
        write_interval: int,
        jsonl_mode: bool,
        written_sets: int,
        events_created: int,
        skipped_identical: int,
        rows_processed: int
    ) -> Tuple[Optional[LOBSnapshot], bool, int, int, int, int]:
        """
        Process a chunk of orderbook data.

        Returns:
            Tuple of (prev_snapshot, first_update, written_sets, events_created, skipped_identical, rows_processed)
        """
        MAX_CACHE_SIZE = 10000
        trade_times = sorted(trades_by_time.keys())

        # Convert to numpy arrays for faster iteration
        timestamps = chunk['timestamp'].values
        types = chunk['type'].values
        asks_data = chunk['asks'].values
        bids_data = chunk['bids'].values

        for idx in range(len(chunk)):
            ts = timestamps[idx]
            update_type = types[idx]
            asks_str = asks_data[idx]
            bids_str = bids_data[idx]
            rows_processed += 1

            # Parse with cache
            if asks_str in parse_cache:
                asks_updates = parse_cache[asks_str]
            else:
                try:
                    asks_updates = ast.literal_eval(asks_str) if asks_str != '[]' else []
                    if len(parse_cache) < MAX_CACHE_SIZE:
                        parse_cache[asks_str] = asks_updates
                except:
                    asks_updates = []

            if bids_str in parse_cache:
                bids_updates = parse_cache[bids_str]
            else:
                try:
                    bids_updates = ast.literal_eval(bids_str) if bids_str != '[]' else []
                    if len(parse_cache) < MAX_CACHE_SIZE:
                        parse_cache[bids_str] = bids_updates
                except:
                    bids_updates = []

            # Apply updates and get deltas
            bid_deltas: Dict[float, float] = {}
            ask_deltas: Dict[float, float] = {}

            if update_type == 'u' and not first_update:
                # incremental update
                bid_deltas = self._apply_side_updates_return_deltas(current_bids, bids_updates)
                ask_deltas = self._apply_side_updates_return_deltas(current_asks, asks_updates)

            elif update_type == 's' or first_update:
                # snapshot boundary: full replace
                current_bids.clear()
                current_asks.clear()
                current_bids.update({float(p): float(v) for p, v in bids_updates if float(v) > 0})
                current_asks.update({float(p): float(v) for p, v in asks_updates if float(v) > 0})
                bid_deltas, ask_deltas = {}, {}
                if first_update:
                    first_update = False

            # Build TOP-K snapshots from full dict state
            curr_snapshot = self._top_k_snapshot_from_full_books(ts, current_bids, current_asks)

            if prev_snapshot is None:
                prev_snapshot = curr_snapshot
                continue

            if prev_snapshot.timestamp == curr_snapshot.timestamp:
                skipped_identical += 1
                prev_snapshot = curr_snapshot
                continue

            # Get trades in (prev_ts, curr_ts]. Exact-timestamp matching loses
            # almost all MOs: Kaiko trade timestamps (arbitrary ms) rarely
            # coincide with ~100ms-spaced book update timestamps, so trade
            # volume was being misclassified as cancels.
            lo = bisect.bisect_right(trade_times, prev_snapshot.timestamp)
            hi = bisect.bisect_right(trade_times, curr_snapshot.timestamp)
            trades = [t for k in trade_times[lo:hi] for t in trades_by_time[k]]

            # Classify events
            try:
                events = self.classify_events_from_deltas(
                    prev_snapshot=prev_snapshot,
                    curr_snapshot=curr_snapshot,
                    trades=trades,
                    bid_deltas=bid_deltas,
                    ask_deltas=ask_deltas,
                )
            except ValueError as e:
                if "tick size" in str(e).lower():
                    prev_snapshot = curr_snapshot
                    continue
                raise

            if events:
                event_set = set(events)
                events_created += len(events)

                if stream_output:
                    # JSONL format - one complete JSON object per line
                    event_data = {
                        'timestamp': int(curr_snapshot.timestamp),  # Convert numpy int64 to int
                        'event_count': len(event_set),
                        'events': []
                    }

                    for event in sorted(event_set):
                        event_data['events'].append({
                            'event_type': event.event_type.value,
                            'side': event.side.value,
                            'level': int(event.level),  # Convert to int
                            'volume': float(event.volume),  # Convert to float
                            'price': float(event.price),  # Convert to float
                            'timestamp': int(event.timestamp)  # Convert to int
                        })

                    # LOB state: top-k book AFTER applying this update. Row t's book is
                    # the conditioning context for generating the event set at row t+1
                    # (MarS-style: order tokens conditioned on k-level LOB volumes + mid).
                    bid_prices_sorted = sorted(current_bids.keys(), reverse=True)[:self.k_levels]
                    ask_prices_sorted = sorted(current_asks.keys())[:self.k_levels]
                    total_bid = sum(current_bids[p] for p in bid_prices_sorted)
                    total_ask = sum(current_asks[p] for p in ask_prices_sorted)
                    if total_bid + total_ask > 0:
                        imbalance = float((total_ask - total_bid) / (total_ask + total_bid))
                    else:
                        imbalance = 0.0
                    if imbalance < -0.4:
                        lob_discrete_state = 0
                    elif imbalance > 0.4:
                        lob_discrete_state = 2
                    else:
                        lob_discrete_state = 1
                    if bid_prices_sorted and ask_prices_sorted:
                        mid = round((bid_prices_sorted[0] + ask_prices_sorted[0]) / 2.0, 10)
                    else:
                        mid = None
                    event_data['lob_state'] = {
                        'imbalance': round(imbalance, 6),
                        'state': lob_discrete_state,
                        'mid': mid,
                        'bids': [[p, round(float(current_bids[p]), 8)] for p in bid_prices_sorted],
                        'asks': [[p, round(float(current_asks[p]), 8)] for p in ask_prices_sorted],
                    }

                    if jsonl_mode:
                        # Write as single line JSON (JSONL format)
                        stream_output.write(json.dumps(event_data) + '\n')
                    else:
                        # Legacy format (single large JSON array)
                        if written_sets == 0:
                            stream_output.write('{"events": [\n')
                        else:
                            stream_output.write(',\n')
                        stream_output.write(json.dumps(event_data['events']))

                    written_sets += 1

                    # Flush based on written_sets counter
                    if written_sets % write_interval == 0:
                        stream_output.flush()
                        print(f"  Flushed after {written_sets} event sets")

            prev_snapshot = curr_snapshot

            # Clear cache based on rows_processed counter
            if rows_processed % 10000 == 0 and len(parse_cache) >= MAX_CACHE_SIZE:
                parse_cache.clear()
                print(f"  Cleared cache after {rows_processed} rows")

            if rows_processed % 5000 == 0:
                print(f"  Processed {rows_processed} rows... events={events_created}")
                if rows_processed % 50000 == 0:
                    print(f"    Cache size: {len(parse_cache)}, Bids: {len(current_bids)}, Asks: {len(current_asks)}")

        return prev_snapshot, first_update, written_sets, events_created, skipped_identical, rows_processed

    def construct_events_chunked(
        self,
        orderbook_file: str,
        trades_df: pd.DataFrame,
        chunksize: int = 10000,
        max_rows: Optional[int] = None,
        stream_output: Optional[TextIO] = None,
        write_interval: int = 1000,
        jsonl_mode: bool = True
    ) -> None:
        """
        CHUNKED VERSION for HPC - processes orderbook in chunks to prevent OOM.

        Args:
            orderbook_file: Path to orderbook CSV file
            trades_df: Trades dataframe (still loaded in memory - usually much smaller)
            chunksize: Number of rows to process at once
            max_rows: Limit total rows for testing
            stream_output: Optional file handle for streaming output
            write_interval: Flush to disk every N event sets
            jsonl_mode: If True, write JSONL format (recommended for HPC)
        """
        print(f"Processing orderbook in chunks of {chunksize} rows")
        if max_rows:
            print(f"Limiting to {max_rows} total rows for testing")

        # Parse trades (these are typically much smaller)
        trades_by_time = self.parse_trades_data(trades_df)

        # Detect tick if needed
        if self.tick_size is None:
            all_prices = list(trades_df['price'].values) if len(trades_df) > 0 else []
            self.tick_size = self.detect_tick_size(all_prices)
            print(f"Detected tick size: {self.tick_size}")

        # State that persists across chunks
        current_bids: Dict[float, float] = {}
        current_asks: Dict[float, float] = {}
        first_update = True
        prev_snapshot = None
        parse_cache = {}

        # Counters
        written_sets = 0
        events_created = 0
        skipped_identical = 0
        rows_processed = 0

        # Process orderbook in chunks
        compression = 'gzip' if orderbook_file.endswith('.gz') else None

        chunk_iterator = pd.read_csv(
            orderbook_file,
            compression=compression,
            sep=';',
            chunksize=chunksize,
            nrows=max_rows  # This limits total rows if specified
        )

        for chunk_idx, chunk in enumerate(chunk_iterator):
            print(f"Processing chunk {chunk_idx + 1} ({len(chunk)} rows)...")

            # Process this chunk
            prev_snapshot, first_update, written_sets, events_created, skipped_identical, rows_processed = \
                self.process_orderbook_chunk(
                    chunk=chunk,
                    trades_by_time=trades_by_time,
                    current_bids=current_bids,
                    current_asks=current_asks,
                    prev_snapshot=prev_snapshot,
                    first_update=first_update,
                    parse_cache=parse_cache,
                    stream_output=stream_output,
                    write_interval=write_interval,
                    jsonl_mode=jsonl_mode,
                    written_sets=written_sets,
                    events_created=events_created,
                    skipped_identical=skipped_identical,
                    rows_processed=rows_processed
                )

            # Check if we've hit the max_rows limit
            if max_rows and rows_processed >= max_rows:
                print(f"Reached max_rows limit ({max_rows})")
                break

        # Close JSON if using legacy format
        if stream_output and not jsonl_mode:
            stream_output.write(f'\n], "metadata": {{"total_events": {events_created}, "skipped_identical": {skipped_identical}}}}}\n')

        if stream_output:
            stream_output.flush()
            print(f"Stream complete: {written_sets} event sets written")

        print("Finished constructing events:")
        print(f"  Total events created: {events_created}")
        print(f"  Total rows processed: {rows_processed}")
        print(f"  Skipped identical timestamps: {skipped_identical}")


def process_data_files_chunked(
    orderbook_file: str,
    trades_file: str,
    k_levels: int = 10,
    chunksize: int = 10000,
    max_rows: Optional[int] = None,
    tick_size: Optional[float] = None,
    output_file: Optional[str] = None,
    jsonl_format: bool = True
) -> None:
    """
    Process orderbook and trade files using chunked reading for HPC.

    Args:
        orderbook_file: Path to orderbook CSV file
        trades_file: Path to trades CSV file
        k_levels: Number of price levels to track
        chunksize: Number of orderbook rows to process at once
        max_rows: Optional limit on total rows to process (for testing)
        tick_size: Optional tick size (will be detected if None)
        output_file: Output file for streaming mode
        jsonl_format: If True, use JSONL format (recommended for HPC)
    """
    print(f"Processing files (chunked mode):")
    print(f"  Orderbook: {orderbook_file}")
    print(f"  Trades: {trades_file}")
    print(f"  Chunk size: {chunksize} rows")
    format_type = "JSONL" if jsonl_format else "JSON"
    print(f"  Output: {output_file} (format: {format_type})")

    # Read trades data (typically much smaller than orderbook)
    print("Reading trades data...")
    trades_df = pd.read_csv(trades_file, compression='gzip' if trades_file.endswith('.gz') else None)
    print(f"  Loaded {len(trades_df)} trades")

    # Construct events with chunked processing
    print("Constructing events (chunked processing)...")
    constructor = EventConstructor(k_levels=k_levels, fast_mode=False, tick_size=tick_size)

    # Stream to file (gzip-compressed if the filename says so — uncompressed
    # event JSONL is ~9x the size of the gzipped raw input)
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    _opener = (lambda p: gzip.open(p, 'wt')) if output_file.endswith('.gz') else (lambda p: open(p, 'w'))
    with _opener(output_file) as f:
        constructor.construct_events_chunked(
            orderbook_file=orderbook_file,
            trades_df=trades_df,
            chunksize=chunksize,
            max_rows=max_rows,
            stream_output=f,
            jsonl_mode=jsonl_format
        )

    print(f"Events streamed to: {output_file}")


def test_chunked_processing():
    """Test chunked processing for HPC."""

    orderbook_file = '/Users/honglinfu/ucl/volume-set-mtpp/data/orderbook/gmni/spot/ethusdt/full_order_book_gmni_spot_ethusdt_2026-01-02.csv.gz'
    trades_file = '/Users/honglinfu/ucl/volume-set-mtpp/data/trades/gmni/spot/ethusdt/trades_gmni_spot_ethusdt_2026-01-02.csv.gz'

    print("=" * 70)
    print("TESTING CHUNKED EVENT CONSTRUCTION FOR HPC")
    print("=" * 70)

    # Test with small chunks to verify chunking works
    print("\n1. Small chunk test (1000 rows per chunk, 5000 total):")
    output_file = '/Users/honglinfu/ucl/volume-set-mtpp/test_events_chunked.jsonl'

    process_data_files_chunked(
        orderbook_file=orderbook_file,
        trades_file=trades_file,
        k_levels=10,
        chunksize=1000,  # Small chunks for testing
        max_rows=5000,    # Limit total for test
        output_file=output_file,
        jsonl_format=True
    )

    # Verify output
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            lines = f.readlines()
            print(f"\nJSONL file has {len(lines)} event sets")
            if lines:
                first_line = json.loads(lines[0])
                print(f"First event set: timestamp={first_line['timestamp']}, "
                      f"events={first_line['event_count']}")
                last_line = json.loads(lines[-1])
                print(f"Last event set: timestamp={last_line['timestamp']}, "
                      f"events={last_line['event_count']}")

    print("\n" + "=" * 70)
    print("✅ Chunked processing test completed successfully!")
    print("This approach prevents OOM on HPC by processing orderbook in chunks")
    print("=" * 70)


if __name__ == "__main__":
    test_chunked_processing()