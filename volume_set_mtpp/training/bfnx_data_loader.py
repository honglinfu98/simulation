"""
Data loader for BFNX event sequences.
Loads processed JSONL event files and prepares tensorized sliding windows for training.
"""

import glob
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


TOTAL_EVENT_TYPES = 62


def _fixed_bfnx_event_names() -> List[str]:
    """Canonical 62-channel LOB event schema.

    Layout:
      - LO bid/ask levels 1..10: 20 channels
      - CO bid/ask levels 1..10: 20 channels
      - MO bid/ask level 1:      2 channels
      - IS bid/ask levels 1..10: 20 channels

    The active BFNX JSONL files may not observe every canonical channel in a
    short date range.  We still keep the zero columns so model checkpoints and
    metrics use the paper/reference 62-event schema rather than a dynamic
    observed-only mapping.
    """
    names: List[str] = []
    for event_type in ("LO", "CO"):
        for side in ("b", "a"):
            for level in range(1, 11):
                names.append(f"{event_type}_{side}_L{level}")
    for side in ("b", "a"):
        names.append(f"MO_{side}_L1")
    for side in ("b", "a"):
        for level in range(1, 11):
            names.append(f"IS_{side}_L{level}")
    assert len(names) == TOTAL_EVENT_TYPES, len(names)
    return names


@dataclass
class EventMapping:
    """Mapping between event types and indices"""
    event_to_idx: Dict[str, int]
    idx_to_event: Dict[int, str]
    num_events: int


def create_event_mapping(event_types: Set[str], fixed_62: bool = True) -> EventMapping:
    """Create bidirectional mapping for event types.

    By default BFNX uses the fixed 62-channel LOB schema for publication-quality
    comparisons.  Dynamic observed-only mappings can hide absent channels and
    make runs across date ranges incomparable.
    """
    sorted_events = _fixed_bfnx_event_names() if fixed_62 else sorted(event_types)
    event_to_idx = {event: idx for idx, event in enumerate(sorted_events)}
    idx_to_event = {idx: event for event, idx in event_to_idx.items()}
    return EventMapping(event_to_idx, idx_to_event, len(sorted_events))


def _selected_jsonl_files(data_dir: str, max_files: Optional[int] = None) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")) +
                   glob.glob(os.path.join(data_dir, "*.jsonl.gz")))
    if max_files:
        files = files[:max_files]
    if not files:
        raise ValueError(f"No JSONL files found in {data_dir}")
    return files


def _open_jsonl(path: str):
    if path.endswith(".gz"):
        import gzip
        return gzip.open(path, "rt")
    return open(path, "r")


LOB_FEATURE_DIM = 6


def _lob_features_from_state(ls: dict) -> List[float]:
    """Continuous book features from the constructor's lob_state.

    [imbalance, log1p(bid depth sum), log1p(ask depth sum),
     log1p(best-bid vol), log1p(best-ask vol), spread in bps of mid]
    All reproducible from a replayed book at simulation time, which is what
    allows closed-loop state conditioning.  Old-format files (no ladders)
    yield zeros except the scalar imbalance.
    """
    import math as _math
    bids = ls.get("bids") or []
    asks = ls.get("asks") or []
    imb = float(ls.get("imbalance", 0.0))
    if not bids or not asks:
        return [imb, 0.0, 0.0, 0.0, 0.0, 0.0]
    bid_sum = sum(v for _, v in bids)
    ask_sum = sum(v for _, v in asks)
    b1p, b1v = bids[0]
    a1p, a1v = asks[0]
    mid = ls.get("mid") or (0.5 * (b1p + a1p))
    spread_bps = ((a1p - b1p) / mid * 1e4) if mid else 0.0
    return [imb, _math.log1p(bid_sum), _math.log1p(ask_sum),
            _math.log1p(b1v), _math.log1p(a1v), float(spread_bps)]


def _cache_path(data_dir: str, files: List[str], max_files: Optional[int], cache_dir: Optional[str]) -> str:
    """Build a cache filename that changes when selected files change."""
    if cache_dir is None:
        cache_dir = os.path.join(data_dir, ".tensor_cache")
    os.makedirs(cache_dir, exist_ok=True)

    h = hashlib.sha1()
    h.update(os.path.abspath(data_dir).encode())
    h.update(str(max_files).encode())
    for path in files:
        st = os.stat(path)
        h.update(os.path.basename(path).encode())
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime_ns)).encode())
    return os.path.join(cache_dir, f"bfnx_tensor_cache_{h.hexdigest()[:16]}.pt")


def _parse_bfnx_jsonl(files: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, EventMapping, np.ndarray]:
    """Parse JSONL once into contiguous per-timestamp tensors.

    Returns:
        time_deltas: [N] float32 seconds
        marks: [N, C] float32 multi-hot labels
        volumes: [N, C] float32 log1p volumes
        lob_states: [N] int64 state ids
        lob_features: [N, LOB_FEATURE_DIM] float32
        event_mapping
        file_lengths: [num_files] int64 events per source file (segment lengths)
    """
    all_timestamps: List[float] = []
    all_events: List[List[Tuple[str, float]]] = []
    all_event_types: Set[str] = set()
    all_lob_states: List[int] = []
    all_lob_features: List[List[float]] = []
    file_lengths: List[int] = []

    print(f"Loading {len(files)} BFNX event files from JSONL...")
    for file_path in tqdm(files, desc="Loading JSONL"):
        n_before = len(all_timestamps)
        with _open_jsonl(file_path) as f:
            for line in f:
                data = json.loads(line)
                timestamp_events = []
                for event in data["events"]:
                    event_type = f"{event['event_type']}_{event['side']}_L{event['level']}"
                    all_event_types.add(event_type)
                    timestamp_events.append((event_type, event["volume"]))

                if timestamp_events:
                    all_timestamps.append(data["timestamp"] / 1e9)
                    all_events.append(timestamp_events)
                    lob = data.get("lob_state", {})
                    all_lob_states.append(int(lob.get("state", 1)))
                    all_lob_features.append(_lob_features_from_state(lob))
        file_lengths.append(len(all_timestamps) - n_before)

    event_mapping = create_event_mapping(all_event_types, fixed_62=True)
    missing_canonical = sorted(set(event_mapping.event_to_idx) - all_event_types)
    out_of_schema = sorted(all_event_types - set(event_mapping.event_to_idx))
    if missing_canonical:
        print(f"Fixed 62-channel schema: {len(missing_canonical)} canonical channels absent in selected data")
    if out_of_schema:
        print(f"Fixed 62-channel schema: skipping {len(out_of_schema)} out-of-schema observed event types: {out_of_schema[:20]}")
    n = len(all_timestamps)
    c = event_mapping.num_events

    timestamps = np.asarray(all_timestamps, dtype=np.float64)
    # Per-file deltas: never difference across file/asset boundaries (the old
    # global np.diff produced negative deltas at btc->eth->sol transitions and
    # spurious day gaps).  First event of every file gets dt = 0.
    time_deltas = np.zeros(n, dtype=np.float32)
    pos = 0
    for fl in file_lengths:
        if fl > 1:
            time_deltas[pos + 1: pos + fl] = np.diff(timestamps[pos: pos + fl]).astype(np.float32)
        pos += fl
    time_deltas = np.clip(time_deltas, 0.0, None)
    marks = np.zeros((n, c), dtype=np.float32)
    volumes = np.zeros((n, c), dtype=np.float32)

    for row_idx, events in enumerate(tqdm(all_events, desc="Tensorizing marks/volumes")):
        for event_type, volume in events:
            col_idx = event_mapping.event_to_idx.get(event_type)
            if col_idx is None:
                continue
            marks[row_idx, col_idx] = 1.0
            volumes[row_idx, col_idx] = np.log1p(float(volume))

    lob_states = np.asarray(all_lob_states, dtype=np.int64)
    lob_features = np.asarray(all_lob_features, dtype=np.float32)
    file_lengths_np = np.asarray(file_lengths, dtype=np.int64)
    print(f"Loaded {n} timestamps with {c} unique event types")
    return time_deltas, marks, volumes, lob_states, lob_features, event_mapping, file_lengths_np


def load_bfnx_tensors(
    data_dir: str,
    max_files: Optional[int] = None,
    cache_dir: Optional[str] = None,
    rebuild_cache: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, EventMapping, torch.Tensor]:
    """Load BFNX data as contiguous tensors, using an on-disk cache when possible.

    Returns (time_deltas, marks, volumes, lob_states, lob_features,
    event_mapping, file_lengths).  ``file_lengths`` is an int64 tensor with the
    number of events contributed by each source file; old caches without the
    key fall back to a single segment covering the full stream.
    """
    # Escape hatch: load a known-good tensor cache directly, bypassing the
    # source-file stat used for cache keying.  Needed when the source JSONL
    # files have been compressed/archived (e.g. gzip cleanup on /SAN) but the
    # tensor cache itself is intact.
    override = os.environ.get("BFNX_CACHE_FILE")
    if override and os.path.exists(override) and not rebuild_cache:
        start = time.time()
        print(f"Loading tensorized BFNX cache (BFNX_CACHE_FILE override): {override}")
        try:
            payload = torch.load(override, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(override, map_location="cpu")
        print(f"Loaded tensor cache in {time.time() - start:.2f}s")
        lf = payload.get("lob_features")
        if lf is None:
            lf = torch.zeros(payload["time_deltas"].shape[0], LOB_FEATURE_DIM)
        fl = payload.get("file_lengths")
        if fl is None:
            # Old cache without per-file segment info: single segment = [N].
            fl = torch.tensor([int(payload["time_deltas"].shape[0])], dtype=torch.long)
        return (
            payload["time_deltas"],
            payload["marks"],
            payload["volumes"],
            payload["lob_states"],
            lf,
            payload["event_mapping"],
            fl,
        )

    files = _selected_jsonl_files(data_dir, max_files)
    cache_file = _cache_path(data_dir, files, max_files, cache_dir)

    if os.path.exists(cache_file) and not rebuild_cache:
        start = time.time()
        print(f"Loading tensorized BFNX cache: {cache_file}")
        try:
            payload = torch.load(cache_file, map_location="cpu", weights_only=False)
        except TypeError:  # Older torch without weights_only
            payload = torch.load(cache_file, map_location="cpu")
        print(f"Loaded tensor cache in {time.time() - start:.2f}s")
        lf = payload.get("lob_features")
        if lf is None:
            lf = torch.zeros(payload["time_deltas"].shape[0], LOB_FEATURE_DIM)
        fl = payload.get("file_lengths")
        if fl is None:
            # Old cache without per-file segment info: single segment = [N].
            fl = torch.tensor([int(payload["time_deltas"].shape[0])], dtype=torch.long)
        return (
            payload["time_deltas"],
            payload["marks"],
            payload["volumes"],
            payload["lob_states"],
            lf,
            payload["event_mapping"],
            fl,
        )

    start = time.time()
    time_deltas_np, marks_np, volumes_np, lob_states_np, lob_features_np, event_mapping, file_lengths_np = _parse_bfnx_jsonl(files)
    payload = {
        "time_deltas": torch.from_numpy(time_deltas_np),
        "marks": torch.from_numpy(marks_np),
        "volumes": torch.from_numpy(volumes_np),
        "lob_states": torch.from_numpy(lob_states_np),
        "lob_features": torch.from_numpy(lob_features_np),
        "event_mapping": event_mapping,
        "source_files": [os.path.basename(f) for f in files],
        "file_lengths": torch.from_numpy(file_lengths_np),
    }
    tmp_file = f"{cache_file}.tmp"
    torch.save(payload, tmp_file)
    os.replace(tmp_file, cache_file)
    print(f"Built tensor cache in {time.time() - start:.2f}s: {cache_file}")
    return (payload["time_deltas"], payload["marks"], payload["volumes"], payload["lob_states"],
            payload["lob_features"], event_mapping, payload["file_lengths"])


def load_bfnx_events(
    data_dir: str,
    max_files: Optional[int] = None,
) -> Tuple[List[float], List[List[Tuple[int, float]]], EventMapping, List[int]]:
    """Backward-compatible loader returning Python lists.

    New training should use load_bfnx_tensors/create_bfnx_dataloaders, which avoid
    repeated per-sample Python tensor construction.
    """
    files = _selected_jsonl_files(data_dir, max_files)
    time_deltas, marks, volumes, lob_states, _lob_features, event_mapping, _file_lengths = load_bfnx_tensors(
        data_dir=data_dir, max_files=max_files
    )
    marks_with_volumes: List[List[Tuple[int, float]]] = []
    active_rows = marks.nonzero(as_tuple=False)
    row_to_items: Dict[int, List[Tuple[int, float]]] = {}
    for row, col in active_rows.tolist():
        # expm1 reverses cached log1p volume for callers expecting raw volume.
        row_to_items.setdefault(row, []).append((col, float(torch.expm1(volumes[row, col]))))
    for row in range(marks.shape[0]):
        marks_with_volumes.append(row_to_items.get(row, []))
    return time_deltas.tolist(), marks_with_volumes, event_mapping, lob_states.tolist()


class TensorBFNXEventDataset(Dataset):
    """Sliding-window BFNX dataset backed by contiguous tensors.

    __getitem__ is only tensor slicing, avoiding the previous Python loop that
    allocated zero vectors and re-log-transformed volumes for every sample.

    Splitting: when ``file_lengths`` is given, EACH file segment is split
    chronologically 70/15/15 with a gap of (sequence_length + 1) events skipped
    between partitions, so no window straddles a partition (or file) boundary
    and no event is shared between splits.  Tensors stay global; only the
    window start indices are restricted to the requested split's zones.  When
    ``file_lengths`` is None, the whole stream is treated as one segment.
    """

    def __init__(
        self,
        time_deltas: torch.Tensor,
        marks: torch.Tensor,
        volumes: torch.Tensor,
        lob_states: torch.Tensor,
        lob_features: torch.Tensor,
        event_mapping: EventMapping,
        sequence_length: int = 50,
        stride: int = 10,
        split: str = "train",
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        file_lengths=None,
    ):
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        self.num_channels = event_mapping.num_events
        self.event_mapping = event_mapping

        n_total = int(time_deltas.shape[0])

        # Keep tensors GLOBAL (no physical slicing); window start indices below
        # are global indices restricted to the requested split's zones.
        self.time_deltas = time_deltas.contiguous().float()
        self.marks = marks.contiguous().float()
        self.volumes = volumes.contiguous().float()
        self.lob_states = lob_states.contiguous().long()
        feats = lob_features.contiguous().float()
        # Book BEFORE event i conditions event i (constructor: row t's book
        # is the context for the event set at row t+1).  NOTE: the one-step
        # shift is GLOBAL, so the first row of each file segment receives the
        # previous file's last post-event features (row 0 of the stream is
        # clamped to itself).  This affects only one event per file and is an
        # accepted approximation of "clamp at segment starts".
        self.lob_features = torch.cat([feats[:1], feats[:-1]], dim=0)
        self.lob_features_post = feats

        if file_lengths is None:
            segments = [n_total]
        else:
            if torch.is_tensor(file_lengths):
                segments = [int(x) for x in file_lengths.tolist()]
            else:
                segments = [int(x) for x in np.asarray(file_lengths).reshape(-1).tolist()]
            if sum(segments) != n_total:
                raise ValueError(
                    f"file_lengths sum {sum(segments)} != number of events {n_total}"
                )

        gap = self.sequence_length + 1  # events skipped between partitions
        starts_chunks: List[torch.Tensor] = []
        seg_start = 0
        for seg_len in segments:
            seg_end = seg_start + seg_len
            train_end = seg_start + int(seg_len * split_ratio[0])
            val_end = seg_start + int(seg_len * (split_ratio[0] + split_ratio[1]))
            if split == "train":
                zone_lo, zone_hi = seg_start, train_end
            elif split == "val":
                zone_lo, zone_hi = min(train_end + gap, seg_end), val_end
            else:
                zone_lo, zone_hi = min(val_end + gap, seg_end), seg_end
            # Window [s, s + seq_length] (target at s + seq_length) must lie
            # fully inside [zone_lo, zone_hi).
            last_valid_start = zone_hi - self.sequence_length - 1
            if last_valid_start >= zone_lo:
                starts_chunks.append(
                    torch.arange(zone_lo, last_valid_start + 1, self.stride, dtype=torch.long)
                )
            seg_start = seg_end

        if starts_chunks:
            self.starts = torch.cat(starts_chunks)
        else:
            self.starts = torch.empty(0, dtype=torch.long)
        print(f"Created {len(self.starts)} {split} sequences")

    def __len__(self):
        return int(self.starts.numel())

    def __getitem__(self, idx):
        start = int(self.starts[idx])
        end = start + self.sequence_length
        target = end
        return {
            "input_times": self.time_deltas[start:end],
            "input_marks": self.marks[start:end],
            "input_volumes": self.volumes[start:end],
            "input_state": self.lob_states[start:end],
            "input_lob_features": self.lob_features[start:end],
            "target_lob_features": self.lob_features_post[end - 1],
            "target_time": self.time_deltas[target],
            "target_marks": self.marks[target],
            "target_volumes": self.volumes[target],
        }


# Preserve the original class name for external imports.
BFNXEventDataset = TensorBFNXEventDataset


def create_bfnx_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    sequence_length: int = 50,
    stride: int = 10,
    max_files: Optional[int] = None,
    num_workers: int = 0,
    cache_dir: Optional[str] = None,
    rebuild_cache: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, EventMapping]:
    """
    Create train, validation, and test dataloaders for BFNX data.
    """
    time_deltas, marks, volumes, lob_states, lob_features, event_mapping, file_lengths = load_bfnx_tensors(
        data_dir=data_dir,
        max_files=max_files,
        cache_dir=cache_dir,
        rebuild_cache=rebuild_cache,
    )

    train_dataset = TensorBFNXEventDataset(
        time_deltas, marks, volumes, lob_states, lob_features, event_mapping,
        sequence_length, stride, split="train", file_lengths=file_lengths
    )
    val_dataset = TensorBFNXEventDataset(
        time_deltas, marks, volumes, lob_states, lob_features, event_mapping,
        sequence_length, stride, split="val", file_lengths=file_lengths
    )
    test_dataset = TensorBFNXEventDataset(
        time_deltas, marks, volumes, lob_states, lob_features, event_mapping,
        sequence_length, stride, split="test", file_lengths=file_lengths
    )

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 4,
        })

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        **loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        **loader_kwargs
    )

    return train_loader, val_loader, test_loader, event_mapping


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "data/events/bfnx"

    print(f"Testing BFNX data loader with directory: {data_dir}")
    train_loader, val_loader, test_loader, event_mapping = create_bfnx_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        sequence_length=50,
        stride=10,
        max_files=2,
        num_workers=0,
    )

    print("\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)} sequences")
    print(f"  Val: {len(val_loader.dataset)} sequences")
    print(f"  Test: {len(test_loader.dataset)} sequences")
    print(f"  Number of event types: {event_mapping.num_events}")

    batch = next(iter(train_loader))
    print("\nBatch shapes:")
    print(f"  input_times: {batch['input_times'].shape}")
    print(f"  input_marks: {batch['input_marks'].shape}")
    print(f"  input_volumes: {batch['input_volumes'].shape}")
    print(f"  input_state: {batch['input_state'].shape}")
    print(f"  target_time: {batch['target_time'].shape}")
    print(f"  target_marks: {batch['target_marks'].shape}")
    print(f"  target_volumes: {batch['target_volumes'].shape}")
