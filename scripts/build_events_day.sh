#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N build_events
#$ -l h_rt=36:00:00
#$ -l tmem=24G
#$ -t 1-7
set -o pipefail

# Build trade-correct, lob_state event JSONL from raw order-book + trades, one
# SGE array task per January day. Parametrized over venue/symbol so a single
# script serves every (venue, symbol) -- the SINGLE event-construction pipeline.
#
# Constructor: volume_set_mtpp/process/event_construction_chunked.py, imported
# from the canonical `simulation` repo (flat layout). This is the only event
# construction code path; the old `volume-set-mtpp` forks are deprecated.
#
# Override per run with qsub -v, e.g.:
#   qsub -v VENUE=cbse,SYMBOL=btcusdt,OUTSET=cbse_btc_v3 -t 1-31 scripts/build_events_day.sh
#   qsub -v VENUE=gmni,SYMBOL=ethusdt,OUTSET=gmni_eth_v3 -t 1-31 scripts/build_events_day.sh
#   qsub -v VENUE=gmni,SYMBOL=btcusdt,OUTSET=gmni_btc_v3 -t 1-31 scripts/build_events_day.sh
#
# h_rt is 36h (was 12h in the old per-venue scripts -- that walltime, not any
# code cap, truncated cbse_btc_7_v2 to ~14.5h/day; ast.literal_eval per book row
# is the bottleneck). A full 24h day needs ~20h compute on a dense book.

VENUE="${VENUE:-cbse}"
SYMBOL="${SYMBOL:-btcusdt}"
OUTSET="${OUTSET:-${VENUE}_${SYMBOL}_v3}"
CHUNKSIZE="${CHUNKSIZE:-20000}"
KLEVELS="${KLEVELS:-10}"
MAXROWS="${MAXROWS:-}"   # empty = full day; set (e.g. MAXROWS=120000) for a quick smoke test

# Canonical code tree (flat layout: PYTHONPATH is the repo root, NOT repo/src).
REPO="${REPO:-$HOME/simulation}"
cd "$REPO"
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
# Reuse the existing working venv (pandas/numpy) from the old deploy; the canonical
# CODE comes from $REPO via PYTHONPATH, the DEPS come from the venv.
source "$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

DAY=$(printf "2026-01-%02d" "$SGE_TASK_ID")
BASE=/SAN/medic/TFOW/data
RAW=$BASE/full_order_book/$VENUE/spot/$SYMBOL/2026-01/full_order_book_${VENUE}_spot_${SYMBOL}_${DAY}.csv.gz
TRD=$BASE/trades/$VENUE/spot/$SYMBOL/2026-01/trades_${VENUE}_spot_${SYMBOL}_${DAY}.csv.gz
OUTDIR=$BASE/events/$OUTSET
OUT=$OUTDIR/events_${VENUE}_${SYMBOL}_${DAY}.jsonl.gz
mkdir -p "$OUTDIR"

[ -r "$RAW" ] || { echo "SAN_NOT_VISIBLE_OR_MISSING host=$(hostname) $RAW" >&2; exit 1; }
[ -r "$TRD" ] || { echo "TRADES_MISSING $TRD" >&2; exit 1; }

if [ -s "$OUT" ] && gzip -t "$OUT" 2>/dev/null; then
  echo "SKIP_EXISTS_VALID $OUT"; exit 0
fi

echo "BUILD_START $(date) venue=$VENUE sym=$SYMBOL day=$DAY host=$(hostname) repo=$REPO"
python3 - <<PY
from volume_set_mtpp.process.event_construction_chunked import process_data_files_chunked
_mr = "$MAXROWS"
process_data_files_chunked(
    orderbook_file="$RAW",
    trades_file="$TRD",
    k_levels=$KLEVELS,
    chunksize=$CHUNKSIZE,
    max_rows=(int(_mr) if _mr else None),
    output_file="$OUT",
    jsonl_format=True,
)
PY
rc=$?
[ $rc -ne 0 ] && { echo "BUILD_FAILED rc=$rc day=$DAY"; rm -f "$OUT"; exit 1; }
gzip -t "$OUT" || { echo "GZIP_INVALID $OUT"; rm -f "$OUT"; exit 1; }
echo "BUILD_OK $(date) day=$DAY size=$(du -h "$OUT" | cut -f1)"
