#!/usr/bin/env bash
# Submit 7-day event-construction jobs for BTC, ETH and SOL on one venue.
#
# Run this on an HPC LOGIN node:
#     bash scripts/build_btc_eth_sol_7d.sh
#
# It does NOT crunch data itself -- it qsubs one SGE array job per coin (one
# array task per day), so the heavy work runs on compute nodes. Each task uses
# the optimized, single-source constructor in $REPO via scripts/build_events_day.sh.
#
# Defaults: Coinbase (cbse), days 1-7. Override with env vars, e.g.:
#     VENUE=gmni bash scripts/build_btc_eth_sol_7d.sh      # Gemini (SOL = solusd)
#     DAYS=1-31  bash scripts/build_btc_eth_sol_7d.sh      # full month
#     SUFFIX=v3  bash scripts/build_btc_eth_sol_7d.sh      # OUTSET = <venue>_<coin>_v3
#
# Outputs: /SAN/medic/TFOW/data/events/<venue>_<coin>_<suffix>/events_<venue>_<symbol>_<day>.jsonl.gz
# Existing valid outputs are skipped, so re-running resumes safely.
set -euo pipefail

REPO="${REPO:-$HOME/simulation}"
VENUE="${VENUE:-cbse}"
DAYS="${DAYS:-1-7}"
SUFFIX="${SUFFIX:-7d}"
WORKER="$REPO/scripts/build_events_day.sh"

[ -f "$WORKER" ] || { echo "ERROR: worker not found: $WORKER (deploy the simulation repo to \$REPO first)"; exit 1; }

echo "Submitting event builds | venue=$VENUE days=$DAYS suffix=$SUFFIX repo=$REPO"
echo "-------------------------------------------------------------------"
for coin in btc eth sol; do
  # Raw symbol name depends on venue (Gemini names SOL differently).
  case "$VENUE:$coin" in
    cbse:btc|binc:btc|gmni:btc) sym=btcusdt ;;
    cbse:eth|binc:eth|gmni:eth) sym=ethusdt ;;
    cbse:sol|binc:sol)          sym=solusdt ;;
    gmni:sol)                   sym=solusd ;;
    *) echo "ERROR: unknown VENUE=$VENUE (expected cbse|gmni|binc)"; exit 1 ;;
  esac
  outset="${VENUE}_${coin}_${SUFFIX}"
  # -t / -v / -N on the command line override the worker's embedded #$ directives.
  jid=$(qsub -terse -N "ev_${VENUE}_${coin}" \
        -v "VENUE=${VENUE},SYMBOL=${sym},OUTSET=${outset},REPO=${REPO}" \
        -t "$DAYS" "$WORKER")
  echo "  ${coin}  ${sym}  ->  outset=${outset}  job=${jid}"
done
echo "-------------------------------------------------------------------"
echo "Watch:   qstat -u \"\$USER\""
echo "Outputs: /SAN/medic/TFOW/data/events/${VENUE}_{btc,eth,sol}_${SUFFIX}/"
