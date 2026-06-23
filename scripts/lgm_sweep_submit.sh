#!/usr/bin/env bash
# Submit the LGM hyperparameter sweep on the HPC.  Run on a login node:
#     bash scripts/lgm_sweep_submit.sh
# 1) builds the shared tensor cache once (CPU job), then
# 2) submits the 12-config GPU array, held until the cache is ready.
set -euo pipefail
REPO="${REPO:-$HOME/simulation}"
cd "$REPO"
echo "Submitting LGM sweep from $REPO"
CACHE_JID=$(qsub -terse "$REPO/scripts/lgm_cache.sh" | cut -d. -f1)
echo "  cache job:  $CACHE_JID  (lgm_cache.sh)"
ARRAY_JID=$(qsub -terse -hold_jid "$CACHE_JID" "$REPO/scripts/lgm_sweep.sh" | head -1)
echo "  sweep array: $ARRAY_JID  (lgm_sweep.sh, held on cache)"
echo
echo "Watch:   qstat -u \"\$USER\""
echo "Collect: python3 scripts/lgm_sweep_collect.py   (after the array finishes)"
echo "Results: $REPO/experiments/lgm_sweep/<TAG>/"
