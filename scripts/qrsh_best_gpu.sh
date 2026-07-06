#!/usr/bin/env bash
# Land the BEST AVAILABLE GPU *NOW* via qrsh (immediate scheduling), best->worst.
#
#   nohup bash qrsh_best_gpu.sh <worker.sh> [task_id] > runner_<tag>.log 2>&1 &
#
# Policy (user-set, 2026-07-06): every submission tries qrsh -now yes per tier
# (h100 -> a40, the only ACL-granted pools); the first tier with capacity runs
# the experiment IN the qrsh allocation. If no tier grants, re-probe every
# RETRY_S seconds -- the job is never parked in a queue and always lands on the
# best tier available at grant time. Run this under nohup ON THE LOGIN NODE so
# the qrsh client (and thus the training) survives laptop/SSH disconnects.
# Facts baked in: never bare gpu=true (hoots/gonzo hang); SGE_TASK_ID is set
# explicitly (no qsub array, and qsub -v hangs python anyway).
set -u
WORKER="$(readlink -f "$1")"
TASK="${2:-7}"
TIERS=(h100 a40)
RES="h_rt=28800,tmem=32G,gpu=true"
RETRY_S=60

[ -f "$WORKER" ] || { echo "no such worker: $WORKER"; exit 2; }

attempt=0
while true; do
  attempt=$((attempt + 1))
  for t in "${TIERS[@]}"; do
    echo "[$(date '+%F %T')] attempt $attempt tier $t: requesting immediate allocation"
    ERRF=$(mktemp)
    qrsh -now yes -l "$RES,gpu_type=$t" bash -lc "SGE_TASK_ID=$TASK exec bash '$WORKER'" 2>"$ERRF"
    rc=$?
    if grep -q "could not be scheduled" "$ERRF"; then
      echo "  tier $t: no capacity right now"
      rm -f "$ERRF"
      continue
    fi
    cat "$ERRF" >&2
    rm -f "$ERRF"
    echo "[$(date '+%F %T')] worker exited rc=$rc on tier $t"
    exit $rc
  done
  echo "[$(date '+%F %T')] no tier available; re-probing all tiers in ${RETRY_S}s"
  sleep "$RETRY_S"
done
