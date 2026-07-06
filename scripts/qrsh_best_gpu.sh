#!/usr/bin/env bash
# Land the BEST AVAILABLE GPU *NOW* via qrsh (immediate scheduling), best->worst.
#
#   nohup bash qrsh_best_gpu.sh <worker.sh> [task_id] > runner_<tag>.log 2>&1 &
#
# Policy (user-set, 2026-07-06): try qrsh -now yes per tier in performance
# order across ALL ACL-accessible tiers; the first tier with capacity runs the
# experiment IN the qrsh allocation. If no tier grants, re-probe every RETRY_S
# seconds -- never parked in a queue, always the best tier available at grant
# time. Each allocation first verifies the /SAN data mount (20s timeout, so a
# dead automount cannot hang us); a tier that fails the mount check is
# released and blacklisted for the rest of this run.
# Accessible tiers (qselect -U honglifu, 2026-07-06):
#   h100=seymour4, l40s=hoots-207-1/2, a40=animal-206-1/2,
#   rtx6000=fozzie, v100=webb.  (a6000/1080ti: ACL-blocked.)
# Run under nohup ON THE LOGIN NODE so training survives SSH disconnects.
set -u
WORKER="$(readlink -f "$1")"
TASK="${2:-7}"
TIERS=(h100 l40s a40 rtx6000 v100)
RES="h_rt=28800,tmem=32G,gpu=true"
RETRY_S=60
SAN_CHECK="/SAN/medic/TFOW/data/events"

[ -f "$WORKER" ] || { echo "no such worker: $WORKER"; exit 2; }

declare -A BAD=()
attempt=0
while true; do
  attempt=$((attempt + 1))
  for t in "${TIERS[@]}"; do
    [ "${BAD[$t]:-}" = "1" ] && continue
    echo "[$(date '+%F %T')] attempt $attempt tier $t: requesting immediate allocation"
    ERRF=$(mktemp)
    qrsh -now yes -l "$RES,gpu_type=$t" bash -lc \
      "echo NODE=\$(hostname); if ! timeout 20 ls '$SAN_CHECK' >/dev/null 2>&1; then echo NO_SAN_MOUNT; exit 99; fi; SGE_TASK_ID=$TASK exec bash '$WORKER'" 2>"$ERRF"
    rc=$?
    if grep -q "could not be scheduled" "$ERRF"; then
      echo "  tier $t: no capacity right now"
      rm -f "$ERRF"
      continue
    fi
    cat "$ERRF" >&2
    rm -f "$ERRF"
    if [ $rc -eq 99 ]; then
      echo "[$(date '+%F %T')] tier $t: allocation granted but /SAN not visible -> released + blacklisted"
      BAD[$t]=1
      continue
    fi
    echo "[$(date '+%F %T')] worker exited rc=$rc on tier $t"
    exit $rc
  done
  echo "[$(date '+%F %T')] no usable tier available; re-probing in ${RETRY_S}s"
  sleep "$RETRY_S"
done
