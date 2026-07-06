#!/usr/bin/env bash
# Submit a worker script to the BEST AVAILABLE GPU, never waiting for a better one.
#
#   bash scripts/submit_best_gpu.sh <worker.sh> [extra qsub args, e.g. -t 7]
#
# Policy (user-set): probe accessible tiers best->worst for ACTUAL free slots
# and submit to the first tier with capacity NOW; if none has capacity, submit
# to the fastest-draining accessible pool (a40) rather than queueing on a
# better-but-full tier. Facts baked in (qselect -U honglifu, 2026-07-03):
#   h100 -> gpu.q@seymour4 only; a40 -> gpu.q@animal-206-[12] only.
#   chip/behemoth/bubba/seymour1-3: ACL-blocked (never grant).
#   hoots/gonzo: broken mounts (silently hang) -- never submit bare gpu=true.
#   Do NOT pass env via `qsub -v` (jobs hang at python startup).
set -euo pipefail
WORKER="$1"; shift || true
RES_BASE="h_rt=28800,tmem=32G,gpu=true"
TIERS=(h100 a40)   # best -> worst among tiers our account can actually use

pick=""
for t in "${TIERS[@]}"; do
  # accessible queue instances for this tier
  QS=$(qselect -l "gpu_type=$t" -U "$USER" 2>/dev/null | grep "^gpu.q@" || true)
  [ -z "$QS" ] && continue
  free=0
  while read -r q; do
    [ -z "$q" ] && continue
    line=$(qstat -f 2>/dev/null | grep -F "$q" | head -1)
    # slots field "res/used/total"; disabled states contain d/E/u
    used=$(echo "$line" | awk '{print $3}' | cut -d/ -f2)
    total=$(echo "$line" | awk '{print $3}' | cut -d/ -f3)
    state=$(echo "$line" | awk '{print $6}')
    if [ -n "$total" ] && [ "${used:-0}" -lt "$total" ] && ! echo "${state:-}" | grep -qE "[dEu]"; then
      free=$((free + total - used))
    fi
  done <<< "$QS"
  echo "tier $t: $free free accessible slot(s)"
  if [ "$free" -gt 0 ]; then pick="$t"; break; fi
done

if [ -z "$pick" ]; then
  pick="a40"
  echo "no tier has free capacity now -> submitting to $pick (fastest-draining), it takes the next slot"
fi

JID=$(qsub -terse "$@" -l "$RES_BASE,gpu_type=$pick" "$WORKER" | head -1 | cut -d. -f1)
echo "submitted $JID on gpu_type=$pick"
