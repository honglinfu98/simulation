#!/usr/bin/env bash
# Land the BEST AVAILABLE GPU *NOW* via qrsh (immediate scheduling), best->worst.
#
#   nohup bash qrsh_best_gpu.sh <worker.sh> <task_id> <results_dir> > runner_<tag>.log 2>&1 &
#
# Policy (user-set, 2026-07-06): try qrsh -now yes per tier in performance
# order across ALL ACL-accessible tiers; the first tier with capacity runs the
# experiment IN the qrsh allocation. No tier grants -> re-probe every RETRY_S
# seconds (never parked in a queue). Each allocation first verifies the /SAN
# mount (20s timeout); failing tiers are released + blacklisted for the run.
#
# Robustness (learned 2026-07-06): qrsh mangles multi-word command strings
# (returns rc=0 in seconds without running anything), so the work is wrapped
# in a generated PAYLOAD SCRIPT invoked by absolute path; and rc=0 alone is
# NOT trusted -- success requires the worker's DONE marker in <results_dir>.
# Accessible tiers (qselect -U honglifu): h100=seymour4, l40s=hoots-207-1/2,
# a40=animal-206-1/2, rtx6000=fozzie, v100=webb. Run under nohup on the
# LOGIN node so training survives SSH disconnects.
set -u
WORKER="$(readlink -f "$1")"
TASK="${2:-7}"
RESULTS="${3:?usage: qrsh_best_gpu.sh <worker.sh> <task_id> <results_dir>}"
TIERS=(h100 l40s a40 rtx6000 v100)
RES="h_rt=28800,tmem=32G,gpu=true"
RETRY_S=60
SAN_CHECK="/SAN/medic/TFOW/data/events"

[ -f "$WORKER" ] || { echo "no such worker: $WORKER"; exit 2; }

PAYLOAD="$(dirname "$WORKER")/.qrsh_payload_$$.sh"
cat > "$PAYLOAD" <<EOF
#!/usr/bin/env bash
echo "PAYLOAD_NODE=\$(hostname)"
if ! timeout 20 ls "$SAN_CHECK" >/dev/null 2>&1; then
  echo "NO_SAN_MOUNT on \$(hostname)"
  exit 99
fi
# Under qrsh the SGE prolog does NOT bind CUDA_VISIBLE_DEVICES (unlike qsub),
# so on shared nodes we must pick an idle GPU ourselves or we collide with
# exclusive-mode jobs ("CUDA-capable device(s) is/are busy or unavailable").
if [ -z "\${CUDA_VISIBLE_DEVICES:-}" ]; then
  FREE=\$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
          | awk -F', *' '\$2 < 300 && \$3 < 5 {print \$1; exit}')
  if [ -z "\$FREE" ]; then
    echo "NO_IDLE_GPU on \$(hostname)"
    exit 98
  fi
  export CUDA_VISIBLE_DEVICES="\$FREE"
  echo "PAYLOAD_GPU=\$CUDA_VISIBLE_DEVICES"
fi
export SGE_TASK_ID=$TASK
exec bash "$WORKER"
EOF
chmod +x "$PAYLOAD"
trap 'rm -f "$PAYLOAD"' EXIT
echo "payload: $PAYLOAD -> worker $WORKER (task $TASK), results expected in $RESULTS"

declare -A BAD=()
attempt=0
while true; do
  attempt=$((attempt + 1))
  for t in "${TIERS[@]}"; do
    [ "${BAD[$t]:-}" = "1" ] && continue
    echo "[$(date '+%F %T')] attempt $attempt tier $t: requesting immediate allocation"
    ERRF=$(mktemp)
    qrsh -now yes -l "$RES,gpu_type=$t" "$PAYLOAD" 2>"$ERRF"
    rc=$?
    if grep -q "could not be scheduled" "$ERRF"; then
      echo "  tier $t: no capacity right now"
      rm -f "$ERRF"
      continue
    fi
    cat "$ERRF" >&2
    rm -f "$ERRF"
    if [ $rc -eq 99 ]; then
      echo "[$(date '+%F %T')] tier $t: granted but /SAN not visible -> released + blacklisted"
      BAD[$t]=1
      continue
    fi
    if [ $rc -eq 98 ]; then
      echo "[$(date '+%F %T')] tier $t: granted but no idle GPU on node (exclusive-mode collision) -> released, will retry"
      continue
    fi
    # Trust artifacts, not exit codes: qrsh has returned rc=0 without running.
    if grep -q "^DONE " "$RESULTS/master.log" 2>/dev/null; then
      echo "[$(date '+%F %T')] worker COMPLETED on tier $t (DONE marker present), rc=$rc"
      exit 0
    fi
    if [ -f "$RESULTS/master.log" ]; then
      echo "[$(date '+%F %T')] tier $t: worker STARTED but did not finish (rc=$rc) -- inspect $RESULTS; not retrying automatically"
      exit 1
    fi
    echo "[$(date '+%F %T')] tier $t: qrsh returned rc=$rc but worker never started (qrsh mangle/ghost grant) -> retrying"
  done
  echo "[$(date '+%F %T')] no usable tier available; re-probing in ${RETRY_S}s"
  sleep "$RETRY_S"
done
