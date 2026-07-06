#!/usr/bin/env bash
# Local watcher: wait for an LGM sweep array to finish on HPC, collect the ranked
# results, and EMAIL them via notify_email.py (uses SMTP_* from .env).
# Usage: bash scripts/sweep_email_watch.sh <JOBID> <REMOTE_ROOT_REL> <LABEL>
set +e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT_DIR"
source scripts/hpc-common.sh
JID="${1:?jobid}"; ROOTREL="${2:?remote sweep root (relative to ~/simulation)}"; LABEL="${3:-lgm-sweep}"
TAGS="L64_r92_M4_vfb L64_r86_M4_vfb L64_r97_M4 L64_r92_M8 L128_r92_M8 L128_r97_M4_vfb"

OUT=$(remote_ssh "
  cd \$HOME/simulation
  while qstat -u \$USER 2>/dev/null | grep -q $JID; do
    now=\$(date +%s); hung=0
    for t in $TAGS; do
      b=$ROOTREL/\$t
      st=\$(grep -oE 'TRAIN_START|DONE|NO_CKPT|SAN_NOT_VISIBLE' \$b/master.log 2>/dev/null | tail -1)
      if [ -f \$b/train.log ]; then m=\$(stat -c %Y \$b/train.log); age=\$(((now-m)/60));
        [ \"\$st\" = TRAIN_START ] && [ \$age -gt 30 ] && hung=1; fi
    done
    [ \$hung -eq 1 ] && { echo SWEEP_STALLED; break; }
    sleep 150
  done
  echo '---COLLECT---'
  source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null
  source \$HOME/volume-set-mtpp/venv/bin/activate 2>/dev/null
  export PYTHONPATH=\$PWD
  python3 scripts/lgm_sweep_collect.py $ROOTREL 2>/dev/null
")

SUBJECT="[$LABEL] sweep DONE"
echo "$OUT" | grep -q SWEEP_STALLED && SUBJECT="[$LABEL] sweep STALLED (needs attention)"
BESTLINE=$(echo "$OUT" | grep -E "BEST" | tr '\n' ' ')
[ -n "$BESTLINE" ] && SUBJECT="$SUBJECT — $BESTLINE"

set -a; source .env 2>/dev/null; set +a
printf '%s\n' "$OUT" | python3 scripts/notify_email.py --subject "$SUBJECT" 2>&1 | tail -3
echo "=== EMAILED: $SUBJECT ==="
echo "$OUT"
