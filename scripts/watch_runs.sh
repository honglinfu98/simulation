#!/usr/bin/env bash
# Poll registered HPC runs and email on completion/failure/crash.
#
#   bash scripts/hpc-common.sh open     # once: seed the SSH master
#   set -a; source .env; set +a         # so SMTP_* reach notify_email.py
#   bash scripts/watch_runs.sh          # foreground loop (or via launchd, see docs)
#
# CARDINAL RULE: exactly ONE ssh per cycle (a single remote probe covers qstat +
# every active run's master.log + .last_<tag>_base). Never fan out parallel ssh.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./hpc-common.sh
source "$SCRIPT_DIR/hpc-common.sh"
set +e   # hpc-common.sh turns on `set -e`; the watcher must NOT die on a non-zero
         # command (a failed ssh/grep is handled per-cycle, never fatal to the loop).
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REG="$ROOT_DIR/.runs/registry.tsv"
OUTROOT="$ROOT_DIR/outputs/runs"; mkdir -p "$OUTROOT"

INTERVAL="${WATCH_INTERVAL:-90}"
GRACE="${WATCH_CRASH_GRACE:-180}"
UNREACH_AFTER="${WATCH_UNREACHABLE_ALERT_AFTER:-10}"
ONCE=0; EXIT_EMPTY=0
for a in "$@"; do case "$a" in --once) ONCE=1;; --exit-when-empty) EXIT_EMPTY=1;; esac; done

LOCK="$ROOT_DIR/.runs/registry.lock.d"
_lock(){ mkdir -p "$ROOT_DIR/.runs"; until mkdir "$LOCK" 2>/dev/null; do sleep 0.3; done; }
_unlock(){ rmdir "$LOCK" 2>/dev/null || true; }

email(){ # $1 subject ; body on stdin
  # Point Python at certifi's CA bundle so SMTP TLS verifies even under launchd
  # (where the default python3 may lack root certs). Harmless if already set.
  ( set -a; source "$ROOT_DIR/.env"; set +a
    export SSL_CERT_FILE="${SSL_CERT_FILE:-$(python3 -m certifi 2>/dev/null)}"
    python3 "$SCRIPT_DIR/notify_email.py" --subject "$1" ); }

set_field(){ # run_id field_index(1-based) value  -> rewrite registry row
  local rid="$1" idx="$2" val="$3"
  _lock
  awk -F'\t' -v OFS='\t' -v rid="$rid" -v i="$idx" -v v="$val" \
    '$1==rid{$i=v} {print}' "$REG" > "$REG.tmp" && mv "$REG.tmp" "$REG"
  _unlock
}

probe(){ # one ssh: qstat + per-run base/master tail. echoes blob; returns ssh rc
  local cmd="echo '@@QSTAT'; qstat 2>/dev/null || true; echo '@@QSTAT_END';"
  while IFS=$'\t' read -r rid tag jobid _dec _sub _glob _rbase state _rc _notif _m; do
    [[ "$state" == "active" ]] || continue
    cmd+="echo '@@RUN $rid $tag $jobid';"
    cmd+="b=\$(cat '$HPC_RUN_HOME/.last_${tag}_base' 2>/dev/null);"
    # .last_<tag>_base holds a path RELATIVE to HPC_RUN_HOME; make it absolute.
    cmd+="case \"\$b\" in /*) ;; ?*) b='$HPC_RUN_HOME/'\$b;; esac;"
    # if still not a real dir, fall back to the newest matching experiment dir (absolute).
    cmd+="[ -d \"\$b\" ] || b=\$(ls -1dt $HPC_RUN_HOME/experiments/gmni_marks_${tag}_*/ 2>/dev/null | head -1);"
    cmd+="echo \"@@BASE \$b\"; echo '@@MASTER'; tail -n 60 \"\$b/master.log\" 2>/dev/null; echo '@@END';"
  done < <(tail -n +2 "$REG" 2>/dev/null)
  remote_ssh "$cmd"
}

section(){ # extract lines between '@@RUN <rid> ' and next '@@RUN'/'@@QSTAT' from blob file
  awk -v rid="$1" '
    $1=="@@RUN" && $2==rid {grab=1; next}
    $1=="@@RUN" && $2!=rid {grab=0}
    grab {print}' "$2"; }

cycle(){
  [[ -f "$REG" ]] || return 0
  local n; n=$(awk -F'\t' 'NR>1 && $8=="active"{c++} END{print c+0}' "$REG")
  if [[ "$n" -eq 0 ]]; then [[ "$EXIT_EMPTY" -eq 1 ]] && return 9; return 0; fi

  local blob; blob=$(mktemp)
  if ! probe > "$blob" 2>/dev/null; then rm -f "$blob"; return 1; fi   # ssh failed this cycle
  local qjobs; qjobs=$(awk '/@@QSTAT$/{q=1;next} /@@QSTAT_END/{q=0} q{print}' "$blob" | grep -oE '^[[:space:]]*[0-9]+' | tr -d ' ')

  while IFS=$'\t' read -r rid tag jobid dec sub glob rbase state rc notif metrics; do
    [[ "$state" == "active" ]] || continue
    local sec base mlog outcome=""
    sec=$(section "$rid" "$blob")
    base=$(printf '%s\n' "$sec" | awk '$1=="@@BASE"{print $2; exit}')
    mlog=$(printf '%s\n' "$sec" | awk '/@@MASTER/{m=1;next} /@@END/{m=0} m{print}')

    if printf '%s\n' "$mlog" | grep -qE 'DONE .* STATUS=0 '; then outcome=done
    elif printf '%s\n' "$mlog" | grep -qE 'DONE .* STATUS=1 |SAN_NOT_VISIBLE|TRAIN_END .* RC=[1-9]'; then outcome=failed
    elif ! printf '%s\n' "$qjobs" | grep -qx "$jobid"; then
      local age=$(( $(date +%s) - sub ))
      [[ "$age" -gt "$GRACE" ]] && outcome=crashed
    fi
    [[ -z "$outcome" ]] && continue

    # pull artifacts (best-effort), parse metrics, email once
    local od="$OUTROOT/$rid"; mkdir -p "$od"
    if [[ -n "$base" ]]; then
      rsync -az -e "$SSH_CMD" \
        "$SSH_TARGET:$base/master.log" "$SSH_TARGET:$base/genuine_${tag}.json" "$od/" 2>/dev/null || true
      rsync -az -e "$SSH_CMD" "$SSH_TARGET:$base/stylized_facts/" "$od/stylized_facts/" 2>/dev/null || true
    fi
    local summary; summary=$(python3 "$SCRIPT_DIR/_parse_metrics.py" "$od" "$tag" "$outcome" 2>/dev/null || echo "outcome=$outcome")
    set_field "$rid" 7 "${base:--}"; set_field "$rid" 8 "$outcome"
    if [[ "$notif" != "1" ]]; then
      local OUTCOME; OUTCOME=$(printf '%s' "$outcome" | tr '[:lower:]' '[:upper:]')  # bash 3.2 has no ${x^^}
      local subj="[sim] $tag $OUTCOME ($summary)"
      if printf '%s\nrun_id=%s job=%s base=%s\n\n--- master.log tail ---\n%s\n' \
            "$summary" "$rid" "$jobid" "$base" "$mlog" | email "$subj"; then
        set_field "$rid" 10 1
        echo "$(date '+%H:%M:%S') emailed $tag ($outcome)"
      else
        echo "$(date '+%H:%M:%S') email failed for $tag; will retry"
      fi
    fi
  done < <(tail -n +2 "$REG")
  rm -f "$blob"
  return 0
}

echo "watch_runs: interval=${INTERVAL}s grace=${GRACE}s. (Ctrl-C to stop.)"
fails=0; unreach_alerted=0
while :; do
  cycle; rc=$?
  if [[ "$rc" -eq 9 ]]; then echo "no active runs; exiting (--exit-when-empty)"; break; fi
  if [[ "$rc" -eq 1 ]]; then
    fails=$((fails+1)); echo "$(date '+%H:%M:%S') cluster unreachable (cycle fail $fails)"
    if [[ "$fails" -ge "$UNREACH_AFTER" && "$unreach_alerted" -eq 0 ]]; then
      echo "watcher cannot reach the cluster after $fails cycles. Re-run: bash scripts/hpc-common.sh open" \
        | email "[sim] watcher: cluster UNREACHABLE" && unreach_alerted=1
    fi
    sleep $(( INTERVAL < 30 ? 30 : INTERVAL )); continue
  fi
  fails=0; unreach_alerted=0
  [[ "$ONCE" -eq 1 ]] && break
  sleep "$INTERVAL"
done
