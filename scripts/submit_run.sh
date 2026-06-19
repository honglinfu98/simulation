#!/usr/bin/env bash
# Render a run script from scripts/_template_run.sh, sync it to the cluster,
# qsub it via the shared ControlMaster, and register the run for the watcher.
#
#   bash scripts/hpc-common.sh open            # once: seed the SSH master
#   bash scripts/submit_run.sh --tag lgm086 --decoder lgm \
#        --extra "--decoder-type lgm --nmh-timescales 4 --ptp-dim 8 \
#                 --lgm-target-rate 2.381 --nmh-project-rho 0.86 --mark-head categorical"
#
# Notes: gpu_type=h100/a100_80 are gated for this account, so the renderer forces
# plain `-l gpu=true`. The run script itself cd's into $HPC_RUN_HOME on the cluster.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./hpc-common.sh
source "$SCRIPT_DIR/hpc-common.sh"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMPLATE="$SCRIPT_DIR/_template_run.sh"

TAG=""; DECODER=""; EXTRA=""; EPOCHS="40"; SEED="1"; HRT="8:00:00"; TMEM="24G"; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2;;
    --decoder) DECODER="$2"; shift 2;;
    --extra) EXTRA="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --h-rt) HRT="$2"; shift 2;;
    --tmem) TMEM="$2"; shift 2;;
    --template) TEMPLATE="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[[ -z "$TAG" || -z "$DECODER" ]] && { echo "usage: submit_run.sh --tag T --decoder D [--extra ...] [--epochs N] [--dry-run]" >&2; exit 2; }

# --- portable lock (macOS has no flock) ---
LOCK="$ROOT_DIR/.runs/registry.lock.d"
_lock(){ mkdir -p "$ROOT_DIR/.runs"; until mkdir "$LOCK" 2>/dev/null; do sleep 0.3; done; }
_unlock(){ rmdir "$LOCK" 2>/dev/null || true; }
REG="$ROOT_DIR/.runs/registry.tsv"

# refuse duplicate active tag
if [[ -f "$REG" ]] && awk -F'\t' -v t="$TAG" '$2==t && $8=="active"{f=1} END{exit !f}' "$REG"; then
  echo "submit_run: tag '$TAG' already has an active run in $REG" >&2; exit 1
fi

NOW=$(date +%s)
GEN_NAME="gmni_marks_${TAG}_$(date +%Y%m%d_%H%M%S).sh"
GEN_LOCAL="$ROOT_DIR/jobs/gen/$GEN_NAME"; mkdir -p "$ROOT_DIR/jobs/gen"

# Render: substitute tag/decoder/extra/epochs/seed, force plain gpu=true, drop gpu_type.
sed -E \
  -e "s/^#\\\$ -N .*/#\$ -N gmni_${TAG}/" \
  -e "/^#\\\$ -l gpu_type=/d" \
  -e "s/^#\\\$ -l gpu=.*/#\$ -l gpu=true/" \
  -e "s/^#\\\$ -l h_rt=.*/#\$ -l h_rt=${HRT}/" \
  -e "s/^#\\\$ -l tmem=.*/#\$ -l tmem=${TMEM}/" \
  -e "s/^TAG=.*/TAG=${TAG}/" \
  -e "s/^DECODER=.*/DECODER=${DECODER}/" \
  -e "s/^SEED=.*/SEED=${SEED}/" \
  -e "s/^EXTRA=.*/EXTRA=\"${EXTRA}\"/" \
  -e "s/--epochs [0-9]+/--epochs ${EPOCHS}/" \
  "$TEMPLATE" > "$GEN_LOCAL"
echo "rendered -> $GEN_LOCAL"

EXPECT_GLOB="$HPC_RUN_HOME/experiments/gmni_marks_${TAG}_*"
if [[ "$DRY" -eq 1 ]]; then
  echo "--- DRY RUN: would sync + qsub the above, and register:"
  printf 'run_id=%s\ttag=%s\tdecoder=%s\tsubmit=%s\texpect=%s\n' "${TAG}_${NOW}" "$TAG" "$DECODER" "$NOW" "$EXPECT_GLOB"
  exit 0
fi

# Sync the single job script (cheap; one round-trip over the master socket).
remote_ssh "mkdir -p '$HPC_REMOTE_PROJECT/jobs/gen'"
rsync -az -e "$SSH_CMD" "$GEN_LOCAL" "$SSH_TARGET:$HPC_REMOTE_PROJECT/jobs/gen/$GEN_NAME"

OUT=$(remote_ssh "cd '$HPC_REMOTE_PROJECT' && qsub 'jobs/gen/$GEN_NAME'")
echo "$OUT"
JOBID=$(printf '%s' "$OUT" | grep -oE 'Your job [0-9]+' | grep -oE '[0-9]+' | head -1)
[[ -z "$JOBID" ]] && { echo "submit_run: could not parse job id from qsub output" >&2; exit 1; }

_lock
[[ -f "$REG" ]] || printf 'run_id\ttag\tjobid\tdecoder\tsubmit_epoch\texpected_base_glob\tresolved_base\tstate\trc\tnotified\tmetrics_json\n' > "$REG"
printf '%s\t%s\t%s\t%s\t%s\t%s\t\tactive\t\t0\t\n' "${TAG}_${NOW}" "$TAG" "$JOBID" "$DECODER" "$NOW" "$EXPECT_GLOB" >> "$REG"
_unlock
echo "registered run ${TAG}_${NOW} (job $JOBID). Start/keep the watcher running: bash scripts/watch_runs.sh"
