#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N rate_cal
#$ -l h_rt=2:00:00
#$ -l tmem=32G
#$ -l gpu=true
#$ -l h=!hoots-207-1*
#$ -pe gpu 1
#$ -t 1-4
set -o pipefail

# POST-HOC RATE CALIBRATION ABLATION (Route 1) -- simulation-only, NO training.
#
# SS2P2's rate head is lambda = s * softplus(z), EXACTLY linear in the scale s,
# and the thinning ceiling s*softplus(c) scales with it -- so a 1-D bisection
# on s calibrates the free-run rate while preserving the certificate.
# --calibrate-rate -1 bisects short probe rollouts (120s x 8 seq) until the
# free-run rate matches the measured real rate, then runs the full rollout.
#
# Design: reuses the TRAINED checkpoints from experiments/tbptt_ablation
# (no retraining; marks head untouched by construction). 2 checkpoints x
# {uncalibrated, calibrated}:
#   1) tbptt-uncal   ss2p2-tbptt checkpoint, k=1        (control)
#   2) tbptt-cal     ss2p2-tbptt checkpoint, bisected k (treatment)
#   3) cold-uncal    ss2p2-cold checkpoint,  k=1        (control)
#   4) cold-cal      ss2p2-cold checkpoint,  bisected k (treatment)
#
# Primary endpoints: F0 sim_rate (should hit real by construction for cal arms;
# the found k itself estimates the closed-loop inflation factor) and whether the
# STRUCTURE facts (Fano/clus/retACF), currently computed on a ~7x-hot stream,
# improve once the clock rate is right.
REPO="${REPO:-$HOME/simulation}"
DATA="${DATA:-/SAN/medic/TFOW/data/events/gmni_eth_7_v2_marks}"
MAXFILES="${MAXFILES:-7}"
CACHE="${CACHE:-$DATA/.tensor_cache_eval}"
SEQ="${SEQ:-1024}"; STRIDE="${STRIDE:-1024}"
CKROOT="${CKROOT:-$REPO/experiments/tbptt_ablation}"
ROOT="${ROOT:-$REPO/experiments/rate_cal}"

case "$SGE_TASK_ID" in
  1) TAG=tbptt-uncal; CKPT="$CKROOT/ss2p2-tbptt/train/best_model.pt"; CAL="" ;;
  2) TAG=tbptt-cal;   CKPT="$CKROOT/ss2p2-tbptt/train/best_model.pt"; CAL="--calibrate-rate -1" ;;
  3) TAG=cold-uncal;  CKPT="$CKROOT/ss2p2-cold/train/best_model.pt";  CAL="" ;;
  4) TAG=cold-cal;    CKPT="$CKROOT/ss2p2-cold/train/best_model.pt";  CAL="--calibrate-rate -1" ;;
  *) echo "no config for task $SGE_TASK_ID"; exit 1 ;;
esac

cd "$REPO"
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4
B="$ROOT/$TAG"
rm -rf "$B"
mkdir -p "$B/stylized_facts"; ML="$B/master.log"
log(){ echo "$@" | tee -a "$ML"; }
fail(){ log "DONE $(date) STATUS=1 stage=$1 rc=$2 BASE=$B"; exit 1; }
log "START $(date) TAG=$TAG host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tee -a "$ML"
[ -s "$CKPT" ] || fail ckpt_missing 1

log "SF $(date) ckpt=$CKPT cal='${CAL:-none}'"
python3 -u -m volume_set_mtpp.evaluation.stylized_facts --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" \
  --checkpoint "$CKPT" --label "$TAG" --output-dir "$B/stylized_facts" --device cuda --sampler inversion \
  --context-mode carried $CAL \
  --seq-length "$SEQ" --stride "$STRIDE" --batch-size 256 --rollout-duration 600 --rollout-sequences 32 \
  --rollout-seed 1 --bucket-seconds 1.0 --max-real-windows 4096 > "$B/sf.log" 2>&1
SF_RC=$?
grep -E "CALIBRAT|CAL probe" "$B/sf.log" | tee -a "$ML"
{ [ "$SF_RC" -eq 0 ] && [ -s "$B/stylized_facts/stylized_facts_${TAG}.json" ]; } \
  || { tail -25 "$B/sf.log" | tee -a "$ML"; fail sf "$SF_RC"; }
log "DONE $(date) STATUS=0 BASE=$B"
