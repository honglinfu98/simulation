#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N final_cmp
#$ -l h_rt=16:00:00
#$ -l tmem=32G
#$ -l gpu=true
#$ -l h=!hoots-207-1*
#$ -pe gpu 1
#$ -t 1-6
set -o pipefail

# FINAL COMPARISON BENCHMARK (paper headline tables), seq-1024 regime.
#
# 6 models: 5 literature baselines with their STANDARD training (cold-start
# windowed MLE) vs OUR full recipe on task 6:
#   SS2P2 heads + TBPTT training (--tbptt) + carried-state rollout
#   (--context-mode carried) + post-hoc rate calibration (--calibrate-rate -1).
#
# Fairness: ALL tasks share seq 1024 / stride 1024 / batch 64 / 40 epochs /
# seed 1 / endpoint compensator -- identical windows and gradient-step count
# (stride=seq because TBPTT requires non-overlapping windows; controls match).
# Rollout: 600s x 32 sequences; --context-mode carried is requested for all --
# S2P2-family and PCT-LSTM decoders support it, window-shaped baselines
# (NHP/LSTM/SAHP) fall back to window mode with a log line (architectural,
# documented in the paper).
# Calibration is OURS ALONE by design: the factorized bounded head is what
# makes a certificate-preserving rate knob exist; coupled-head baselines have
# no equivalent. Prediction metrics come from the trained (uncalibrated = MLE)
# checkpoint for everyone, incl. ours -- calibration only affects simulation.
REPO="${REPO:-$HOME/simulation}"
DATA="${DATA:-/SAN/medic/TFOW/data/events/gmni_eth_7_v2_marks}"
MAXFILES="${MAXFILES:-7}"
CACHE="${CACHE:-$DATA/.tensor_cache_eval}"
TARGET_RATE="${TARGET_RATE:-3.77}"
EPOCHS="${EPOCHS:-40}"
SEQ="${SEQ:-1024}"; STRIDE="${STRIDE:-1024}"
ROOT="${ROOT:-$REPO/experiments/final_comparison}"

CAL=""
case "$SGE_TASK_ID" in
  1) TAG=nhp;      EXTRA="--decoder-type hawkes" ;;
  2) TAG=lstm;     EXTRA="--decoder-type lstm" ;;
  3) TAG=sahp;     EXTRA="--decoder-type sahp --sahp-layers 2 --sahp-heads 4" ;;
  4) TAG=pct-lstm; EXTRA="--decoder-type pct-lstm --ptp-dim 8" ;;
  5) TAG=s2p2;     EXTRA="--decoder-type s2p2 --s2p2-layers 2 --s2p2-readout output" ;;
  6) TAG=ss2p2-full
     EXTRA="--decoder-type ss2p2 --s2p2-layers 2 --ss2p2-wnorm-cap 6.0 --target-rate $TARGET_RATE --tbptt"
     CAL="--calibrate-rate -1" ;;
  *) echo "no config for task $SGE_TASK_ID"; exit 1 ;;
esac

cd "$REPO"
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4
B="$ROOT/$TAG"
# Fresh outputs every run: never reuse a stale checkpoint or old eval JSONs.
rm -rf "$B"
mkdir -p "$B/stylized_facts"; ML="$B/master.log"
log(){ echo "$@" | tee -a "$ML"; }
fail(){ log "DONE $(date) STATUS=1 stage=$1 rc=$2 BASE=$B"; exit 1; }
log "START $(date) TAG=$TAG host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tee -a "$ML"
CKPT="$B/train/best_model.pt"

log "TRAIN $(date)"
python3 -u -m volume_set_mtpp.training.train --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" \
  --channel-emb-size 64 --time-emb-size 64 --recurrent-hidden 64 \
  --batch-size 64 --epochs "$EPOCHS" --lr 2e-3 --weight-decay 1e-6 \
  --seq-length "$SEQ" --stride "$STRIDE" --num-workers 0 --save-every "$EPOCHS" \
  --mark-head categorical --set-loss-reduction sum --no-volume-input-scaling --allow-tf32 --seed 1 \
  $EXTRA --output-dir "$B/train" --log-dir "$B/train/logs" > "$B/train.log" 2>&1
TRAIN_RC=$?
log "TRAIN_RC=$TRAIN_RC"
{ [ "$TRAIN_RC" -eq 0 ] && [ -s "$CKPT" ]; } || { tail -25 "$B/train.log" | tee -a "$ML"; fail train "$TRAIN_RC"; }

log "GENUINE $(date)"
python3 -u -m volume_set_mtpp.evaluation.genuine_eval --checkpoint "$CKPT" --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" \
  --seq-length "$SEQ" --stride "$STRIDE" --batch-size 256 --device cuda --label "$TAG" \
  --dt-horizon 60 --dt-grid-points 128 --output "$B/genuine_${TAG}.json" 2>&1 | tee -a "$ML"
GEN_RC=$?
{ [ "$GEN_RC" -eq 0 ] && [ -s "$B/genuine_${TAG}.json" ]; } || fail genuine "$GEN_RC"

log "SF $(date) cal='${CAL:-none}'"
python3 -u -m volume_set_mtpp.evaluation.stylized_facts --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" \
  --checkpoint "$CKPT" --label "$TAG" --output-dir "$B/stylized_facts" --device cuda --sampler inversion \
  --context-mode carried $CAL \
  --seq-length "$SEQ" --stride "$STRIDE" --batch-size 256 --rollout-duration 600 --rollout-sequences 32 \
  --rollout-seed 1 --bucket-seconds 1.0 --max-real-windows 4096 > "$B/sf.log" 2>&1
SF_RC=$?
grep -E "CONTEXT_MODE|CALIBRAT" "$B/sf.log" | tee -a "$ML"
{ [ "$SF_RC" -eq 0 ] && [ -s "$B/stylized_facts/stylized_facts_${TAG}.json" ]; } \
  || { tail -25 "$B/sf.log" | tee -a "$ML"; fail sf "$SF_RC"; }
log "DONE $(date) STATUS=0 BASE=$B"
