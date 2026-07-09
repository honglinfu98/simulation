#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N tbptt_abl
#$ -l h_rt=16:00:00
#$ -l tmem=32G
#$ -l gpu=true
#$ -l h=!hoots-207-1*
#$ -pe gpu 1
#$ -t 1-4
set -o pipefail

# TBPTT ABLATION (single factor, paired arms).
#
# Hypothesis: windowed COLD-START training (state reset to init every window)
# mis-calibrates the learned baseline -- the model never trains on the warm
# states a free rollout lives in (cause A of the free-run rate inflation).
# Treatment: --tbptt (stateful training: stream-order lane batching, decoder
# state carried across windows, detached at boundaries; resets only at
# zone/file edges).
#
# Design: 2 models x 2 arms; ONLY --tbptt differs within a pair. Both arms use
# stride = seq = 1024 (TBPTT requires non-overlapping windows, so the control
# matches -- same windows, same gradient-step count, shuffled vs stream-order).
# Endpoint compensator everywhere (the MC estimator is a SEPARATE factor;
# round-1 showed its high-variance form collapses under clip+Adam).
# Eval = genuine_eval + stylized_facts --context-mode carried (600s x 32).
#
# Primary endpoints:
#   free-run sim_rate -> real   (cold arms sit ~30-64 ev/s vs real ~3.5)
#   mean_u                      (expect LESS moved than sim_rate: the integral
#                                bias is the other factor)
# Secondary: overall NLL / ACC (warmer states should not hurt prediction),
# F6/F8 long memory (training-side memory now matches the carried rollout).
REPO="${REPO:-$HOME/simulation}"
DATA="${DATA:-/SAN/medic/TFOW/data/events/gmni_eth_7_v2_marks}"
MAXFILES="${MAXFILES:-7}"
CACHE="${CACHE:-$DATA/.tensor_cache_eval}"
TARGET_RATE="${TARGET_RATE:-3.77}"
EPOCHS="${EPOCHS:-40}"
SEQ="${SEQ:-1024}"; STRIDE="${STRIDE:-1024}"
ROOT="${ROOT:-$REPO/experiments/tbptt_ablation}"

SS2P2="--decoder-type ss2p2 --s2p2-layers 2 --ss2p2-wnorm-cap 6.0 --target-rate $TARGET_RATE"
S2P2="--decoder-type s2p2 --s2p2-layers 2 --s2p2-readout output"
case "$SGE_TASK_ID" in
  1) TAG=ss2p2-cold;  EXTRA="$SS2P2" ;;                  # control: cold-start windows
  2) TAG=ss2p2-tbptt; EXTRA="$SS2P2 --tbptt" ;;          # treatment: carried state
  3) TAG=s2p2-cold;   EXTRA="$S2P2" ;;
  4) TAG=s2p2-tbptt;  EXTRA="$S2P2 --tbptt" ;;
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

log "SF $(date)"
python3 -u -m volume_set_mtpp.evaluation.stylized_facts --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" \
  --checkpoint "$CKPT" --label "$TAG" --output-dir "$B/stylized_facts" --device cuda --sampler inversion \
  --context-mode carried \
  --seq-length "$SEQ" --stride "$STRIDE" --batch-size 256 --rollout-duration 600 --rollout-sequences 32 \
  --rollout-seed 1 --bucket-seconds 1.0 --max-real-windows 4096 > "$B/sf.log" 2>&1
SF_RC=$?
{ [ "$SF_RC" -eq 0 ] && [ -s "$B/stylized_facts/stylized_facts_${TAG}.json" ]; } \
  || { tail -25 "$B/sf.log" | tee -a "$ML"; fail sf "$SF_RC"; }
log "DONE $(date) STATUS=0 BASE=$B"
