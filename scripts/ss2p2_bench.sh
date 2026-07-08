#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N ss2p2_bench
#$ -l h_rt=8:00:00
#$ -l tmem=32G
#$ -l gpu=true
#$ -l gpu_type=h100
#$ -pe gpu 1
#$ -t 1-2
set -o pipefail

# Comparable SS2P2 vs S2P2 benchmark on Gemini-ETH single-item data.
# IDENTICAL config for both (seq64/stride32, 40 epochs, categorical marks).
# Per task: train -> genuine_eval (prediction ACC/PPL) -> stylized_facts
# (simulation: Fano/clustering/kurtosis vs real). Task1=ss2p2, Task2=s2p2.
REPO="${REPO:-$HOME/simulation}"
DATA="${DATA:-/SAN/medic/TFOW/data/events/gmni_eth_7_v2_marks}"
MAXFILES="${MAXFILES:-7}"
CACHE="${CACHE:-$DATA/.tensor_cache_ss2p2bench}"
TARGET_RATE="${TARGET_RATE:-3.77}"
EPOCHS="${EPOCHS:-40}"
SEQ="${SEQ:-64}"; STRIDE="${STRIDE:-32}"
ROOT="${ROOT:-$REPO/experiments/ss2p2_bench}"

case "$SGE_TASK_ID" in
  1) TAG=ss2p2; EXTRA="--decoder-type ss2p2 --s2p2-layers 2 --ss2p2-wnorm-cap 6.0 --target-rate $TARGET_RATE" ;;
  2) TAG=s2p2;  EXTRA="--decoder-type s2p2 --s2p2-layers 2 --s2p2-readout output" ;;
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
  --batch-size 256 --epochs "$EPOCHS" --lr 2e-3 --weight-decay 1e-6 \
  --seq-length "$SEQ" --stride "$STRIDE" --num-workers 0 --save-every "$EPOCHS" \
  --mark-head categorical --set-loss-reduction sum --no-volume-input-scaling --allow-tf32 --seed 1 \
  $EXTRA --output-dir "$B/train" --log-dir "$B/train/logs" > "$B/train.log" 2>&1
TRAIN_RC=$?
log "TRAIN_RC=$TRAIN_RC"
{ [ "$TRAIN_RC" -eq 0 ] && [ -s "$CKPT" ]; } || { tail -25 "$B/train.log" | tee -a "$ML"; fail train "$TRAIN_RC"; }

log "GENUINE $(date)"
python3 -u -m volume_set_mtpp.evaluation.genuine_eval --checkpoint "$CKPT" --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" \
  --seq-length "$SEQ" --stride "$STRIDE" --batch-size 256 --device cuda --label "$TAG" --output "$B/genuine_${TAG}.json" 2>&1 | tee -a "$ML"
GEN_RC=$?
{ [ "$GEN_RC" -eq 0 ] && [ -s "$B/genuine_${TAG}.json" ]; } || fail genuine "$GEN_RC"

log "SF $(date)"
python3 -u -m volume_set_mtpp.evaluation.stylized_facts --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" \
  --checkpoint "$CKPT" --label "$TAG" --output-dir "$B/stylized_facts" --device cuda \
  --seq-length "$SEQ" --stride "$STRIDE" --batch-size 256 --rollout-duration 600 --rollout-sequences 32 \
  --rollout-seed 1 --bucket-seconds 1.0 --max-real-windows 4096 > "$B/sf.log" 2>&1
SF_RC=$?
{ [ "$SF_RC" -eq 0 ] && [ -s "$B/stylized_facts/stylized_facts_${TAG}.json" ]; } \
  || { tail -25 "$B/sf.log" | tee -a "$ML"; fail sf "$SF_RC"; }
log "DONE $(date) STATUS=0 BASE=$B"
