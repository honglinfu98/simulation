#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N s2p2_pub
#$ -l h_rt=48:00:00
#$ -l tmem=32G
#$ -l gpu=true
#$ -l h=!hoots-207-1*
#$ -pe gpu 1
#$ -t 1-9
set -o pipefail

# FAITHFUL PUBLISHED-S2P2 baseline arms (s2p2-pub): protocol is
# multi_asset_cbse.sh VERBATIM (same data, epochs, eval, calibration, SF
# stages), one extra model. Decoder: the official PAPER configuration per
# the repo's example config (Int_Backward_LLH: backward-ZOH input drive,
# relative-time input-dependent dynamics, post-norm residual, GELU, complex
# DPLR states) with the per-type ScaledSoftplus head; verified value-level
# against yuxinc17/EasyTemporalPointProcess@70038ed (see
# volume_set_mtpp/models/s2p2_pub_decoder.py). Trains with the exact
# Hillis-Steele parallel scan over events (fp64 scan-vs-loop parity ~1e-12);
# samples by compensator inversion (no closed-form dominating rate, like
# the other baselines). Sizes benchmark-matched: H=64, L=2, P=64.
#
# Task map (9 = 3 coins x 3 seeds):
#   t 1-3 -> btc s1-3    t 4-6 -> eth s1-3    t 7-9 -> sol s1-3
REPO="${REPO:-$HOME/simulation}"
COINS=(btc eth sol)
CI=$(( (SGE_TASK_ID - 1) / 3 ))
COIN="${COINS[$CI]}"
DATA="${DATA:-/SAN/medic/TFOW/data/events/cbse_${COIN}_7d}"
MAXFILES="${MAXFILES:-7}"
CACHE="${CACHE:-$DATA/.tensor_cache_eval}"
EPOCHS="${EPOCHS:-12}"
SEQ="${SEQ:-1024}"; STRIDE="${STRIDE:-1024}"
ROOT="${ROOT:-$REPO/experiments/ma_cbse/$COIN}"
ROLLOUT_SEEDS="${ROLLOUT_SEEDS:-1 2 3}"

MODEL="s2p2-pub"
SEED=$(( (SGE_TASK_ID - 1) % 3 + 1 ))
SF_CAL="--calibrate-rate -1 --calibrate-split val --calibrate-probe-duration 600 --calibrate-final-tol 0.15"
EXTRA="--decoder-type s2p2-pub --pub-state-dim 64 --pub-layers 2 --s2p2-scan"
TAG="${MODEL}-s${SEED}"
SAMPLER=inversion

cd "$REPO"
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4
B="$ROOT/$TAG"
CKPT="$B/train/best_model.pt"
if [ "${RESUME:-0}" = "1" ] && [ -s "$CKPT" ]; then
  rm -rf "$B"/sf_r*
else
  rm -rf "$B"
fi
mkdir -p "$B"; ML="$B/master.log"
log(){ echo "$@" | tee -a "$ML"; }
fail(){ log "DONE $(date) STATUS=1 stage=$1 rc=$2 BASE=$B"; exit 1; }
log "START $(date) COIN=$COIN TAG=$TAG host=$(hostname) RESUME=${RESUME:-0}"; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tee -a "$ML"

if [ "${RESUME:-0}" = "1" ] && [ -s "$CKPT" ] && [ -s "$B/genuine_${TAG}.json" ]; then
  log "RESUME: reusing checkpoint + genuine json; redoing SF only"
else
log "TRAIN $(date) seed=$SEED"
python3 -u -m volume_set_mtpp.training.train --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" \
  --channel-emb-size 64 --time-emb-size 64 --recurrent-hidden 64 \
  --batch-size 64 --epochs "$EPOCHS" --lr 2e-3 --weight-decay 1e-6 \
  --seq-length "$SEQ" --stride "$STRIDE" --num-workers 0 --save-every "$EPOCHS" \
  --mark-head categorical --set-loss-reduction sum --no-volume-input-scaling --allow-tf32 --seed "$SEED" \
  $EXTRA --output-dir "$B/train" --log-dir "$B/train/logs" > "$B/train.log" 2>&1
TRAIN_RC=$?
log "TRAIN_RC=$TRAIN_RC"
{ [ "$TRAIN_RC" -eq 0 ] && [ -s "$CKPT" ]; } || { tail -25 "$B/train.log" | tee -a "$ML"; fail train "$TRAIN_RC"; }

log "GENUINE-STREAMING $(date)"
python3 -u -m volume_set_mtpp.evaluation.genuine_eval --checkpoint "$CKPT" --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" \
  --seq-length "$SEQ" --stride "$STRIDE" --batch-size 64 --device cuda --label "$TAG" \
  --streaming --dt-horizon 60 --dt-grid-points 32 --output "$B/genuine_${TAG}.json" 2>&1 | tail -30 | tee -a "$ML"
GEN_RC=$?
{ [ "$GEN_RC" -eq 0 ] && [ -s "$B/genuine_${TAG}.json" ]; } || fail genuine "$GEN_RC"
fi

for R in $ROLLOUT_SEEDS; do
  log "SF $(date) rollout_seed=$R"
  mkdir -p "$B/sf_r$R"
  python3 -u -m volume_set_mtpp.evaluation.stylized_facts --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" \
    --checkpoint "$CKPT" --label "$TAG" --output-dir "$B/sf_r$R" --device cuda --sampler "$SAMPLER" \
    --context-mode carried $SF_CAL --match-durations \
    --seq-length "$SEQ" --stride "$STRIDE" --batch-size 256 --rollout-duration 600 --rollout-sequences 32 \
    --rollout-seed "$R" --bucket-seconds 1.0 --max-real-windows 4096 > "$B/sf_r$R.log" 2>&1
  SF_RC=$?
  grep -E "CONTEXT_MODE|CALIBRAT" "$B/sf_r$R.log" | tee -a "$ML"
  { [ "$SF_RC" -eq 0 ] && [ -s "$B/sf_r$R/stylized_facts_${TAG}.json" ]; } \
    || { tail -25 "$B/sf_r$R.log" | tee -a "$ML"; fail "sf_r$R" "$SF_RC"; }
done
log "DONE $(date) STATUS=0 COIN=$COIN BASE=$B"
