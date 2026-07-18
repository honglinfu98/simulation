#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N ma_cbse
#$ -l h_rt=48:00:00
#$ -l tmem=32G
#$ -l gpu=true
#$ -l h=!hoots-207-1*
#$ -pe gpu 1
#$ -t 1-54
set -o pipefail

# MULTI-ASSET COMPARISON on Coinbase (BTC / ETH / SOL), 7 days Jan 2026.
# Protocol is final_comparison_v2.sh VERBATIM per coin -- same 6 models x
# 3 training seeds x 3 rollout seeds, val-split calibration with full-scale
# verification, lossless streaming prediction, equal-duration facts, strict
# failure modes. See final_comparison_v2.sh for the full methodology notes.
#
# Task map (54 = 3 coins x 6 models x 3 seeds):
#   t 1-18  -> btc      t 19-36 -> eth      t 37-54 -> sol
#   within a coin: model = ((t-1)%6), seed = ((t-1)%18)/6 + 1  (same as v2)
#   e.g. rerun all eth:      qsub -t 19-36 scripts/multi_asset_cbse.sh
#        rerun sol ss2p2 s2: t = 36 + 6 + 6 = 48
#
# Differences vs the Gemini ETH benchmark (deliberate):
#   - EPOCHS defaults to 12 (not 40): Coinbase has ~8-13x the events/day of
#     Gemini ETH, so 12 epochs is ~3-5x the ETH run's total gradient steps.
#     Cross-model comparisons are WITHIN a coin, so this is protocol-clean;
#     step counts remain exactly equal across models within each coin
#     (standard loader drops its partial final batch, matching TBPTT lanes).
#   - h_rt 48h: training + streaming eval + rollouts all scale with rate
#     (cbse btc ~19 ev/s vs gmni eth ~1.5 ev/s raw).
#   - SAHP stays UNCALIBRATED (k=1, dagger): its closed-loop rate divergence
#     is architectural (documented 2026-07-11/12 on gmni eth); we keep one
#     consistent protocol across assets rather than letting one asset's
#     bisection fail the task.
# Rates are never hard-coded: calibration target = measured VAL split rate,
# ss2p2's rate-head init = measured TRAIN split rate (--target-rate -1).
REPO="${REPO:-$HOME/simulation}"
COINS=(btc eth sol)
CI=$(( (SGE_TASK_ID - 1) / 18 ))
COIN="${COINS[$CI]}"
DATA="${DATA:-/SAN/medic/TFOW/data/events/cbse_${COIN}_7d}"
MAXFILES="${MAXFILES:-7}"
CACHE="${CACHE:-$DATA/.tensor_cache_eval}"
EPOCHS="${EPOCHS:-12}"
SEQ="${SEQ:-1024}"; STRIDE="${STRIDE:-1024}"
ROOT="${ROOT:-$REPO/experiments/ma_cbse/$COIN}"
ROLLOUT_SEEDS="${ROLLOUT_SEEDS:-1 2 3}"

MODELS=(nhp lstm sahp pct-lstm s2p2 ss2p2-full)
MI=$(( (SGE_TASK_ID - 1) % 6 ))
SEED=$(( ((SGE_TASK_ID - 1) % 18) / 6 + 1 ))
MODEL="${MODELS[$MI]}"
SF_CAL="--calibrate-rate -1 --calibrate-split val --calibrate-probe-duration 600 --calibrate-final-tol 0.15"
[ "$MODEL" = "sahp" ] && SF_CAL=""
case "$MODEL" in
  nhp)        EXTRA="--decoder-type hawkes" ;;
  lstm)       EXTRA="--decoder-type lstm" ;;
  sahp)       EXTRA="--decoder-type sahp --sahp-layers 2 --sahp-heads 4" ;;
  pct-lstm)   EXTRA="--decoder-type pct-lstm --ptp-dim 8" ;;
  s2p2)       EXTRA="--decoder-type s2p2 --s2p2-layers 2 --s2p2-readout output --s2p2-scan" ;;
  ss2p2-full) EXTRA="--decoder-type ss2p2 --s2p2-layers 2 --ss2p2-wnorm-cap 6.0 --target-rate -1 --tbptt --s2p2-scan" ;;
esac
TAG="${MODEL}-s${SEED}"

cd "$REPO"
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4
B="$ROOT/$TAG"
CKPT="$B/train/best_model.pt"
# RESUME=1: keep the existing checkpoint + genuine json and redo ONLY the SF
# stage (e.g. after a CAL_FINAL_FAIL) -- same trained model, fresh rollouts.
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
    --checkpoint "$CKPT" --label "$TAG" --output-dir "$B/sf_r$R" --device cuda --sampler inversion \
    --context-mode carried $SF_CAL --match-durations \
    --seq-length "$SEQ" --stride "$STRIDE" --batch-size 256 --rollout-duration 600 --rollout-sequences 32 \
    --rollout-seed "$R" --bucket-seconds 1.0 --max-real-windows 4096 > "$B/sf_r$R.log" 2>&1
  SF_RC=$?
  grep -E "CONTEXT_MODE|CALIBRAT" "$B/sf_r$R.log" | tee -a "$ML"
  { [ "$SF_RC" -eq 0 ] && [ -s "$B/sf_r$R/stylized_facts_${TAG}.json" ]; } \
    || { tail -25 "$B/sf_r$R.log" | tee -a "$ML"; fail "sf_r$R" "$SF_RC"; }
done
log "DONE $(date) STATUS=0 COIN=$COIN BASE=$B"
