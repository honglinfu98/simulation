#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N final_v2
#$ -l h_rt=24:00:00
#$ -l tmem=32G
#$ -l gpu=true
#$ -l h=!hoots-207-1*
#$ -pe gpu 1
#$ -t 1-18
set -o pipefail

# FINAL COMPARISON v2 (paper headline tables), seq-1024, multi-seed, strict.
#
# 6 models x 3 TRAINING seeds (tasks: model = (t-1)%6, seed = (t-1)/6):
#   nhp        Neural Hawkes CT-LSTM (Mei & Eisner)      [carried rollout]
#   lstm       plain-LSTM decoder -- ADAPTED GENERIC backbone, not a
#              paper-faithful re-implementation (see lstm_decoder.py)
#                                                        [carried rollout]
#   sahp       SAHP-style causal attention -- ADAPTED GENERIC backbone, not a
#              paper-faithful SAHP (see sahp_decoder.py) [window: no recurrent state]
#   pct-lstm   per-type parallel CT-LSTM                 [carried rollout]
#   s2p2       S2P2 (Shi & Cartlidge)                    [carried rollout]
#   ss2p2-full OURS: SS2P2 heads + TBPTT training        [carried rollout]
#
# Methodology fixes vs v1 (review items):
#   - EVERY model is rate-calibrated (--calibrate-rate -1): the sim-time
#     intensity scale k is mark-preserving for all decoders; SS2P2 additionally
#     keeps an exact thinning bound under k. Calibration target + probe
#     warm-starts come from the VALIDATION split (--calibrate-split val).
#   - Prediction scored on EVERY test event with the streaming evaluator
#     (--streaming: state carried across windows where supported).
#   - Real-vs-sim facts on EQUAL-DURATION bootstrap segments (--match-durations).
#   - 3 rollout seeds per checkpoint; report aggregates with 95% CIs
#     (scripts/final_report.py).
#   - Training/eval are strict: any batch failure or non-finite value aborts
#     the task; the collector fails on missing models.
# Shared config: seq 1024 / stride 1024 / batch 64 / 40 epochs / endpoint
# compensator (identical windows + gradient-step count for all).
REPO="${REPO:-$HOME/simulation}"
DATA="${DATA:-/SAN/medic/TFOW/data/events/gmni_eth_7_v2_marks}"
MAXFILES="${MAXFILES:-7}"
CACHE="${CACHE:-$DATA/.tensor_cache_eval}"
TARGET_RATE="${TARGET_RATE:-3.77}"
EPOCHS="${EPOCHS:-40}"
SEQ="${SEQ:-1024}"; STRIDE="${STRIDE:-1024}"
ROOT="${ROOT:-$REPO/experiments/final_v2}"
ROLLOUT_SEEDS="${ROLLOUT_SEEDS:-1 2 3}"

MODELS=(nhp lstm sahp pct-lstm s2p2 ss2p2-full)
MI=$(( (SGE_TASK_ID - 1) % 6 ))
SEED=$(( (SGE_TASK_ID - 1) / 6 + 1 ))
MODEL="${MODELS[$MI]}"
case "$MODEL" in
  nhp)        EXTRA="--decoder-type hawkes" ;;
  lstm)       EXTRA="--decoder-type lstm" ;;
  sahp)       EXTRA="--decoder-type sahp --sahp-layers 2 --sahp-heads 4" ;;
  pct-lstm)   EXTRA="--decoder-type pct-lstm --ptp-dim 8" ;;
  s2p2)       EXTRA="--decoder-type s2p2 --s2p2-layers 2 --s2p2-readout output" ;;
  ss2p2-full) EXTRA="--decoder-type ss2p2 --s2p2-layers 2 --ss2p2-wnorm-cap 6.0 --target-rate $TARGET_RATE --tbptt" ;;
esac
TAG="${MODEL}-s${SEED}"

cd "$REPO"
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4
B="$ROOT/$TAG"
rm -rf "$B"
mkdir -p "$B"; ML="$B/master.log"
log(){ echo "$@" | tee -a "$ML"; }
fail(){ log "DONE $(date) STATUS=1 stage=$1 rc=$2 BASE=$B"; exit 1; }
log "START $(date) TAG=$TAG host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tee -a "$ML"
CKPT="$B/train/best_model.pt"

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

for R in $ROLLOUT_SEEDS; do
  log "SF $(date) rollout_seed=$R"
  mkdir -p "$B/sf_r$R"
  python3 -u -m volume_set_mtpp.evaluation.stylized_facts --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" \
    --checkpoint "$CKPT" --label "$TAG" --output-dir "$B/sf_r$R" --device cuda --sampler inversion \
    --context-mode carried --calibrate-rate -1 --calibrate-split val --match-durations \
    --seq-length "$SEQ" --stride "$STRIDE" --batch-size 256 --rollout-duration 600 --rollout-sequences 32 \
    --rollout-seed "$R" --bucket-seconds 1.0 --max-real-windows 4096 > "$B/sf_r$R.log" 2>&1
  SF_RC=$?
  grep -E "CONTEXT_MODE|CALIBRAT" "$B/sf_r$R.log" | tee -a "$ML"
  { [ "$SF_RC" -eq 0 ] && [ -s "$B/sf_r$R/stylized_facts_${TAG}.json" ]; } \
    || { tail -25 "$B/sf_r$R.log" | tee -a "$ML"; fail "sf_r$R" "$SF_RC"; }
done
log "DONE $(date) STATUS=0 BASE=$B"
