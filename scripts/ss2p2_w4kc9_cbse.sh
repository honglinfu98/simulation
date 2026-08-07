#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N ss2p2_w4kc9
#$ -l h_rt=48:00:00
#$ -l tmem=32G
#$ -l gpu=true
#$ -l h=!hoots-207-1*
#$ -pe gpu 1
#$ -t 1-9
set -o pipefail

# LGM vs SS2P2 on Coinbase BTC/ETH/SOL. Protocol is multi_asset_cbse.sh
# VERBATIM (same data, cache, epochs, seq, streaming prediction, calibration,
# equal-duration facts); results land in experiments/ma_cbse/<coin>/ next to
# the existing ss2p2-full-s* baselines, so nothing is rerun.
#
# The arm: LGMSetDecoder = the SAME 2-layer S2P2 backbone + the SAME softmax
# mark head as SS2P2; ONLY the scalar total-rate factor differs (linear
# multi-timescale Hawkes ground, branching n projected to 0.95, mean rate
# pinned to the measured TRAIN-split rate). Sampler is inversion (the ground
# is unbounded, so SS2P2's constant thinning ceiling does not apply).
#
# Task map: t 1-3 -> btc s1-3, t 4-6 -> eth s1-3, t 7-9 -> sol s1-3.
REPO="${REPO:-$HOME/simulation}"
COINS=(btc eth sol)
CI=$(( (SGE_TASK_ID - 1) / 3 ))
COIN="${COINS[$CI]}"
SEED=$(( (SGE_TASK_ID - 1) % 3 + 1 ))
DATA="${DATA:-/SAN/medic/TFOW/data/events/cbse_${COIN}_7d}"
MAXFILES="${MAXFILES:-7}"
CACHE="${CACHE:-$DATA/.tensor_cache_eval}"
EPOCHS="${EPOCHS:-48}"
SEQ="${SEQ:-4096}"; STRIDE="${STRIDE:-4096}"
ROOT="${ROOT:-$REPO/experiments/ma_cbse/$COIN}"
ROLLOUT_SEEDS="${ROLLOUT_SEEDS:-1 2 3}"

EXTRA="--decoder-type ss2p2 --s2p2-layers 2 --ss2p2-wnorm-cap 9.0 --target-rate -1 --tbptt --s2p2-scan"
TAG="ss2p2-w4kc9-s${SEED}"
SAMPLER=thinning
SF_CAL="--calibrate-rate -1 --calibrate-split val --calibrate-probe-duration 600 --calibrate-final-tol 0.15"

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

log "RHO $(date)"
python3 -u - "$CKPT" <<'PY' 2>&1 | tee -a "$ML"
import sys, torch
from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False); cfg = ck["config"]
m = create_volume_set_mtpp(cfg.get("num_channels", 62), cfg, torch.device("cpu"), use_volume=cfg.get("use_volume", False))
m.load_state_dict(ck["model_state_dict"])
d = m.decoder
print("RHO n=%.4f pinned_rate=%.4f betas=%s" % (d.closed_form_rho(), float(d.target_rate),
      [round(float(b), 4) for b in d._betas()]))
PY

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
