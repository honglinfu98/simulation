#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N lgm_gen
#$ -l h_rt=12:00:00
#$ -l tmem=32G
#$ -l gpu=true
#$ -l h=!hoots-207-1*
#$ -pe gpu 1
#$ -t 1-9
set -o pipefail

# MODE-SPACE GENERATOR: explicit mutual excitation between marks, carried by R
# latent modes with a DENSE generator G, parameterized in G's eigenbasis
# (lam_r = -delta_r + i*omega_r; P folded into the emission/readout maps, since
# only the spectrum is identifiable).  Induced lag-integrated mark-to-mark
# transition matrix has the closed form  M = Re( U (-Lam)^-1 V^T ).
#
# WHY THIS IS NOT STAGE A AGAIN.  Stage A put type structure in the ground's
# MAGNITUDE, which multiplies n.  At n=0.99 the budget for that is 1% and the
# measured spread needed 24-528%, so dispersion blew up 1.9-4.0x.  The
# generator writes only into the mark LOGITS, which are softmax normalized, so
# the total intensity is algebraically independent of every generator
# parameter.  scripts/test_gen.py asserts d(Lambda)/d(gen) == 0 by autograd.
# Fano cannot move.  That makes prediction (1) below a test of the PLUMBING,
# not of the science.
#
# Only gen_* trains; backbone, mark head and ground are frozen and verified
# bit-identical afterwards.  gen_U is zero-init, so epoch 0 reproduces the
# donor exactly and every delta reported is attributable to the generator.
#
# PRE-REGISTERED PREDICTIONS:
#   1. Fano within rollout-seed noise of kf2 at every scale, and kappa ~1.01
#      -- STRUCTURAL.  Any violation is a state-plumbing bug, not a result.
#   2. val set_nll and genuine mark perplexity improve vs the donor.
#      (kf2 SOL s3 reference: acc 0.1952, ppl 23.9575)
#   3. rollout p(MO) moves toward empirical.
#      (kf2 SOL s3 markcal: emp 0.00174, tf 0.00118, roll 0.00064)
# Failure of (2) means mark-to-mark mutual excitation carries no signal the
# S2P2 backbone had not already captured -- the honest null for this design.
REPO="${REPO:-$HOME/simulation}"
COINS=(btc eth sol); COIN=${COINS[$(( (SGE_TASK_ID-1)/3 ))]}
DATA="${DATA:-/SAN/medic/TFOW/data/events/cbse_${COIN}_7d}"
MAXFILES=7
CACHE="${CACHE:-$DATA/.tensor_cache_eval}"
SEQ=4096; STRIDE=4096
ROOT="$REPO/experiments/ma_cbse/$COIN"
SEED=$(( (SGE_TASK_ID-1)%3 + 1 ))
TAG="lgm-gen-s${SEED}"
DONOR="$ROOT/lgm-kf2-s${SEED}/train/best_model.pt"
RANK="${RANK:-32}"
EPOCHS="${EPOCHS:-4}"
SAMPLER=inversion
SF_CAL="--calibrate-rate -1 --calibrate-split val --calibrate-probe-duration 600 --calibrate-final-tol 0.15"

hostname; date
cd "$REPO" || exit 1
[ -d "$DATA" ] || { echo "SAN_NOT_VISIBLE $DATA"; exit 1; }
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4

B="$ROOT/$TAG"; mkdir -p "$B/train"
CKPT="$B/train/best_model.pt"
[ -s "$DONOR" ] || { echo "missing donor $DONOR"; exit 1; }
rm -rf "$B"/sf_r*
ML="$B/master.log"; : > "$ML"
log(){ echo "$@" | tee -a "$ML"; }
fail(){ log "DONE $(date) STATUS=1 stage=$1 rc=$2 BASE=$B"; exit 1; }
log "START $(date) COIN=$COIN TAG=$TAG R=$RANK host=$(hostname)"

log "SELFTEST $(date)"
python3 -u scripts/test_gen.py 2>&1 | tail -20 | tee -a "$ML"
[ "${PIPESTATUS[0]}" -eq 0 ] || fail selftest 1

log "FINETUNE $(date)"
python3 -u scripts/finetune_generator.py --checkpoint "$DONOR" --data-dir "$DATA" \
  --out "$CKPT" --rank "$RANK" --epochs "$EPOCHS" --max-files "$MAXFILES" \
  --cache-dir "$CACHE" --seq-length "$SEQ" --stride "$STRIDE" --batch-size 16 \
  --device cuda --seed "$SEED" 2>&1 | tee -a "$ML"
{ [ "${PIPESTATUS[0]}" -eq 0 ] && [ -s "$CKPT" ]; } || fail finetune 1

log "RHO $(date)"
python3 -u - "$CKPT" <<'PY' 2>&1 | tee -a "$ML"
import sys, torch
from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False); cfg = ck["config"]
m = create_volume_set_mtpp(cfg.get("num_channels", 62), cfg, torch.device("cpu"), use_volume=cfg.get("use_volume", False))
m.load_state_dict(ck["model_state_dict"])
d = m.decoder
print("RHO n=%.6f pinned_rate=%.4f betas=%s" % (
      d.closed_form_rho(), float(d.target_rate),
      [round(float(b), 4) for b in d._betas()]))
print("GEN " + d.gen_summary())
PY

log "GENUINE-STREAMING $(date)"
python3 -u -m volume_set_mtpp.evaluation.genuine_eval --checkpoint "$CKPT" --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" \
  --seq-length "$SEQ" --stride "$STRIDE" --batch-size 64 --device cuda --label "$TAG" \
  --streaming --dt-horizon 60 --dt-grid-points 32 --output "$B/genuine_${TAG}.json" 2>&1 | tail -20 | tee -a "$ML"
[ -s "$B/genuine_${TAG}.json" ] || fail genuine 1

for R in 1 2 3; do
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

log "MARKCAL $(date)"
python3 -u scripts/mark_calibration_probe.py --checkpoint "$CKPT" --data-dir "$DATA" \
  --max-files "$MAXFILES" --cache-dir "$CACHE" --typed-json "$REPO/kirchner_typed_${COIN}.json" --label "$TAG" \
  --output "$B/markcal_${TAG}.json" --seq-length "$SEQ" --stride "$STRIDE" \
  --batch-size 64 --device cuda --tf-batches 200 --max-real-windows 4096 \
  --rollout-duration 600 --rollout-sequences 32 --rollout-seed 1 2>&1 | tail -8 | tee -a "$ML"

log "DONE $(date) STATUS=0 COIN=$COIN TAG=$TAG BASE=$B"
