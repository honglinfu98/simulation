#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N lgm_kf_sol
#$ -l h_rt=12:00:00
#$ -l tmem=32G
#$ -l gpu=true
#$ -l h=!hoots-207-1*
#$ -pe gpu 1
#$ -t 1-3
set -o pipefail

# KIRCHNER-FITTED LGM, SOL: eval-only. The ground (a_m, beta_m) was fitted by
# binned-count regression on the real train zones (scripts/kirchner_fit_lgm.py
# in the Directional_market_making repo; n=0.99, validated by ground-only
# simulation to match real Fano within a few % at 5-50s) and transplanted into
# the trained lgm-s* mark checkpoints (marks untouched -- rate-neutral
# factorization). This job runs RHO + streaming genuine eval + 3 calibrated
# rollouts on the pre-built checkpoints in experiments/ma_cbse/sol/lgm-kf-s*/.
REPO="${REPO:-$HOME/simulation}"
COIN=sol
DATA="${DATA:-/SAN/medic/TFOW/data/events/cbse_${COIN}_7d}"
MAXFILES=7
CACHE="${CACHE:-$DATA/.tensor_cache_eval}"
SEQ=4096; STRIDE=4096
ROOT="$REPO/experiments/ma_cbse/$COIN"
ROLLOUT_SEEDS="1 2 3"
SEED=$SGE_TASK_ID
TAG="lgm-kf-s${SEED}"
SAMPLER=inversion
SF_CAL="--calibrate-rate -1 --calibrate-split val --calibrate-probe-duration 600 --calibrate-final-tol 0.15"

cd "$REPO"
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4
B="$ROOT/$TAG"
CKPT="$B/train/best_model.pt"
[ -s "$CKPT" ] || { echo "missing checkpoint $CKPT"; exit 1; }
rm -rf "$B"/sf_r*
ML="$B/master.log"; : > "$ML"
log(){ echo "$@" | tee -a "$ML"; }
fail(){ log "DONE $(date) STATUS=1 stage=$1 rc=$2 BASE=$B"; exit 1; }
log "START $(date) COIN=$COIN TAG=$TAG host=$(hostname) (eval-only, Kirchner ground)"

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
