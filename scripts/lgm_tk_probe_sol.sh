#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N lgm_tk_probe
#$ -l h_rt=6:00:00
#$ -l tmem=32G
#$ -l gpu=true
#$ -l h=!hoots-207-1*
#$ -pe gpu 1
#$ -t 1-6
set -o pipefail

# TYPED-KICK LGM, SOL, PIPELINE PROBE. The marked-cluster tuner (tune_n_typed.py,
# Directional_market_making) lands n*~0.80 for both kick variants but with a
# FLAT cross-scale shape under its iid-mark approximation; this probe re-scores
# n through the pipeline estimator itself (ignition feedback included), exactly
# the kfp protocol that arbitrated BTC/ETH. One donor mark seed (lgm-w4k48-s1),
# SF-only, 1 rollout seed. Task map: 1-3 = channel kicks n in {0.80,0.85,0.90};
# 4-6 = group kicks, same n grid. Checkpoints pre-assembled by
# assemble_typed_ckpt.py into lgm-tk{c,g}<nn>-s1/train/best_model.pt.
REPO="${REPO:-$HOME/simulation}"
COIN=sol
DATA="${DATA:-/SAN/medic/TFOW/data/events/cbse_${COIN}_7d}"
MAXFILES=7
CACHE="${CACHE:-$DATA/.tensor_cache_eval}"
SEQ=4096; STRIDE=4096
ROOT="$REPO/experiments/ma_cbse/$COIN"

NS=(80 85 90)
V=$([ "$SGE_TASK_ID" -le 3 ] && echo c || echo g)
NN=${NS[$(( (SGE_TASK_ID-1)%3 ))]}
TAG="lgm-tk${V}${NN}-s1"
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
log "START $(date) COIN=$COIN TAG=$TAG host=$(hostname) (typed-kick probe)"

log "RHO $(date)"
python3 -u - "$CKPT" <<'PY' 2>&1 | tee -a "$ML"
import sys, torch
import torch.nn.functional as F
from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False); cfg = ck["config"]
m = create_volume_set_mtpp(cfg.get("num_channels", 62), cfg, torch.device("cpu"), use_volume=cfg.get("use_volume", False))
m.load_state_dict(ck["model_state_dict"])
d = m.decoder
ew = float((d.p_bar * F.softplus(d.kick_raw)).sum())
print("RHO n=%.4f E[w]=%.4f w[MO]=%.1f/%.1f pinned_rate=%.4f" % (
    d.closed_form_rho(), ew, F.softplus(d.kick_raw[40]), F.softplus(d.kick_raw[41]), float(d.target_rate)))
PY

R=1
log "SF $(date) rollout_seed=$R"
mkdir -p "$B/sf_r$R"
python3 -u -m volume_set_mtpp.evaluation.stylized_facts --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" \
  --checkpoint "$CKPT" --label "$TAG" --output-dir "$B/sf_r$R" --device cuda --sampler "$SAMPLER" \
  --context-mode carried $SF_CAL --match-durations \
  --seq-length "$SEQ" --stride "$STRIDE" --batch-size 256 --rollout-duration 600 --rollout-sequences 32 \
  --rollout-seed "$R" --bucket-seconds 1.0 --max-real-windows 4096 > "$B/sf_r$R.log" 2>&1
SF_RC=$?
grep -E "CONTEXT_MODE|CALIBRAT|CAL_FALLBACK|CAL_VERIFY" "$B/sf_r$R.log" | tee -a "$ML"
{ [ "$SF_RC" -eq 0 ] && [ -s "$B/sf_r$R/stylized_facts_${TAG}.json" ]; } \
  || { tail -25 "$B/sf_r$R.log" | tee -a "$ML"; fail "sf_r$R" "$SF_RC"; }
log "DONE $(date) STATUS=0 COIN=$COIN BASE=$B"
