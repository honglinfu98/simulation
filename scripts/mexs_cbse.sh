#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N mexs
#$ -l h_rt=20:00:00
#$ -l tmem=32G
#$ -l gpu=true
#$ -l h=!hoots-207-1*
#$ -pe gpu 1
#$ -t 1-3
set -o pipefail

# MEXS: MEX-S2P2 with SPECTRAL CONTROL on the K x K routing -- a DOSE-RESPONSE.
#
#   T[:,j] = c_j * v[:,j] / ||v[:,j]||_1      c_j = c_max * sigmoid(mag_j)
#   => ||T[:,j]||_1 = c_j exactly, so rho(T) <= ||T||_1 <= c_max before any
#      eigendecomposition; the exact rho comes from torch.linalg.eigvals and is
#      hard-projected after every optimizer step (project_spectral), the matrix
#      analogue of the scalar project_subcritical.
#
# c_j is the net offspring of a type-j event -- exactly what the binned-count
# regression measures (SOL: LO/CO/IS 0.87-1.19, MO 1.04/0.25) -- so c_init=1.0
# is the measured mean rather than an arbitrary scale.
#
# THIS IS A DIAGNOSTIC, NOT ONLY A FIX.  The three arms are identical except for
# rho_max in {0.50, 0.80, 0.95}.  Unconstrained mex-matrix rolled out at 5.61x
# real rate with Fano 0.08-0.31 of real.  If the routing gain is what drives the
# operating point, rollout rate must fall monotonically with rho_max.
#
# PRE-REGISTERED:
#   1. rho(T) after training ~= rho_max on every arm (projection works)
#   2. rollout rate DECREASES monotonically in rho_max
#   3. IF rollout rate is FLAT across 0.50 / 0.80 / 0.95, the routing is NOT the
#      cause -- that falsifies the transition-matrix explanation and implicates
#      the carried-state rollout path shared by every tower model I built
#      (pct +190..483%, mex +354..372%, while pct-lstm sits at +0.3..3.1% and
#      s2p2-pub at +2.0..5.9% on the same harness).
# Outcome (3) is the informative one and I expect it is live: nothing in the
# tower architecture varied so far has moved CAL_TRANSFER at all.
REPO="${REPO:-$HOME/simulation}"
COIN="${COIN:-sol}"
RHOS=(0.50 0.80 0.95);  RHO=${RHOS[$((SGE_TASK_ID-1))]}
ARM="rho${RHO}"
DATA="${DATA:-/SAN/medic/TFOW/data/events/cbse_${COIN}_7d}"
MAXFILES=7
CACHE="${CACHE:-$DATA/.tensor_cache_eval}"
SEQ=4096; STRIDE=4096
ROOT="$REPO/experiments/mexs_cbse/$COIN"
TAG="mexs-${ARM}"
EPOCHS="${EPOCHS:-12}"
SEED="${SEED:-1}"

# per-asset mean event rate (compfit R_group totals); sets the s_k init only
case "$COIN" in
  sol) TRATE=22.26 ;;
  eth) TRATE=42.40 ;;
  btc) TRATE=38.40 ;;
esac
# every arm is identical EXCEPT rho_max: a dose-response on the routing gain.
MODE_ARGS="--pct-impulse-mode matrix --pct-per-tower-head --pct-rho-max $RHO \
  --pct-c-max 2.0 --pct-c-init 1.0" 

hostname; date
cd "$REPO" || exit 1
[ -d "$DATA" ] || { echo "SAN_NOT_VISIBLE $DATA"; exit 1; }
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4
# Scan memory scales as B*N*K*P and the doubling scan retains log2(N) steps:
# at B=8,N=4096,K=62,P=8 that is ~6 GB of intermediates alone -> OOM on a 12 GB
# card (job 7183138).  The K factor is inherent to one tower per type, so the
# batch is the knob.  expandable_segments cuts fragmentation on top.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

B="$ROOT/$TAG"; mkdir -p "$B"
CKPT="$B/best_model.pt"   # train.py --output-dir writes here (NOT $B/train/)
rm -rf "$B"/sf_r*
ML="$B/master.log"; : > "$ML"
log(){ echo "$@" | tee -a "$ML"; }
fail(){ log "DONE $(date) STATUS=1 stage=$1 rc=$2 BASE=$B"; exit 1; }
log "START $(date) COIN=$COIN ARM=$ARM TAG=$TAG rate=$TRATE host=$(hostname)"

log "SELFTEST $(date)"
python3 -u scripts/test_pct.py 2>&1 | tail -6 | tee -a "$ML"
[ "${PIPESTATUS[0]}" -eq 0 ] || fail selftest 1

log "TRAIN $(date)"
python3 -u -m volume_set_mtpp.training.train --data-dir "$DATA" --max-files "$MAXFILES" \
  --cache-dir "$CACHE" --output-dir "$B" --decoder-type mex-s2p2 $MODE_ARGS \
  --pct-tower-dim 8 --pct-state-dim 8 --pct-layers 2 --pct-rate-cap 6.0 \
  --target-rate "$TRATE" --mark-head categorical \
  --channel-emb-size 64 --time-emb-size 64 --recurrent-hidden 64 \
  --seq-length "$SEQ" --stride "$STRIDE" --batch-size "${BS:-2}" --epochs "$EPOCHS" \
  --lr 1e-3 --seed "$SEED" --device cuda 2>&1 | tail -40 | tee -a "$ML"
{ [ "${PIPESTATUS[0]}" -eq 0 ] && [ -s "$CKPT" ]; } || fail train 1

log "RATE-BOUND $(date)"
python3 -u - "$CKPT" <<'PY' 2>&1 | tee -a "$ML"
import sys, json, torch
from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False); cfg = ck["config"]
m = create_volume_set_mtpp(cfg.get("num_channels", 62), cfg, torch.device("cpu"),
                           use_volume=cfg.get("use_volume", False))
m.load_state_dict(ck["model_state_dict"]); d = m.decoder
lo, hi = d.rate_bounds()
print("BOUND ell_minus=%.4f ell_plus=%.3f ev/s  ratio_to_target=%.2fx  impulse=%s" % (
    lo, hi, hi / float(cfg.get("target_rate", 1.0)), d.impulse_mode))
if hasattr(d, "spectral_radius"):
    cb = d.column_budgets()
    print("SPECTRAL rho(T)=%.4f  c_j: min %.4f mean %.4f max %.4f  (rho_max=%s)" % (
        d.spectral_radius(), cb.min(), cb.mean(), cb.max(),
        cfg.get("pct_rho_max", "n/a")))
sp = d.spectrum()
for s in sp:
    ts = torch.tensor(s["timescale_s"]); fq = torch.tensor(s["freq_hz"])
    print("  layer %d: timescale %.4f-%.1f s  |freq| max %.3f Hz  osc frac %.2f" % (
        s["layer"], ts.min(), ts.max(), fq.abs().max(),
        float((fq.abs() > 1e-3).float().mean())))
PY

log "GENUINE $(date)"
python3 -u -m volume_set_mtpp.evaluation.genuine_eval --checkpoint "$CKPT" --data-dir "$DATA" \
  --max-files "$MAXFILES" --cache-dir "$CACHE" --seq-length "$SEQ" --stride "$STRIDE" \
  --batch-size 16 --device cuda --label "$TAG" --streaming --dt-horizon 60 \
  --dt-grid-points 32 --output "$B/genuine_${TAG}.json" 2>&1 | tail -20 | tee -a "$ML"
[ -s "$B/genuine_${TAG}.json" ] || fail genuine 1

# thinning is valid here: rate_bounds() is an exact closed-form ceiling
for R in 1 2; do
  log "SF $(date) rollout_seed=$R"
  mkdir -p "$B/sf_r$R"
  python3 -u -m volume_set_mtpp.evaluation.stylized_facts --data-dir "$DATA" \
    --max-files "$MAXFILES" --cache-dir "$CACHE" --checkpoint "$CKPT" --label "$TAG" \
    --output-dir "$B/sf_r$R" --device cuda --sampler thinning --context-mode carried \
    --calibrate-rate -1 --calibrate-split val --calibrate-probe-duration 600 \
    --calibrate-final-tol 0.15 --match-durations --seq-length "$SEQ" --stride "$STRIDE" --batch-size 64 \
    --rollout-duration 600 --rollout-sequences 32 --rollout-seed "$R" \
    --bucket-seconds 1.0 --max-real-windows 4096 > "$B/sf_r$R.log" 2>&1
  SF_RC=$?
  grep -E "CONTEXT_MODE|CALIBRAT" "$B/sf_r$R.log" | tee -a "$ML"
  { [ "$SF_RC" -eq 0 ] && [ -s "$B/sf_r$R/stylized_facts_${TAG}.json" ]; } \
    || { tail -25 "$B/sf_r$R.log" | tee -a "$ML"; fail "sf_r$R" "$SF_RC"; }
done

log "DONE $(date) STATUS=0 COIN=$COIN ARM=$ARM TAG=$TAG BASE=$B"
