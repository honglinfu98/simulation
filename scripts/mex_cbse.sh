#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N mex
#$ -l h_rt=20:00:00
#$ -l tmem=32G
#$ -l gpu=true
#$ -l h=!hoots-207-1*
#$ -pe gpu 1
#$ -t 1-3
set -o pipefail

# MEX-S2P2: one SS2P2 per event type, coupled by a SHARED EXPLICIT transition.
#
#   tower k = its OWN SS2P2 with its OWN decoupled softmin-bounded head
#   impulse routing:  eps_k = sum_j T[k,j] * m_j * alpha_j     (T learned, K x K)
#   lambda_k = s_k softplus( c - softplus(c - w_k^T(o_k . tanh u_k) - b_k) )
#   Lambda   = sum_k lambda_k        p(k) = lambda_k / Lambda
#
# So lambda_k is driven by impulses from EVERY type -- mutual excitation is in
# the RATE, not merely in the composition.  T is a readable parameter: T[k,j] is
# how strongly type j drives tower k, a direct answer to "where does impact come
# from and where does it go", without needing counterfactual intervention.
# T is initialised at I + 0.01*noise, so the model STARTS at "own" and must pay
# likelihood for any cross-excitation it ends up with.
#
# KNOWN TRADE (stated, not hidden): Lambda = sum_k lambda_k and p = lambda_k/Lambda
# are two readings of the same vector, so this inherits pct-s2p2's
# non-separability -- there E[u] drifted 1.43 -> 2.76 while val loss fell
# monotonically, and rollout ran 3.4-12x hot.  dec-s2p2 on the same asset gave
# that up and reached E[u] 1.83 / KS 0.195 / mark ppl 19.68.  This arm buys
# rate-level mutual excitation back and will likely pay for it in calibration.
#
# ARMS (SOL): routing only -- all three use per-tower decoupled heads.
#   matrix : learned K x K transition   (the proposed model)
#   all    : every event jumps every tower, untyped   (pct-s2p2 default)
#   own    : tower k sees only type k; cross-talk via mixers only
#
# PRE-REGISTERED:
#   1. matrix beats all/own on mark perplexity (typed routing should help)
#   2. T learns off-diagonal mass: max_{k!=j}|T[k,j]| >> 0.01 init noise
#   3. E[u] lands nearer pct-s2p2 (2.8) than dec-s2p2 (1.8) -- the trade is real
#      and I expect to see it; if E[u] stays ~1.8 the coupling story needs revising
REPO="${REPO:-$HOME/simulation}"
COIN="${COIN:-sol}"
ARMS=(matrix all own);  ARM=${ARMS[$((SGE_TASK_ID-1))]}
DATA="${DATA:-/SAN/medic/TFOW/data/events/cbse_${COIN}_7d}"
MAXFILES=7
CACHE="${CACHE:-$DATA/.tensor_cache_eval}"
SEQ=4096; STRIDE=4096
ROOT="$REPO/experiments/mex_cbse/$COIN"
TAG="mex-${ARM}"
EPOCHS="${EPOCHS:-12}"
SEED="${SEED:-1}"

# per-asset mean event rate (compfit R_group totals); sets the s_k init only
case "$COIN" in
  sol) TRATE=22.26 ;;
  eth) TRATE=42.40 ;;
  btc) TRATE=38.40 ;;
esac
# every arm uses per-tower decoupled heads; they differ only in HOW impulses
# are routed between types, which is the mutual-excitation question.
case "$ARM" in
  matrix) MODE_ARGS="--pct-impulse-mode matrix --pct-per-tower-head" ;;
  all)    MODE_ARGS="--pct-impulse-mode all    --pct-per-tower-head" ;;
  own)    MODE_ARGS="--pct-impulse-mode own    --pct-per-tower-head" ;;
esac

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
