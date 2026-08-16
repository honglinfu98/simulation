#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N dec
#$ -l h_rt=20:00:00
#$ -l tmem=32G
#$ -l gpu=true
#$ -l h=!hoots-207-1*
#$ -pe gpu 1
#$ -t 1-3
set -o pipefail

# DEC-S2P2: decomposed rate x per-type-tower composition, DISJOINT parameters.
#
#     lambda_k(t) = Lambda_SS2P2(u_rate) * softmax_k( tower readouts )
#
# The DTPP decomposition applied to an intensity model.  Two blocks with their
# own backbones AND their own channel embeddings; nothing is shared.  This
# restores d(Lambda)/d(theta_mark) == 0, which pct-s2p2 gave up.
#
# WHAT WENT WRONG IN PCT-S2P2 (job 7183266, same asset, same eval):
#   E[u]  1.43 (ep1) -> 2.76 (ep12)   while val loss fell MONOTONICALLY
#   KS    0.154      -> 0.492
#   CAL_TRANSFER: calibration verified on val (+3.2%) then missed test by +392%
#   rollout rates 3.4-12x real on all nine arms, seed spread <= 1.05x
# The mark term (~3.5 nats/event over 62 classes) dominated the time term and
# dragged the absolute rate, because p = lambda_k/sum(lambda) and Lambda were
# functions of the SAME vector.  scripts/test_dec.py asserts that is now
# structurally unreachable: rate gradients are bit-identical at 1x and 100x
# mark-loss weight.
#
# ARMS (SOL only, matched to the pct-s2p2 arms for a paired comparison):
#   all / own / ind   -- identical tower ablation, so any difference in
#                        calibration is attributable to the decomposition.
#
# PRE-REGISTERED PREDICTIONS:
#   1. E[u] stays ~1.0-1.3 and does NOT drift with epochs   (pct: 2.76)
#   2. time-rescaling KS well below pct's 0.48
#   3. CAL_TRANSFER within a few percent, not +392%
#   4. mark perplexity comparable to pct (towers do the same job)
# Failure of (1) falsifies the decomposition explanation of the pct failure.
#
# TRADED AWAY: mutual excitation now lives in the composition only -- the towers
# choose WHICH type fires, never HOW MANY events happen.
REPO="${REPO:-$HOME/simulation}"
COIN="${COIN:-sol}"
ARMS=(all own ind);  ARM=${ARMS[$((SGE_TASK_ID-1))]}
DATA="${DATA:-/SAN/medic/TFOW/data/events/cbse_${COIN}_7d}"
MAXFILES=7
CACHE="${CACHE:-$DATA/.tensor_cache_eval}"
SEQ=4096; STRIDE=4096
ROOT="$REPO/experiments/dec_cbse/$COIN"
TAG="dec-${ARM}"
EPOCHS="${EPOCHS:-12}"
SEED="${SEED:-1}"

# per-asset mean event rate (compfit R_group totals); sets the s_k init only
case "$COIN" in
  sol) TRATE=22.26 ;;
  eth) TRATE=42.40 ;;
  btc) TRATE=38.40 ;;
esac
case "$ARM" in
  all) MODE_ARGS="--pct-impulse-mode all" ;;
  own) MODE_ARGS="--pct-impulse-mode own" ;;
  ind) MODE_ARGS="--pct-impulse-mode own --pct-block-diag-mixers" ;;
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
python3 -u scripts/test_dec.py 2>&1 | tail -6 | tee -a "$ML"
[ "${PIPESTATUS[0]}" -eq 0 ] || fail selftest 1

log "TRAIN $(date)"
python3 -u -m volume_set_mtpp.training.train --data-dir "$DATA" --max-files "$MAXFILES" \
  --cache-dir "$CACHE" --output-dir "$B" --decoder-type dec-s2p2 $MODE_ARGS \
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
