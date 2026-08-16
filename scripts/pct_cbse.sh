#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N pct
#$ -l h_rt=20:00:00
#$ -l tmem=32G
#$ -l gpu=true
#$ -l h=!hoots-207-1*
#$ -pe gpu 1
#$ -t 1-9
set -o pipefail

# PCT-S2P2: parallel per-type S2P2 with an SS2P2-bounded per-type rate head.
#
# K towers x L diagonalized LLH layers; tower k's top output decodes ONLY
# lambda_k, so DESTINATION attribution is exact by construction.  Cross-tower
# information moves through a shared Linear(K*H -> K*H) BETWEEN layers, never
# at event instants -- that keeps b_i state-independent and the recurrence
# scannable at O(log N).  Eq.-18-faithful mixing would be O(N).
#
# RATE HEAD: the published per-type ScaledSoftplus is unbounded, so sum_k
# lambda_k has no dominating rate and thinning has no valid ceiling.  We use
# SS2P2's asymmetric softmin cap instead:
#     z = c - softplus(c - z_raw) <= c  =>  Lambda <= sum_k s_k softplus(c)
# an EXACT closed-form ceiling.  One-sided by design: ceiling for simulation
# stability, floor exactly zero for prediction (the old symmetric G1 sandwich
# welded the quiet floor to the burst scale and cost the quiet regime).
# At c=6 the ceiling sits at softplus(6)/ln2 = 8.66x the baseline total rate.
#
# ABLATION GRID (the KDD'22 "~2%" ablation analogue), 3 assets x 3 arms:
#   all   : every event jumps every tower through that tower's own E~
#           (direct mutual excitation + mixer route)
#   own   : tower k receives only type-k impulses; cross-talk via mixers ONLY
#   ind   : own + mixers masked to their K diagonal HxH blocks -> fully
#           independent towers, no cross-type path at all
# Differencing (all - own) isolates the direct-impulse route from the mixer
# route -- the one decomposition §7 says is NOT readable from parameters.
#
# Excitation claim discipline: cross-type kernels are sign-unconstrained
# (complex jumps, Re(C x) readout).  These towers are mutually INFLUENCING by
# construction and mutually EXCITING only where the data says so; the softmin
# cap is monotone so it constrains magnitude, never sign.  Read excitation
# empirically (G_jk by counterfactual injection), never claim it structurally.
REPO="${REPO:-$HOME/simulation}"
COINS=(sol eth btc); COIN=${COINS[$(( (SGE_TASK_ID-1)/3 ))]}
ARMS=(all own ind);  ARM=${ARMS[$(( (SGE_TASK_ID-1)%3 ))]}
DATA="${DATA:-/SAN/medic/TFOW/data/events/cbse_${COIN}_7d}"
MAXFILES=7
CACHE="${CACHE:-$DATA/.tensor_cache_eval}"
SEQ=4096; STRIDE=4096
ROOT="$REPO/experiments/pct_cbse/$COIN"
TAG="pct-${ARM}"
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
python3 -u scripts/test_pct.py 2>&1 | tail -6 | tee -a "$ML"
[ "${PIPESTATUS[0]}" -eq 0 ] || fail selftest 1

log "TRAIN $(date)"
python3 -u -m volume_set_mtpp.training.train --data-dir "$DATA" --max-files "$MAXFILES" \
  --cache-dir "$CACHE" --output-dir "$B" --decoder-type pct-s2p2 $MODE_ARGS \
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
