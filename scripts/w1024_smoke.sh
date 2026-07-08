#!/usr/bin/env bash
# Smoke test for the w1024 pipeline: runs the SAME three stages as
# eval_worker_w1024.sh (train seq-1024 -> genuine_eval -> stylized_facts
# --context-mode carried) with tiny knobs, so the full path -- data read,
# training, checkpoint, prediction validation, simulation validation -- is
# verified in minutes before submitting the real array.
#
#   local (synthetic data):  DATA=/tmp/fake_events DEVICE=cpu bash scripts/w1024_smoke.sh
#   cluster (qrsh GPU):      bash scripts/w1024_smoke.sh          # defaults to /SAN data + cuda
#
# Success criteria (printed at the end): checkpoint exists, genuine JSON has
# accuracy/ppl, stylized-facts JSON has the F0 rate gate + context_mode=carried.
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:-/SAN/medic/TFOW/data/events/gmni_eth_7_v2_marks}"
DEVICE="${DEVICE:-cuda}"
MAXFILES="${MAXFILES:-1}"
EPOCHS="${EPOCHS:-1}"
SEQ="${SEQ:-1024}"; STRIDE="${STRIDE:-512}"
BATCH="${BATCH:-8}"
TARGET_RATE="${TARGET_RATE:-3.77}"
TAG="${TAG:-ss2p2}"
EXTRA="${EXTRA:---decoder-type ss2p2 --s2p2-layers 2 --ss2p2-wnorm-cap 6.0 --target-rate $TARGET_RATE}"
B="${B:-$REPO/experiments/w1024_smoke/$TAG}"

cd "$REPO"
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4
mkdir -p "$B/stylized_facts"
CKPT="$B/train/best_model.pt"
fail(){ echo "SMOKE_FAIL: $1"; exit 1; }

[ -n "$(ls "$DATA"/*.jsonl "$DATA"/*.jsonl.gz 2>/dev/null | head -1)" ] \
  || fail "no event files under $DATA (SAN not visible on this node?)"
echo "SMOKE data=$DATA device=$DEVICE seq=$SEQ files=$MAXFILES epochs=$EPOCHS"

echo "== 1/3 TRAIN (seq $SEQ, $EPOCHS epoch) =="
python3 -u -m volume_set_mtpp.training.train --data-dir "$DATA" --max-files "$MAXFILES" \
  --channel-emb-size 64 --time-emb-size 64 --recurrent-hidden 64 --device "$DEVICE" \
  --batch-size "$BATCH" --epochs "$EPOCHS" --lr 2e-3 --weight-decay 1e-6 \
  --seq-length "$SEQ" --stride "$STRIDE" --num-workers 0 --save-every "$EPOCHS" \
  --mark-head categorical --set-loss-reduction sum --no-volume-input-scaling --seed 1 \
  $EXTRA --output-dir "$B/train" --log-dir "$B/train/logs" 2>&1 | tail -12
[ -s "$CKPT" ] || fail "no checkpoint at $CKPT"

echo "== 2/3 PREDICTION VALIDATION (genuine_eval) =="
python3 -u -m volume_set_mtpp.evaluation.genuine_eval --checkpoint "$CKPT" \
  --data-dir "$DATA" --max-files "$MAXFILES" --seq-length "$SEQ" --stride "$STRIDE" \
  --batch-size "$BATCH" --device "$DEVICE" --label "$TAG" \
  --output "$B/genuine_${TAG}.json" 2>&1 | tail -6
[ -s "$B/genuine_${TAG}.json" ] || fail "no genuine_eval output"

echo "== 3/3 SIMULATION VALIDATION (stylized_facts, carried rollout) =="
python3 -u -m volume_set_mtpp.evaluation.stylized_facts --data-dir "$DATA" \
  --max-files "$MAXFILES" --checkpoint "$CKPT" --label "$TAG" \
  --output-dir "$B/stylized_facts" --device "$DEVICE" --sampler inversion \
  --context-mode carried --seq-length "$SEQ" --stride "$STRIDE" --batch-size "$BATCH" \
  --rollout-duration 60 --rollout-sequences 4 --rollout-seed 1 \
  --bucket-seconds 1.0 --max-real-windows 64 2>&1 | grep -E 'CONTEXT_MODE|SAMPLER|REAL events|SIM events|rate' | head -6
SF="$B/stylized_facts/stylized_facts_${TAG}.json"
[ -s "$SF" ] || fail "no stylized_facts output"

python3 - "$B/genuine_${TAG}.json" "$SF" <<'EOF'
import json, sys
g = json.load(open(sys.argv[1])); s = json.load(open(sys.argv[2]))
acc = g.get("genuine_mark_accuracy", g.get("acc")); ppl = g.get("genuine_mark_perplexity", g.get("ppl"))
rate = s["headline"]["F0 mean event rate (ev/s)"]
assert s.get("context_mode") == "carried", s.get("context_mode")
print(f"SMOKE_OK  acc={acc}  ppl={ppl}  sim_rate={rate['model']:.3f}  real_rate={rate['real']:.3f}  context_mode=carried")
EOF
