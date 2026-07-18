#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N realism
#$ -l h_rt=6:00:00
#$ -l tmem=32G
#$ -l gpu=true
#$ -l h=!hoots-207-1*
#$ -pe gpu 1
#$ -t 1-54
set -o pipefail

# UNCONDITIONAL MARKET-REALISM suite over every calibrated rollout of the
# Coinbase benchmark (ma_cbse).  For each (coin, model, seed) arm and each
# rollout seed r in {1,2,3}:
#   - read the BANKED calibration constant k from sf_r<r>/stylized_facts json
#     (probes were seeded, so --fixed-k + the same rollout seed reproduces the
#     exact calibrated rollout without re-running bisection);
#   - re-run the stylized-facts harness with --fixed-k k --realism, which
#     recomputes the same facts AND writes realism_<tag>.json alongside.
# Arms with no stylized-facts artifact (calibration-excluded checkpoints)
# are skipped; SAHP runs with its banked k=1 (uncalibrated by protocol).
#
# Task map: t = 1..54; coin = (t-1)/18; within: model=(t-1)%6, seed=((t-1)%18)/6+1
REPO="${REPO:-$HOME/simulation}"
COINS=(btc eth sol)
CI=$(( (SGE_TASK_ID - 1) / 18 ))
COIN="${COINS[$CI]}"
DATA="${DATA:-/SAN/medic/TFOW/data/events/cbse_${COIN}_7d}"
CACHE="${CACHE:-$DATA/.tensor_cache_eval}"
MAXFILES=7
SEQ=1024; STRIDE=1024
ROOT="$REPO/experiments/ma_cbse/$COIN"

MODELS=(nhp lstm sahp pct-lstm s2p2 ss2p2-full)
# SS2P2 samples by Ogata thinning with its EXACT closed-form ceiling
# (rate_bounds); baselines have no such bound and stay on inversion.
MI=$(( (SGE_TASK_ID - 1) % 6 ))
SEED=$(( ((SGE_TASK_ID - 1) % 18) / 6 + 1 ))
MODEL="${MODELS[$MI]}"
TAG="${MODEL}-s${SEED}"
SAMPLER=inversion
case "$MODEL" in ss2p2*) SAMPLER=thinning ;; esac
B="$ROOT/$TAG"
CKPT="$B/train/best_model.pt"

cd "$REPO"
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4

[ -s "$CKPT" ] || { echo "SKIP $COIN/$TAG: no checkpoint"; echo "REALISM_DONE STATUS=0 SKIP"; exit 0; }

for R in 1 2 3; do
  SF="$B/sf_r$R/stylized_facts_${TAG}.json"
  if [ ! -s "$SF" ]; then
    echo "SKIP $COIN/$TAG r$R: no calibrated stylized-facts artifact"
    continue
  fi
  K=$(python3 -c "import json,sys; print(json.load(open('$SF'))['rate_scale_k'])")
  echo "REALISM $COIN/$TAG r$R fixed-k=$K"
  python3 -u -m volume_set_mtpp.evaluation.stylized_facts --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" \
    --checkpoint "$CKPT" --label "$TAG" --output-dir "$B/sf_r$R" --device cuda --sampler "$SAMPLER" \
    --context-mode carried --fixed-k "$K" --realism --match-durations \
    --seq-length "$SEQ" --stride "$STRIDE" --batch-size 256 --rollout-duration 600 --rollout-sequences 32 \
    --rollout-seed "$R" --bucket-seconds 1.0 --max-real-windows 4096 > "$B/sf_r$R/realism_run.log" 2>&1
  RC=$?
  if [ "$RC" -ne 0 ] || [ ! -s "$B/sf_r$R/realism_${TAG}.json" ]; then
    tail -20 "$B/sf_r$R/realism_run.log"
    echo "REALISM_DONE STATUS=1 $COIN/$TAG r$R rc=$RC"
    exit 1
  fi
done
echo "REALISM_DONE STATUS=0 $COIN/$TAG"
