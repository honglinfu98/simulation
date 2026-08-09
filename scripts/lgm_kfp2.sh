#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N lgm_kfp2
#$ -l h_rt=6:00:00
#$ -l tmem=32G
#$ -l gpu=true
#$ -l h=!hoots-207-1*
#$ -pe gpu 1
#$ -t 1-3
set -o pipefail
REPO="$HOME/simulation"
COINS=(btc btc eth); NS=(995 9975 995)
COIN=${COINS[$((SGE_TASK_ID-1))]}; NT=${NS[$((SGE_TASK_ID-1))]}
DATA=/SAN/medic/TFOW/data/events/cbse_${COIN}_7d
TAG="kfp-${COIN}-${NT}"; B="$REPO/experiments/ma_cbse/$COIN/$TAG"; CKPT="$B/train/best_model.pt"
cd "$REPO"
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4
[ -s "$CKPT" ] || { echo "missing $CKPT"; exit 1; }
rm -rf "$B/sf_r1"; mkdir -p "$B/sf_r1"
python3 -u -m volume_set_mtpp.evaluation.stylized_facts --data-dir "$DATA" --max-files 7 --cache-dir "$DATA/.tensor_cache_eval" \
  --checkpoint "$CKPT" --label "$TAG" --output-dir "$B/sf_r1" --device cuda --sampler inversion \
  --context-mode carried --calibrate-rate -1 --calibrate-split val --calibrate-probe-duration 600 --calibrate-final-tol 0.15 \
  --match-durations --seq-length 4096 --stride 4096 --batch-size 256 --rollout-duration 600 --rollout-sequences 32 \
  --rollout-seed 1 --bucket-seconds 1.0 --max-real-windows 4096 > "$B/sf.log" 2>&1 || { tail -5 "$B/sf.log"; exit 1; }
echo "DONE $TAG"
