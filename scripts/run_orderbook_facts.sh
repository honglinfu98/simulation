#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N ob_facts
#$ -l h_rt=6:00:00
#$ -l tmem=32G
#$ -l gpu=true
#$ -l gpu_type=a40
#$ -pe gpu 1
set -o pipefail

# Order-book stylized facts (Jain Ch.4 set) for all 7 eval_all checkpoints.
# Eval-only: reuses experiments/eval_all/<tag>/train/best_model.pt from
# run_eval_all.sh. Rerun:  ssh peacock && cd ~/simulation && qsub scripts/run_orderbook_facts.sh
# (or bash scripts/run_orderbook_facts.sh on a GPU node)
REPO="${REPO:-$HOME/simulation}"
DATA="${DATA:-/SAN/medic/TFOW/data/events/gmni_eth_7_v2_marks}"
CACHE="${CACHE:-$DATA/.tensor_cache_eval}"
ROOT="${ROOT:-$REPO/experiments/eval_all}"
DUR="${DUR:-600}"; NSEQ="${NSEQ:-32}"

cd "$REPO"
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4

for TAG in hawkes lstm sahp ct-lstm pct-lstm s2p2 ss2p2; do
  CKPT="$ROOT/$TAG/train/best_model.pt"
  [ -s "$CKPT" ] || { echo "SKIP $TAG (no checkpoint)"; continue; }
  echo "=== $TAG $(date) ==="
  python3 -u -m volume_set_mtpp.evaluation.orderbook_facts \
    --checkpoint "$CKPT" --data-dir "$DATA" --cache-dir "$CACHE" --max-files 7 \
    --seq-length 64 --stride 32 --batch-size 256 --device cuda \
    --real-files 2 --real-max-events 150000 \
    --rollout-duration "$DUR" --rollout-sequences "$NSEQ" --rollout-seed 1 \
    --label "$TAG" --output-dir "$ROOT/$TAG/orderbook_facts" \
    > "$ROOT/$TAG/orderbook_facts.log" 2>&1
  echo "RC=$? $TAG"
done

python3 scripts/orderbook_collect.py "$ROOT" | tee "$ROOT/OB_REPORT.txt"
echo "DONE $(date)"
