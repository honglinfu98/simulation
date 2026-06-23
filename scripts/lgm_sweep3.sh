#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N lgm_sweep3
#$ -l h_rt=10:00:00
#$ -l tmem=32G
#$ -l gpu=true
#$ -l gpu_type=h100
#$ -pe gpu 1
#$ -t 1-2
set -o pipefail

# LGM hyperparameter sweep: one SGE array task per config. Each task runs the
# full pipeline (train -> closed_form_rho -> genuine_eval -> stylized_facts) and
# writes its metrics JSONs under $SWEEPROOT/<TAG>/.  Collected afterwards by
# scripts/lgm_sweep_collect.py.
#
# Dataset: cbse_btc_7d, days 1-2 (~6.6M events) for the search; the winning
# config is retrained on all 7 days separately.  Rate-pin = 7-day mean (40.6/s).

REPO="${REPO:-$HOME/simulation}"
DATA="${DATA:-/SAN/medic/TFOW/data/events/cbse_btc_7d}"
MAXFILES="${MAXFILES:-2}"
CACHE="${CACHE:-$DATA/.tensor_cache_sweep}"
TARGET_RATE="${TARGET_RATE:-40.6}"
EPOCHS="${EPOCHS:-40}"
SWEEPROOT="${SWEEPROOT:-$REPO/experiments/lgm_sweep}"

# Config grid: TAG SEQ STRIDE RHO M HID VOLFB
#   SEQ/STRIDE = window size & cold-start lever ; RHO = branching cap (sim stability)
#   M = nmh-timescales ; HID = recurrent+time emb size ; VOLFB = lgm-vol-feedback (tails)
read -r -d '' GRID <<'EOF'
L64_r92_M8_vfb    64  32  0.92 8 128 1
L64_r86_M8_vfb    64  32  0.86 8 128 1
EOF

LINE=$(echo "$GRID" | sed -n "${SGE_TASK_ID}p")
read -r TAG SEQ STRIDE RHO M HID VFB <<<"$LINE"
[ -z "$TAG" ] && { echo "NO CONFIG for task $SGE_TASK_ID"; exit 1; }

cd "$REPO"
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BASE="$SWEEPROOT/$TAG"; mkdir -p "$BASE/stylized_facts"; M_LOG="$BASE/master.log"
log(){ echo "$@" | tee -a "$M_LOG"; }
log "START $(date) TAG=$TAG SEQ=$SEQ STRIDE=$STRIDE RHO=$RHO M=$M HID=$HID VFB=$VFB host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tee -a "$M_LOG"
[ -z "$(ls "$DATA"/*.jsonl.gz 2>/dev/null | head -1)" ] && { log "SAN_NOT_VISIBLE"; exit 1; }

VFB_FLAG=""; [ "$VFB" = "1" ] && VFB_FLAG="--lgm-vol-feedback"
CKPT="$BASE/train/best_model.pt"

log "TRAIN_START $(date)"
python3 -u -m volume_set_mtpp.training.train \
  --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" \
  --decoder-type lgm --mark-head categorical \
  --lgm-target-rate "$TARGET_RATE" --nmh-project-rho "$RHO" --nmh-timescales "$M" --ptp-dim 8 $VFB_FLAG \
  --channel-emb-size 64 --time-emb-size "$HID" --recurrent-hidden "$HID" \
  --batch-size 256 --epochs "$EPOCHS" --lr 2e-3 --weight-decay 1e-6 \
  --seq-length "$SEQ" --stride "$STRIDE" --num-workers 0 --save-every "$EPOCHS" \
  --set-loss-reduction sum --no-volume-input-scaling --allow-tf32 --seed 1 \
  --output-dir "$BASE/train" --log-dir "$BASE/train/logs" > "$BASE/train.log" 2>&1
log "TRAIN_END $(date) RC=$?"
[ -s "$CKPT" ] || { log "NO_CKPT (train failed) — see train.log tail:"; tail -20 "$BASE/train.log" | tee -a "$M_LOG"; exit 1; }

log "RHO $(date)"
python3 -u - "$CKPT" <<'PY' 2>&1 | tee -a "$M_LOG"
import sys, torch
from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp
ck=torch.load(sys.argv[1],map_location="cpu",weights_only=False); cfg=ck["config"]
m=create_volume_set_mtpp(cfg.get("num_channels",62),cfg,torch.device("cpu"),use_volume=cfg.get("use_volume",False))
m.load_state_dict(ck["model_state_dict"])
print("RHO closed_form_rho=%.4f" % m.decoder.closed_form_rho()) if hasattr(m.decoder,"closed_form_rho") else print("RHO n/a")
PY

log "GENUINE $(date)"
python3 -u -m volume_set_mtpp.evaluation.genuine_eval --checkpoint "$CKPT" \
  --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" \
  --seq-length "$SEQ" --stride "$STRIDE" --batch-size 256 --device cuda \
  --label "$TAG" --output "$BASE/genuine_${TAG}.json" 2>&1 | tee -a "$M_LOG"

log "SF $(date)"
python3 -u -m volume_set_mtpp.evaluation.stylized_facts \
  --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" \
  --checkpoint "$CKPT" --label "$TAG" --output-dir "$BASE/stylized_facts" --device cuda \
  --seq-length "$SEQ" --stride "$STRIDE" --batch-size 256 \
  --rollout-duration 600 --rollout-sequences 32 --rollout-seed 1 \
  --bucket-seconds 1.0 --max-real-windows 4096 > "$BASE/sf.log" 2>&1
log "SF_END $(date) RC=$?"
log "DONE $(date) BASE=$BASE"
