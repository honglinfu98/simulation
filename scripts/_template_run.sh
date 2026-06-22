#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N gmni_TAG          # <- rename per model
#$ -l h_rt=8:00:00
#$ -l tmem=24G
#$ -l gpu=true          # NOTE: gpu_type=h100/a100_80 are gated for this account; plain gpu=true dispatches
#$ -pe gpu 1
set -o pipefail

# Template run: train -> rho report -> genuine-event eval -> stylized facts -> price facts.
# Copy this, set TAG/DECODER and the decoder-specific flags, deploy models/ to the
# volume-set-mtpp framework, then qsub.  (See docs/RUNBOOK.md.)

cd "$HOME/volume-set-mtpp"
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source venv/bin/activate 2>/dev/null || true
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA=/SAN/medic/TFOW/data/events/gmni_eth_7_v2_marks
CACHE=$DATA/.tensor_cache_seq50_stride32
SEED=1
TAG=TAG                 # <- e.g. lgm086
DECODER=DECODER         # <- lgm (the model) | pts2p2 | s2p2 | hawkes | rmtpp (baselines)
EXTRA="--mark-head categorical"   # <- decoder-specific flags, e.g.:
# lgm:  --decoder-type lgm --nmh-timescales 4 --ptp-dim 8 --lgm-target-rate 2.381 --nmh-project-rho 0.86 --mark-head categorical
# s2p2: --decoder-type s2p2 --s2p2-readout output --mark-head categorical

BASE="experiments/gmni_marks_${TAG}_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$BASE"; M="$BASE/master.log"
log(){ echo "$@" | tee -a "$M"; }
log "START $(date) HOST=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tee -a "$M"
[ -z "$(ls "$DATA"/*.jsonl.gz 2>/dev/null | head -1)" ] && { log "SAN_NOT_VISIBLE"; exit 1; }
CKPT="$BASE/${TAG}_train/best_model.pt"

log "TRAIN_START $(date)"
python3 -u -m volume_set_mtpp.training.train \
  --data-dir "$DATA" --max-files 7 --cache-dir "$CACHE" $EXTRA \
  --channel-emb-size 64 --time-emb-size 128 --recurrent-hidden 128 \
  --batch-size 512 --epochs 40 --lr 2e-3 --weight-decay 1e-6 \
  --seq-length 50 --stride 32 --num-workers 0 --save-every 40 \
  --set-loss-reduction sum --no-volume-input-scaling --allow-tf32 --seed "$SEED" \
  --output-dir "$BASE/${TAG}_train" --log-dir "$BASE/${TAG}_train/logs" > "$BASE/${TAG}.train.log" 2>&1
log "TRAIN_END $(date) RC=$?"

log "RHO $(date)"
python3 -u - "$CKPT" <<'PY' 2>&1 | tee -a "$M"
import sys, torch
from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp
ck=torch.load(sys.argv[1],map_location="cpu",weights_only=False); cfg=ck["config"]
m=create_volume_set_mtpp(cfg.get("num_channels",62),cfg,torch.device("cpu"),use_volume=cfg.get("use_volume",False))
m.load_state_dict(ck["model_state_dict"])
print("RHO closed_form_rho=%.4f" % m.decoder.closed_form_rho()) if hasattr(m.decoder,"closed_form_rho") else print("RHO n/a")
PY

log "GENUINE $(date)"
python3 -u -m volume_set_mtpp.evaluation.genuine_eval --checkpoint "$CKPT" --data-dir "$DATA" --max-files 7 --cache-dir "$CACHE" \
  --seq-length 50 --stride 32 --batch-size 512 --device cuda --label "$TAG" --output "$BASE/genuine_${TAG}.json" 2>&1 | tee -a "$M"

log "SF $(date)"
python3 -u -m volume_set_mtpp.evaluation.stylized_facts --data-dir "$DATA" --max-files 7 --cache-dir "$CACHE" \
  --checkpoint "$CKPT" --label "$TAG" --output-dir "$BASE/stylized_facts" --device cuda \
  --seq-length 50 --stride 32 --batch-size 512 --rollout-duration 600 --rollout-sequences 32 \
  --rollout-seed "$SEED" --bucket-seconds 1.0 --max-real-windows 4096 > "$BASE/${TAG}.sf.log" 2>&1
log "DONE $(date) BASE=$BASE"
