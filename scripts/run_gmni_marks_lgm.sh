#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N gmni_lgm
#$ -l h_rt=8:00:00
#$ -l tmem=24G
#$ -l gpu=true
#$ -pe gpu 1
set -o pipefail

# NEURAL MULTIVARIATE HAWKES (NMH) on event-driven Gemini ETH.
# Multi-timescale multivariate Hawkes: per-type decayed counts at M timescales,
# linear cross-excitation read-out A, softplus link -> per-type intensities.
# Ground intensity = sum_k lambda_k; mark dist = lambda_k/sum (categorical head
# falls out).  Honest gauge-free branching ratio rho = spectral_radius(A/delta).
# Trained by exact multivariate-Hawkes MLE on the windowed loader (same data,
# loader, seq/stride as the s2p2 and baseline arms -> directly comparable).

cd "$HOME/volume-set-mtpp"
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source venv/bin/activate 2>/dev/null || true
export PYTHONPATH="$PWD/src"
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset BFNX_CACHE_FILE

DATA=/SAN/medic/TFOW/data/events/gmni_eth_7_v2_marks
CACHE=$DATA/.tensor_cache_seq50_stride32
SEED=1
TAG=lgm
BASE="experiments/gmni_marks_lgm_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BASE"
MASTER="$BASE/master.log"
log() { echo "$@" | tee -a "$MASTER"; }
log "RUN_KIND=gmni_marks_lgm START $(date) HOST=$(hostname) CVD=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tee -a "$MASTER"
[ -z "$(ls "$DATA"/*.jsonl.gz 2>/dev/null | head -1)" ] && { log "SAN_NOT_VISIBLE $(hostname)"; exit 1; }

CKPT="$BASE/${TAG}_train/best_model.pt"
status=0

log "TRAIN_START $(date) $TAG"
/usr/bin/time -p python3 -u -m volume_set_mtpp.training.train \
  --data-dir "$DATA" --max-files 7 --cache-dir "$CACHE" \
  --decoder-type lgm --nmh-timescales 4 --ptp-dim 8 --lgm-target-rate 2.381 \
  --channel-emb-size 64 --time-emb-size 128 --recurrent-hidden 128 \
  --batch-size 512 --epochs 40 --lr 2e-3 --weight-decay 1e-6 \
  --seq-length 50 --stride 32 --num-workers 0 --save-every 40 \
  --set-loss-reduction sum --set-loss-weight 1.0 --time-loss-weight 1.0 \
  --no-volume-input-scaling --mark-head categorical --nmh-project-rho 0.8 \
  --allow-tf32 --seed "$SEED" \
  --output-dir "$BASE/${TAG}_train" --log-dir "$BASE/${TAG}_train/logs" > "$BASE/${TAG}.train.log" 2>&1
rc=$?; log "TRAIN_END $(date) $TAG RC=$rc"
[ $rc -ne 0 ] && { status=1; tail -40 "$BASE/${TAG}.train.log" >> "$MASTER"; log "DONE STATUS=$status"; exit 1; }

log "RHO_START $(date)"
python3 -u - "$CKPT" <<'PY' 2>&1 | tee -a "$MASTER"
import sys, json, torch
from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
cfg = ck["config"]
m = create_volume_set_mtpp(cfg.get("num_channels", 62), cfg, torch.device("cpu"),
                           use_volume=cfg.get("use_volume", False), intensity_type="dynamic")
m.load_state_dict(ck["model_state_dict"])
rho = m.decoder.closed_form_rho()
deltas = m.decoder._deltas().tolist()
print(f"NMH_RHO closed_form_rho={rho:.4f} deltas={[round(d,3) for d in deltas]}")
PY
log "RHO_END $(date)"

log "GENUINE_EVAL_START $(date)"
python3 -u -m volume_set_mtpp.evaluation.tfow_genuine_eval \
  --checkpoint "$CKPT" --data-dir "$DATA" --max-files 7 --cache-dir "$CACHE" \
  --seq-length 50 --stride 32 --batch-size 512 --device cuda \
  --label nmh --output "$BASE/genuine_nmh.json" > "$BASE/${TAG}.genuine.log" 2>&1
rc=$?; log "GENUINE_EVAL_END $(date) RC=$rc"; cat "$BASE/genuine_nmh.json" 2>/dev/null | tee -a "$MASTER"

log "SF_START $(date) $TAG"
python3 -u -m volume_set_mtpp.evaluation.tfow_stylized_facts \
  --data-dir "$DATA" --max-files 7 --cache-dir "$CACHE" \
  --checkpoint "$CKPT" --label "$TAG" --output-dir "$BASE/stylized_facts" --device cuda \
  --seq-length 50 --stride 32 --batch-size 512 \
  --rollout-duration 600 --rollout-sequences 32 --rollout-seed "$SEED" \
  --bucket-seconds 1.0 --max-real-windows 4096 > "$BASE/${TAG}.sf.log" 2>&1
log "SF_END $(date) $TAG RC=$?"

log "PV2_START $(date) $TAG"
python3 -u -m volume_set_mtpp.evaluation.tfow_price_facts_v2 \
  --v2-dir "$DATA" --pattern "events_gmni_ethusdt_*.jsonl.gz" \
  --checkpoint "$CKPT" --label "$TAG" --output-dir "$BASE/price_v2" --device cuda \
  --max-events-per-file 150000 \
  --rollout-duration 600 --rollout-sequences 32 --rollout-seed "$SEED" > "$BASE/${TAG}.pv2.log" 2>&1
log "PV2_END $(date) $TAG RC=$?"

log "DONE $(date) STATUS=$status BASE=$BASE"
echo "$BASE" > "$HOME/volume-set-mtpp/.last_lgm_base"
exit $status
