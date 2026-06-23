#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N cmp3way
#$ -l h_rt=8:00:00
#$ -l tmem=32G
#$ -l gpu=true
#$ -l gpu_type=h100
#$ -pe gpu 1
#$ -t 1-3
set -o pipefail

# Matched 3-way comparison on H100: LGM (proposed) vs LGM-SSP (S2P2 backbone +
# LGM heads, the bridge) vs S2P2 (baseline). Same data / seq / hidden / epochs so
# only the model differs. Runs train -> rho -> genuine_eval -> stylized_facts.
REPO="${REPO:-$HOME/simulation}"
DATA="${DATA:-/SAN/medic/TFOW/data/events/cbse_btc_7d}"
MAXFILES="${MAXFILES:-2}"
CACHE="${CACHE:-$DATA/.tensor_cache_sweep}"
TARGET_RATE="${TARGET_RATE:-40.6}"
EPOCHS="${EPOCHS:-40}"
SEQ="${SEQ:-64}"; STRIDE="${STRIDE:-32}"; HID="${HID:-128}"
ROOT="${ROOT:-$REPO/experiments/compare_3way}"

# TAG  DECODER  EXTRA
read -r -d '' GRID <<EOF
lgm     lgm     --decoder-type lgm --mark-head categorical --lgm-target-rate $TARGET_RATE --nmh-project-rho 0.92 --nmh-timescales 8 --ptp-dim 8 --lgm-vol-feedback
lgmssp  lgmssp  --decoder-type lgmssp --mark-head categorical --lgm-target-rate $TARGET_RATE --nmh-timescales 8 --s2p2-layers 2
s2p2    s2p2    --decoder-type s2p2 --mark-head categorical --s2p2-readout output --s2p2-layers 2
EOF
LINE=$(echo "$GRID" | sed -n "${SGE_TASK_ID}p")
TAG=$(echo "$LINE" | awk '{print $1}')
# drop columns 1 (TAG) and 2 (DECODER); the rest is the train.py flag string
EXTRA=$(echo "$LINE" | awk '{$1="";$2="";sub(/^ +/,"");print}')
[ -z "$TAG" ] && { echo "no config for task $SGE_TASK_ID"; exit 1; }

cd "$REPO"
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4
B="$ROOT/$TAG"; mkdir -p "$B/stylized_facts"; ML="$B/master.log"
log(){ echo "$@" | tee -a "$ML"; }
log "START $(date) TAG=$TAG seq=$SEQ host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null|head -1|tee -a "$ML"
CKPT="$B/train/best_model.pt"
log "TRAIN $(date)"
python3 -u -m volume_set_mtpp.training.train --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" $EXTRA \
  --channel-emb-size 64 --time-emb-size "$HID" --recurrent-hidden "$HID" \
  --batch-size 256 --epochs "$EPOCHS" --lr 2e-3 --weight-decay 1e-6 \
  --seq-length "$SEQ" --stride "$STRIDE" --num-workers 0 --save-every "$EPOCHS" \
  --set-loss-reduction sum --no-volume-input-scaling --allow-tf32 --seed 1 \
  --output-dir "$B/train" --log-dir "$B/train/logs" > "$B/train.log" 2>&1
log "TRAIN_RC=$?"
[ -s "$CKPT" ] || { log "NO_CKPT"; tail -20 "$B/train.log" | tee -a "$ML"; exit 1; }
python3 -u - "$CKPT" <<'PY' 2>&1 | tee -a "$ML"
import sys, torch
from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp
ck=torch.load(sys.argv[1],map_location="cpu",weights_only=False); cfg=ck["config"]
m=create_volume_set_mtpp(cfg.get("num_channels",62),cfg,torch.device("cpu"),use_volume=cfg.get("use_volume",False))
m.load_state_dict(ck["model_state_dict"])
print("RHO closed_form_rho=%.4f"%m.decoder.closed_form_rho()) if hasattr(m.decoder,"closed_form_rho") else print("RHO n/a")
PY
log "GENUINE $(date)"
python3 -u -m volume_set_mtpp.evaluation.genuine_eval --checkpoint "$CKPT" --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" \
  --seq-length "$SEQ" --stride "$STRIDE" --batch-size 256 --device cuda --label "$TAG" --output "$B/genuine_${TAG}.json" 2>&1 | tee -a "$ML"
log "SF $(date)"
python3 -u -m volume_set_mtpp.evaluation.stylized_facts --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" \
  --checkpoint "$CKPT" --label "$TAG" --output-dir "$B/stylized_facts" --device cuda \
  --seq-length "$SEQ" --stride "$STRIDE" --batch-size 256 --rollout-duration 600 --rollout-sequences 32 \
  --rollout-seed 1 --bucket-seconds 1.0 --max-real-windows 4096 > "$B/sf.log" 2>&1
log "DONE $(date) RC=$?"
