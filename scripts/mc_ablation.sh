#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N mc_ablation
#$ -l h_rt=16:00:00
#$ -l tmem=32G
#$ -l gpu=true
#$ -l h=!hoots-207-1*
#$ -pe gpu 1
#$ -t 1-4
set -o pipefail

# MC-COMPENSATOR ABLATION (single factor, paired arms).
#
# Hypothesis: the endpoint-rule compensator underestimates int lambda dt on
# self-exciting gaps, under-charging intensity mass during training -> the
# universal free-run rate inflation (mean_u 1.6-2.9, sim_rate 7-15x real).
# Treatment: --mc-compensator (Mei-Eisner unbiased MC estimate, 32 samples/gap).
#
# Design: 2 models x 2 arms, ONLY the compensator flag differs within a pair.
# Controls are retrained in the same array (same code/cache/nodes pool) rather
# than reusing experiments/ss2p2_w1024, so the comparison is contemporaneous.
# Everything else = the w1024 regime: seq 1024 / stride 512 / batch 64 /
# 40 epochs / seed 1; eval = genuine_eval + stylized_facts --context-mode
# carried (600s x 32 seq). Broken GPU node hoots-207-1 excluded.
#
# Primary endpoints:
#   mean_u -> 1          (genuine_eval; endpoint-rule arms sit at 2.3-2.9)
#   sim_rate -> real     (stylized_facts F0 rate gate)
# Secondary: overall NLL should NOT degrade; Fano/clustering shifts.
REPO="${REPO:-$HOME/simulation}"
DATA="${DATA:-/SAN/medic/TFOW/data/events/gmni_eth_7_v2_marks}"
MAXFILES="${MAXFILES:-7}"
CACHE="${CACHE:-$DATA/.tensor_cache_eval}"
TARGET_RATE="${TARGET_RATE:-3.77}"
EPOCHS="${EPOCHS:-40}"
SEQ="${SEQ:-1024}"; STRIDE="${STRIDE:-512}"
ROOT="${ROOT:-$REPO/experiments/mc_ablation}"

SS2P2="--decoder-type ss2p2 --s2p2-layers 2 --ss2p2-wnorm-cap 6.0 --target-rate $TARGET_RATE"
S2P2="--decoder-type s2p2 --s2p2-layers 2 --s2p2-readout output"
case "$SGE_TASK_ID" in
  1) TAG=ss2p2-ep; EXTRA="$SS2P2" ;;                          # control: endpoint rule
  2) TAG=ss2p2-mc; EXTRA="$SS2P2 --mc-compensator" ;;         # treatment
  3) TAG=s2p2-ep;  EXTRA="$S2P2" ;;                           # control: endpoint rule
  4) TAG=s2p2-mc;  EXTRA="$S2P2 --mc-compensator" ;;          # treatment
  *) echo "no config for task $SGE_TASK_ID"; exit 1 ;;
esac

cd "$REPO"
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4
B="$ROOT/$TAG"
# Fresh outputs every run: never reuse a stale checkpoint or old eval JSONs.
rm -rf "$B"
mkdir -p "$B/stylized_facts"; ML="$B/master.log"
log(){ echo "$@" | tee -a "$ML"; }
fail(){ log "DONE $(date) STATUS=1 stage=$1 rc=$2 BASE=$B"; exit 1; }
log "START $(date) TAG=$TAG host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tee -a "$ML"
CKPT="$B/train/best_model.pt"

log "TRAIN $(date)"
python3 -u -m volume_set_mtpp.training.train --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" \
  --channel-emb-size 64 --time-emb-size 64 --recurrent-hidden 64 \
  --batch-size 64 --epochs "$EPOCHS" --lr 2e-3 --weight-decay 1e-6 \
  --seq-length "$SEQ" --stride "$STRIDE" --num-workers 0 --save-every "$EPOCHS" \
  --mark-head categorical --set-loss-reduction sum --no-volume-input-scaling --allow-tf32 --seed 1 \
  $EXTRA --output-dir "$B/train" --log-dir "$B/train/logs" > "$B/train.log" 2>&1
TRAIN_RC=$?
log "TRAIN_RC=$TRAIN_RC"
{ [ "$TRAIN_RC" -eq 0 ] && [ -s "$CKPT" ]; } || { tail -25 "$B/train.log" | tee -a "$ML"; fail train "$TRAIN_RC"; }

log "GENUINE $(date)"
python3 -u -m volume_set_mtpp.evaluation.genuine_eval --checkpoint "$CKPT" --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" \
  --seq-length "$SEQ" --stride "$STRIDE" --batch-size 256 --device cuda --label "$TAG" \
  --dt-horizon 60 --dt-grid-points 128 --output "$B/genuine_${TAG}.json" 2>&1 | tee -a "$ML"
GEN_RC=$?
{ [ "$GEN_RC" -eq 0 ] && [ -s "$B/genuine_${TAG}.json" ]; } || fail genuine "$GEN_RC"

log "SF $(date)"
python3 -u -m volume_set_mtpp.evaluation.stylized_facts --data-dir "$DATA" --max-files "$MAXFILES" --cache-dir "$CACHE" \
  --checkpoint "$CKPT" --label "$TAG" --output-dir "$B/stylized_facts" --device cuda --sampler inversion \
  --context-mode carried \
  --seq-length "$SEQ" --stride "$STRIDE" --batch-size 256 --rollout-duration 600 --rollout-sequences 32 \
  --rollout-seed 1 --bucket-seconds 1.0 --max-real-windows 4096 > "$B/sf.log" 2>&1
SF_RC=$?
{ [ "$SF_RC" -eq 0 ] && [ -s "$B/stylized_facts/stylized_facts_${TAG}.json" ]; } \
  || { tail -25 "$B/sf.log" | tee -a "$ML"; fail sf "$SF_RC"; }
log "DONE $(date) STATUS=0 BASE=$B"
