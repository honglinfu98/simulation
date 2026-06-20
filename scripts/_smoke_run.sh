#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N gmni_TAG
#$ -l h_rt=00:05:00
#$ -l tmem=1G
# NO gpu line -> dispatches on any node. A throwaway job that writes the SAME
# marker grammar as a real run (TRAIN_START/END, RHO, DONE STATUS=0 BASE=...) plus
# fake genuine/stylized-facts JSON, so submit_run -> watch_runs -> email can be
# tested end-to-end in ~1 minute without a GPU, data, or a real model.
set -o pipefail
cd "$HOME/volume-set-mtpp"

TAG=TAG
SEED=1
BASE="experiments/gmni_marks_${TAG}_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$BASE"; M="$BASE/master.log"
log(){ echo "$@" | tee -a "$M"; }
log "START $(date) HOST=$(hostname)"
log "TRAIN_START $(date)"; sleep 30; log "TRAIN_END $(date) $TAG RC=0"
log "RHO closed_form_rho=0.4242"
printf '{"label":"%s","n_genuine_events":100,"genuine_mark_accuracy":0.42,"genuine_mark_perplexity":6.80,"mark_head":"categorical"}' "$TAG" > "$BASE/genuine_${TAG}.json"
mkdir -p "$BASE/stylized_facts"
printf '{"headline":{"F5 Fano at scales":{"model":[12.4,20.1],"real":[10.0,18.0]}}}' > "$BASE/stylized_facts/stylized_facts_${TAG}.json"
log "DONE $(date) STATUS=0 BASE=$BASE"
echo "$BASE" > "$HOME/volume-set-mtpp/.last_${TAG}_base"
