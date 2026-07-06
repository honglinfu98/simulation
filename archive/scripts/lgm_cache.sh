#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N lgm_cache
#$ -l h_rt=4:00:00
#$ -l tmem=48G
#$ -l h_vmem=48G
# Pre-build the tensorized data cache ONCE so the GPU sweep array all reuse it
# (avoids 12 concurrent multi-GB JSONL parses racing on the same cache file).
set -o pipefail
REPO="${REPO:-$HOME/simulation}"
DATA="${DATA:-/SAN/medic/TFOW/data/events/cbse_btc_7d}"
MAXFILES="${MAXFILES:-2}"
CACHE="${CACHE:-$DATA/.tensor_cache_sweep}"
cd "$REPO"
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4
echo "CACHE_BUILD_START $(date) data=$DATA maxfiles=$MAXFILES cache=$CACHE host=$(hostname)"
python3 -u - "$DATA" "$MAXFILES" "$CACHE" <<'PY'
import sys
from volume_set_mtpp.training.data_loader import load_bfnx_tensors
data, mf, cache = sys.argv[1], int(sys.argv[2]), sys.argv[3]
td, marks, vols, st, lf, em, fl = load_bfnx_tensors(data_dir=data, max_files=mf, cache_dir=cache, rebuild_cache=True)
print("CACHE_OK events=%d channels=%d segments=%d" % (td.shape[0], em.num_events, len(fl)))
PY
echo "CACHE_BUILD_END $(date) RC=$?"
