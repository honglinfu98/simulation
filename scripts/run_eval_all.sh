#!/usr/bin/env bash
# SINGLE ENTRY POINT for the full evaluation. Rerun this to repeat everything.
#
#   ssh peacock
#   cd ~/simulation && bash scripts/run_eval_all.sh
#
# Submits a 7-task GPU array (Hawkes, LSTM, SAHP, CT-LSTM, PCT-LSTM, S2P2, SS2P2),
# each running train -> genuine_eval (prediction metrics) -> stylized_facts (fit
# set), then a dependent collector job that prints the comparison tables.
# Best GPU: requests gpu_type=h100. Override knobs via env, e.g.:
#   EPOCHS=40 MAXFILES=7 DATA=/SAN/medic/TFOW/data/events/gmni_eth_7_v2_marks bash scripts/run_eval_all.sh
set -euo pipefail
REPO="${REPO:-$HOME/simulation}"
cd "$REPO"
mkdir -p logs
cd logs

# 1) model array (train + eval + stylized facts), best GPU
AID=$(qsub -terse "$REPO/scripts/eval_worker.sh" | head -1 | cut -d. -f1)
echo "submitted eval_worker array job: $AID (tasks 1-7)"

# 2) collector, held until the whole array finishes; prints the report tables
cat > "$REPO/logs/_eval_collect_job.sh" <<EOF
#!/usr/bin/env bash
#\$ -S /bin/bash
#\$ -cwd
#\$ -j y
#\$ -N eval_collect
#\$ -l h_rt=0:20:00
#\$ -l tmem=8G
cd "$REPO"
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "\$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO"
python3 scripts/eval_collect.py "$REPO/experiments/eval_all" | tee "$REPO/experiments/eval_all/REPORT.txt"
EOF
CID=$(qsub -terse -hold_jid "$AID" "$REPO/logs/_eval_collect_job.sh" | head -1 | cut -d. -f1)
echo "submitted eval_collect job: $CID (held on $AID)"
echo
echo "watch:   qstat -u \$USER"
echo "report:  cat $REPO/experiments/eval_all/REPORT.txt"
