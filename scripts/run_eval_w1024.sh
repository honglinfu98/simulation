#!/usr/bin/env bash
# SINGLE ENTRY POINT for the long-context experiment: seq-1024 training +
# carried-state free rollout. Rerun this to repeat everything.
#
#   ssh peacock
#   cd ~/simulation && bash scripts/run_eval_w1024.sh
#
# Submits the 7-task GPU array (eval_worker_w1024.sh: train at seq 1024 /
# stride 512 / batch 64 -> genuine_eval -> stylized_facts with
# --context-mode carried), then a dependent collector job that prints the
# comparison tables. Results land in experiments/ss2p2_w1024/.
# Compare against the seq-64/window-mode benchmark in experiments/eval_all/ --
# the delta isolates (training context length) + (rollout memory truncation).
set -euo pipefail
REPO="${REPO:-$HOME/simulation}"
cd "$REPO"
mkdir -p logs
cd logs

AID=$(qsub -terse "$REPO/scripts/eval_worker_w1024.sh" | head -1 | cut -d. -f1)
echo "submitted eval_worker_w1024 array job: $AID (tasks 1-7)"

cat > "$REPO/logs/_eval_w1024_collect_job.sh" <<EOF
#!/usr/bin/env bash
#\$ -S /bin/bash
#\$ -cwd
#\$ -j y
#\$ -N eval_w1024_collect
#\$ -l h_rt=0:20:00
#\$ -l tmem=8G
cd "$REPO"
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "\$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO"
python3 scripts/eval_collect.py "$REPO/experiments/ss2p2_w1024" | tee "$REPO/experiments/ss2p2_w1024/REPORT.txt"
EOF
CID=$(qsub -terse -hold_jid "$AID" "$REPO/logs/_eval_w1024_collect_job.sh" | head -1 | cut -d. -f1)
echo "submitted collector job: $CID (held on $AID)"
echo
echo "watch:   qstat -u \$USER"
echo "report:  cat $REPO/experiments/ss2p2_w1024/REPORT.txt"
