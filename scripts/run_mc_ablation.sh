#!/usr/bin/env bash
# SINGLE ENTRY POINT for the MC-compensator ablation (see mc_ablation.sh
# header for the design). Submits the 4-task GPU array (ss2p2/s2p2 x
# endpoint/MC) + a held collector that prints the paired comparison.
#
#   ssh peacock
#   cd ~/simulation && bash scripts/run_mc_ablation.sh
#
# Read the result as: mc arm vs ep arm within each model pair --
# mean_u (prediction table) and sim_rate (facts table) are the endpoints.
set -euo pipefail
REPO="${REPO:-$HOME/simulation}"
cd "$REPO"
mkdir -p logs
cd logs

AID=$(qsub -terse "$REPO/scripts/mc_ablation.sh" | head -1 | cut -d. -f1)
echo "submitted mc_ablation array job: $AID (tasks 1-4: ss2p2-ep ss2p2-mc s2p2-ep s2p2-mc)"

cat > "$REPO/logs/_mc_ablation_collect_job.sh" <<EOF
#!/usr/bin/env bash
#\$ -S /bin/bash
#\$ -cwd
#\$ -j y
#\$ -N mc_abl_collect
#\$ -l h_rt=0:20:00
#\$ -l tmem=8G
cd "$REPO"
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "\$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO"
python3 scripts/eval_collect.py "$REPO/experiments/mc_ablation" | tee "$REPO/experiments/mc_ablation/REPORT.txt"
EOF
CID=$(qsub -terse -hold_jid "$AID" "$REPO/logs/_mc_ablation_collect_job.sh" | head -1 | cut -d. -f1)
echo "submitted collector job: $CID (held on $AID)"
echo
echo "watch:   qstat -u \$USER"
echo "report:  cat $REPO/experiments/mc_ablation/REPORT.txt"
