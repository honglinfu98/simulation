#!/usr/bin/env bash
# SINGLE ENTRY POINT for the post-hoc rate-calibration ablation (Route 1; see
# rate_cal_ablation.sh header). Simulation-only -- reuses the trained
# tbptt_ablation checkpoints, ~minutes per task on a GPU.
#
#   ssh peacock
#   cd ~/simulation && bash scripts/run_rate_cal.sh
set -euo pipefail
REPO="${REPO:-$HOME/simulation}"
cd "$REPO"
[ -s "$REPO/experiments/tbptt_ablation/ss2p2-tbptt/train/best_model.pt" ] \
  || { echo "missing tbptt_ablation checkpoints -- run scripts/run_tbptt_ablation.sh first"; exit 1; }
mkdir -p logs
cd logs

AID=$(qsub -terse "$REPO/scripts/rate_cal_ablation.sh" | head -1 | cut -d. -f1)
echo "submitted rate_cal array job: $AID (tasks 1-4: tbptt-uncal tbptt-cal cold-uncal cold-cal)"

cat > "$REPO/logs/_rate_cal_collect_job.sh" <<EOF
#!/usr/bin/env bash
#\$ -S /bin/bash
#\$ -cwd
#\$ -j y
#\$ -N rate_cal_collect
#\$ -l h_rt=0:20:00
#\$ -l tmem=8G
cd "$REPO"
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "\$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO"
python3 scripts/eval_collect.py "$REPO/experiments/rate_cal" | tee "$REPO/experiments/rate_cal/REPORT.txt"
echo; echo "== calibration constants =="
grep -h "CALIBRATED" "$REPO"/experiments/rate_cal/*/sf.log 2>/dev/null
EOF
CID=$(qsub -terse -hold_jid "$AID" "$REPO/logs/_rate_cal_collect_job.sh" | head -1 | cut -d. -f1)
echo "submitted collector job: $CID (held on $AID)"
echo
echo "watch:   qstat -u \$USER"
echo "report:  cat $REPO/experiments/rate_cal/REPORT.txt"
