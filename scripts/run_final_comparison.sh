#!/usr/bin/env bash
# SINGLE ENTRY POINT for the paper's final comparison benchmark (see
# final_comparison.sh header): NHP, LSTM, SAHP, PCT-LSTM, S2P2 (standard
# training) vs SS2P2-full (TBPTT + carried rollout + calibration), all at
# seq 1024 / stride 1024 / 40 epochs / seed 1.
#
#   ssh peacock
#   cd ~/simulation && bash scripts/run_final_comparison.sh
set -euo pipefail
REPO="${REPO:-$HOME/simulation}"
cd "$REPO"
mkdir -p logs
cd logs

AID=$(qsub -terse "$REPO/scripts/final_comparison.sh" | head -1 | cut -d. -f1)
echo "submitted final_comparison array job: $AID (tasks 1-6: nhp lstm sahp pct-lstm s2p2 ss2p2-full)"

cat > "$REPO/logs/_final_cmp_collect_job.sh" <<EOF
#!/usr/bin/env bash
#\$ -S /bin/bash
#\$ -cwd
#\$ -j y
#\$ -N final_cmp_collect
#\$ -l h_rt=0:20:00
#\$ -l tmem=8G
cd "$REPO"
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "\$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO"
python3 scripts/eval_collect.py "$REPO/experiments/final_comparison" | tee "$REPO/experiments/final_comparison/REPORT.txt"
echo; echo "== rollout modes + calibration =="
grep -h "CONTEXT_MODE\|CALIBRATED" "$REPO"/experiments/final_comparison/*/sf.log 2>/dev/null
EOF
CID=$(qsub -terse -hold_jid "$AID" "$REPO/logs/_final_cmp_collect_job.sh" | head -1 | cut -d. -f1)
echo "submitted collector job: $CID (held on $AID)"
echo
echo "watch:   qstat -u \$USER"
echo "report:  cat $REPO/experiments/final_comparison/REPORT.txt"
