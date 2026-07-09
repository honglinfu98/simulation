#!/usr/bin/env bash
# SINGLE ENTRY POINT for the TBPTT ablation (see tbptt_ablation.sh header for
# the design). Submits the 4-task GPU array ({ss2p2,s2p2} x {cold,tbptt},
# endpoint compensator, stride=seq=1024 in BOTH arms) + a held collector.
#
#   ssh peacock
#   cd ~/simulation && bash scripts/run_tbptt_ablation.sh
#
# Read the result within pairs: tbptt vs cold -- free-run sim_rate is the
# primary endpoint (state-regime fix), mean_u should move less (integral bias
# is the other, separate factor).
set -euo pipefail
REPO="${REPO:-$HOME/simulation}"
cd "$REPO"
mkdir -p logs
cd logs

AID=$(qsub -terse "$REPO/scripts/tbptt_ablation.sh" | head -1 | cut -d. -f1)
echo "submitted tbptt_ablation array job: $AID (tasks 1-4: ss2p2-cold ss2p2-tbptt s2p2-cold s2p2-tbptt)"

cat > "$REPO/logs/_tbptt_ablation_collect_job.sh" <<EOF
#!/usr/bin/env bash
#\$ -S /bin/bash
#\$ -cwd
#\$ -j y
#\$ -N tbptt_abl_collect
#\$ -l h_rt=0:20:00
#\$ -l tmem=8G
cd "$REPO"
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "\$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO"
python3 scripts/eval_collect.py "$REPO/experiments/tbptt_ablation" | tee "$REPO/experiments/tbptt_ablation/REPORT.txt"
EOF
CID=$(qsub -terse -hold_jid "$AID" "$REPO/logs/_tbptt_ablation_collect_job.sh" | head -1 | cut -d. -f1)
echo "submitted collector job: $CID (held on $AID)"
echo
echo "watch:   qstat -u \$USER"
echo "report:  cat $REPO/experiments/tbptt_ablation/REPORT.txt"
