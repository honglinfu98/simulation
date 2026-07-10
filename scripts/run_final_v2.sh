#!/usr/bin/env bash
# SINGLE ENTRY POINT for the final comparison v2 (see final_comparison_v2.sh):
# 6 models x 3 training seeds x 3 rollout seeds, val-split calibration for
# every model, streaming prediction evaluator, equal-duration facts, strict
# failure modes, mean +/- 95% CI report.
#
#   ssh peacock
#   cd ~/simulation && bash scripts/run_final_v2.sh
set -euo pipefail
REPO="${REPO:-$HOME/simulation}"
cd "$REPO"
mkdir -p logs
cd logs

AID=$(qsub -terse "$REPO/scripts/final_comparison_v2.sh" | head -1 | cut -d. -f1)
echo "submitted final_v2 array job: $AID (18 tasks: 6 models x 3 seeds)"

cat > "$REPO/logs/_final_v2_report_job.sh" <<EOF
#!/usr/bin/env bash
#\$ -S /bin/bash
#\$ -cwd
#\$ -j y
#\$ -N final_v2_report
#\$ -l h_rt=0:20:00
#\$ -l tmem=8G
cd "$REPO"
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "\$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO"
export EXPECT_MODELS="nhp,lstm,sahp,pct-lstm,s2p2,ss2p2-full"
python3 scripts/final_report.py "$REPO/experiments/final_v2" --seeds 1,2,3 --rollout-seeds 1,2,3 \
  | tee "$REPO/experiments/final_v2/REPORT.txt"
rc=\${PIPESTATUS[0]}
echo; echo "== rollout modes + calibration constants =="
grep -h "CONTEXT_MODE\|CALIBRATED" "$REPO"/experiments/final_v2/*/sf_r*.log 2>/dev/null | sort | uniq -c
exit \$rc
EOF
CID=$(qsub -terse -hold_jid "$AID" "$REPO/logs/_final_v2_report_job.sh" | head -1 | cut -d. -f1)
echo "submitted report job: $CID (held on $AID; FAILS if any model incomplete)"
echo
echo "watch:   qstat -u \$USER"
echo "report:  cat $REPO/experiments/final_v2/REPORT.txt"
