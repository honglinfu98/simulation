#!/usr/bin/env bash
# SINGLE ENTRY POINT for the Coinbase multi-asset comparison
# (see multi_asset_cbse.sh): BTC/ETH/SOL x 6 models x 3 training seeds x
# 3 rollout seeds, plus one strict per-coin report job held on the array.
#
#   ssh peacock
#   cd ~/simulation && bash scripts/run_multi_asset_cbse.sh
set -euo pipefail
REPO="${REPO:-$HOME/simulation}"
cd "$REPO"
mkdir -p logs
cd logs

AID=$(qsub -terse "$REPO/scripts/multi_asset_cbse.sh" | head -1 | cut -d. -f1)
echo "submitted ma_cbse array job: $AID (54 tasks: 3 coins x 6 models x 3 seeds)"

for COIN in btc eth sol; do
  cat > "$REPO/logs/_ma_cbse_report_${COIN}.sh" <<EOF
#!/usr/bin/env bash
#\$ -S /bin/bash
#\$ -cwd
#\$ -j y
#\$ -N ma_report_${COIN}
#\$ -l h_rt=0:20:00
#\$ -l tmem=8G
cd "$REPO"
source /share/apps/source_files/python/python-3.11.9.source 2>/dev/null || true
source "\$HOME/volume-set-mtpp/venv/bin/activate" 2>/dev/null || true
export PYTHONPATH="$REPO"
export EXPECT_MODELS="nhp,lstm,sahp,pct-lstm,s2p2,ss2p2-full"
# EXCLUDE_SF: set per coin AFTER inspecting any CAL failures (documented
# exclusions only -- see final_report.py header).
python3 scripts/final_report.py "$REPO/experiments/ma_cbse/${COIN}" --seeds 1,2,3 --rollout-seeds 1,2,3 \
  | tee "$REPO/experiments/ma_cbse/${COIN}/REPORT.txt"
rc=\${PIPESTATUS[0]}
echo; echo "== rollout modes + calibration constants (${COIN}) =="
grep -h "CONTEXT_MODE\|CALIBRATED" "$REPO"/experiments/ma_cbse/${COIN}/*/sf_r*.log 2>/dev/null | sort | uniq -c
exit \$rc
EOF
  CID=$(qsub -terse -hold_jid "$AID" "$REPO/logs/_ma_cbse_report_${COIN}.sh" | head -1 | cut -d. -f1)
  echo "submitted ${COIN} report job: $CID (held on $AID; FAILS if incomplete)"
done
echo
echo "watch:    qstat -u \$USER"
echo "reports:  cat $REPO/experiments/ma_cbse/{btc,eth,sol}/REPORT.txt"
