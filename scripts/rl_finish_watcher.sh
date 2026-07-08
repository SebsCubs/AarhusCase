#!/usr/bin/env bash
# Waits for the running PPO training PID to exit, then auto-generates the
# training-stats plot and the RL-vs-baseline comparison. Invoked in the
# background by Claude; results land in the paths echoed at the end.
# Args: <train_pid> <ppo_run_dir> <tag>
#   ppo_run_dir e.g. use_case/logs/PPO_3 ; tag e.g. v2sup_devtarget
set -u
cd /home/sebscubs/repos/AarhusCase/AarhusCase
PID="${1:?usage: rl_finish_watcher.sh <train_pid> <ppo_run_dir> <tag>}"
RUN_DIR="${2:?need PPO run dir, e.g. use_case/logs/PPO_3}"
TAG="${3:?need a tag, e.g. v2sup_devtarget}"
PY=.venv/bin/python
PLOT="use_case/plots_rl_training/training_stats_${TAG}.png"
CMP="use_case/logs/compare_${TAG}.log"
TRAINLOG="use_case/logs/train_1M_${TAG}.log"

echo "[watcher] waiting for training PID $PID to finish..."
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
echo "[watcher] PID $PID exited at $(date -Is)"

# Did it finish cleanly? ppo_skoven.zip is saved only after learn() returns.
if [ -f use_case/logs/ppo_skoven.zip ] && \
   find use_case/logs/ppo_skoven.zip -newermt "-2 hours" | grep -q .; then
  echo "[watcher] ppo_skoven.zip is fresh -> training completed normally."
else
  echo "[watcher] WARNING: ppo_skoven.zip NOT refreshed -> training may have crashed early."
  echo "[watcher] tail of train log:"; tail -25 "$TRAINLOG"
fi

echo "[watcher] === generating training-stats plot ==="
$PY scripts/plot_rl_training.py \
  --run-dir "$RUN_DIR" \
  --out "$PLOT" \
  --title "Skoven PPO v2-sup ${TAG} (dev-from-target 21C) 1M [$(basename "$RUN_DIR")]" \
  2>&1 | grep -viE "Warning|Box high|precision"

echo "[watcher] === RL vs baseline compare (best_model.zip, full winter) ==="
$PY use_case/baseline_eval.py --compare > "$CMP" 2>&1
echo "[watcher] compare exit=$? -> $CMP"

echo "[watcher] === RESULT TABLE ==="
grep -A12 "RL vs. baseline" "$CMP"
echo "[watcher] === COMFORT/ENERGY KPIs (RL side) ==="
grep -E "Total heating energy|Total AHU energy|TOTAL:|mean time in band|RMSE|Mean reward|Supply water" "$CMP" | tail -14
echo "[watcher] DONE $(date -Is)"
