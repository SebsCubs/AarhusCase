"""Plot PPO training curves from a Stable-Baselines3 TensorBoard event dir.

Reads the scalar tags logged during Skoven RL training (rollout/train/eval)
and renders the 9-panel training-stats figure used to diagnose a run:
policy quality (ep_rew_mean, eval reward), convergence (std, explained_var,
value_loss), and the physical reward terms (heat_kW, ahu_kW, temp_violation,
comfort_ok) against the rule-based baseline reference lines.

Usage:
    python scripts/plot_rl_training.py                     # PPO_2 -> default out
    python scripts/plot_rl_training.py --run-dir use_case/logs/PPO_2 \
        --out use_case/plots_rl_training/training_stats_v2sup_fixed.png \
        --title "Skoven PPO v2-sup (AHU@1.0, full-winter eval) 1M"
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(run_dir):
    ea = EventAccumulator(run_dir)
    ea.Reload()
    tags = set(ea.Tags()["scalars"])

    def series(tag):
        if tag not in tags:
            return [], []
        evs = ea.Scalars(tag)
        return [e.step for e in evs], [e.value for e in evs]

    return series


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="use_case/logs/PPO_2")
    p.add_argument("--out", default="use_case/plots_rl_training/training_stats_v2sup_fixed.png")
    p.add_argument("--title", default="Skoven PPO — v2-sup fixed (AHU@1.0, full-winter eval)")
    # Rule-based baseline reference values (heating-season --compare incumbent).
    p.add_argument("--baseline-heat-kw", type=float, default=1.29)
    p.add_argument("--baseline-ahu-kw", type=float, default=0.37)
    args = p.parse_args()

    series = load_scalars(args.run_dir)

    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    fig.suptitle(args.title, fontsize=14, fontweight="bold")

    def panel(ax, tag, title, hline=None, hlabel=None):
        x, y = series(tag)
        if x:
            ax.plot(x, y, lw=1.2)
        if hline is not None:
            ax.axhline(hline, ls="--", color="k", lw=1, label=hlabel)
            if hlabel:
                ax.legend(fontsize=8, loc="best")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("timestep")
        ax.grid(alpha=0.3)

    # Row 1 — policy quality / convergence
    panel(axes[0, 0], "rollout/ep_rew_mean", "ep_rew_mean (train)")
    x, y = series("eval/mean_reward")
    if x:
        axes[0, 1].plot(x, y, color="tab:green")
        ib = max(range(len(y)), key=lambda i: y[i])
        axes[0, 1].scatter([x[ib]], [y[ib]], color="red", zorder=5,
                           label=f"best @ {x[ib]} = {y[ib]:.0f}")
        axes[0, 1].legend(fontsize=8, loc="best")
    axes[0, 1].set_title("eval/mean_reward (full-winter, held-out)", fontsize=11)
    axes[0, 1].set_xlabel("timestep")
    axes[0, 1].grid(alpha=0.3)
    panel(axes[0, 2], "train/std", "train/std (policy noise)")

    # Row 2 — energy terms vs baseline + fit quality
    panel(axes[1, 0], "rollout/heat_kW", "rollout/heat_kW [kW]",
          hline=args.baseline_heat_kw, hlabel=f"baseline {args.baseline_heat_kw:.2f} kW")
    panel(axes[1, 1], "rollout/ahu_kW", "rollout/ahu_kW [kW]",
          hline=args.baseline_ahu_kw, hlabel=f"baseline {args.baseline_ahu_kw:.2f} kW")
    panel(axes[1, 2], "train/explained_variance", "train/explained_variance")

    # Row 3 — comfort terms + value loss. Prefer temp_dev (RMS °C from the 21°C
    # target — the deviation-from-target objective) when logged; fall back to the
    # band-violation term for older runs that didn't log it.
    x, _ = series("rollout/temp_dev")
    if x:
        panel(axes[2, 0], "rollout/temp_dev", "rollout/temp_dev [RMS °C from 21]")
    else:
        panel(axes[2, 0], "rollout/temp_violation", "rollout/temp_violation")
    panel(axes[2, 1], "rollout/comfort_ok", "rollout/comfort_ok [frac in band]")
    panel(axes[2, 2], "train/value_loss", "train/value_loss")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
