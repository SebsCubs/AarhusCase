"""Render the closed-loop ECL310 substation block diagram used in the paper.

Writes scripts/plots/skoven_closed_loop_diagram.png. Pure matplotlib (boxes +
arrows), no simulation needed.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUT, exist_ok=True)

HOT = "#c44"      # hot / primary
WARM = "#e08a3c"  # supply
COOL = "#3b76c4"  # return / recirc
CTRL = "#5a5a5a"  # control
BOX = "#f4f4f4"


def box(ax, xy, w, h, text, ec="#333", fc=BOX, fs=10, lw=1.4):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                ec=ec, fc=fc, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3)
    return (x, y, w, h)


def arrow(ax, p0, p1, color="#333", lw=2.0, style="-|>", rad=0.0, label=None, lpos=0.5, ldy=0.12):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=16,
                                 lw=lw, color=color, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my + ldy, label, ha="center", va="center", fontsize=8.5, color=color, zorder=4)


fig, ax = plt.subplots(figsize=(11, 5.2))
ax.set_xlim(0, 11); ax.set_ylim(0, 5.4); ax.axis("off")

# Components
prim = box(ax, (0.2, 2.3), 1.7, 0.9, "District-heating\nprimary (hot)", ec=HOT, fc="#fbeaea")
valve = box(ax, (2.6, 2.35), 1.5, 0.8, "ECL310\nmixing valve", ec=HOT)
mix = box(ax, (4.7, 2.3), 1.5, 0.9, "Supply mixing\njunction", ec=WARM, fc="#fdf0e3")
rad = box(ax, (6.9, 2.3), 1.6, 0.9, "4 radiators\n(fixed flow)", ec=WARM)
zones = box(ax, (9.0, 2.3), 1.6, 0.9, "4 RC zones\n(ring)", ec="#2a8")
retman = box(ax, (6.9, 0.5), 1.6, 0.8, "Return\nmanifold (mix)", ec=COOL, fc="#e9f0fb")

# Controller
pid = box(ax, (2.55, 4.1), 1.6, 0.8, "ECL310 PID", ec=CTRL, fc="#eee")
setp = box(ax, (0.2, 4.1), 1.9, 0.8, "Measured supply\nset-point (curve)", ec=CTRL, fc="#eee")

# Primary flow path (hot)
arrow(ax, (1.9, 2.75), (2.6, 2.75), color=HOT, label="primary", ldy=0.16)
arrow(ax, (4.1, 2.75), (4.7, 2.75), color=HOT)
# Supply path (warm)
arrow(ax, (6.2, 2.75), (6.9, 2.75), color=WARM, label="supply $T_{sup}$", ldy=0.16)
arrow(ax, (8.5, 2.75), (9.0, 2.75), color="#2a8", label="heat", ldy=0.16)
# Radiators -> return manifold
arrow(ax, (7.7, 2.3), (7.7, 1.3), color=COOL, label="return", lpos=0.5, ldy=0.0)
# Return manifold -> recirc back into mixing junction (closing the loop)
arrow(ax, (6.9, 0.9), (5.45, 0.9), color=COOL)
arrow(ax, (5.45, 0.9), (5.45, 2.3), color=COOL, label="recirc", ldy=0.0)
# Return manifold also scored vs measured (annotation)
ax.text(7.7, 0.25, "scored vs. measured BMS return", ha="center", fontsize=8, color=COOL)

# Control signals (dashed)
arrow(ax, (2.1, 4.5), (2.55, 4.5), color=CTRL, label="$T_{set}$", ldy=0.16)
arrow(ax, (3.35, 4.1), (3.35, 3.15), color=CTRL, label="valve pos.", lpos=0.5, ldy=0.0)
# Feedback: produced supply -> PID
arrow(ax, (5.45, 3.2), (5.45, 3.7), color=CTRL, rad=0.0)
arrow(ax, (5.45, 3.7), (4.15, 3.7), color=CTRL)
arrow(ax, (4.15, 3.7), (4.15, 4.1), color=CTRL, label="$T_{sup}$ feedback", lpos=0.5, ldy=0.18)

ax.text(5.5, 5.15, "Closed-loop ECL310 substation: supply, return and zone temperatures are coupled model outputs",
        ha="center", fontsize=10.5, weight="bold")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "skoven_closed_loop_diagram.png"), dpi=150, bbox_inches="tight")
print("wrote", os.path.join(OUT, "skoven_closed_loop_diagram.png"))
