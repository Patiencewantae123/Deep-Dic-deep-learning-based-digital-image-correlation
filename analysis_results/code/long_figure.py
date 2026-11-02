"""Validation error over both training runs (train.py, then train_long.py), against the paper's final value.

    python long_figure.py <long_run_log_dir> <out_png>
"""
import csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LONG, OUT = sys.argv[1], sys.argv[2]
SURFACE, INK, MUTED, GRID, AXIS, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#2a78d6"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "axes.linewidth": 1,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.titlesize": 12,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})
PAPER = {"disp": 0.024, "strain": 0.031}  # Table 3, validation average after 200 epochs


def rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [(int(r["step"]), int(r["phase"]), float(r["val_mae"])) for r in csv.DictReader(f)]


def series(task, scale):
    first = rows(os.path.join(HERE, "runs", f"{task}_val.csv"))
    second = rows(os.path.join(LONG, f"long_{task}_val.csv"))
    seen = {}
    for s, ph, v in first + second:
        seen[s] = v * scale
    drops = []  # steps where the learning rate fell to 1e-5
    for run in (first, second):
        p2 = [s for s, ph, _ in run if ph == 2]
        if p2:
            drops.append(min(p2))
    end_first = max(s for s, _, _ in first)
    x = np.array(sorted(seen))
    return x, np.array([seen[k] for k in x]), drops, end_first


fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")
for ax, task, scale, unit, name in ((axes[0], "disp", .5, "px", "DisplacementNet"), (axes[1], "strain", 1, "%", "StrainNet")):
    x, y, drops, end_first = series(task, scale)
    ax.set_yscale("log")
    ax.grid(axis="y", color=GRID, linewidth=1, which="major")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    lo, hi = PAPER[task] * .7, max(y) * 1.6
    ax.set_ylim(lo, hi)
    for d in drops:
        ax.axvline(d, color=AXIS, lw=1)
        ax.text(d, hi * .92, "rate 1e-5 ", color=MUTED, fontsize=8.5, va="top", ha="right")
    ax.axvline(end_first, color=AXIS, lw=1, ls=":")
    ax.text(end_first, hi * .92, " long run\n starts", color=MUTED, fontsize=8.5, va="top")
    ax.axhline(PAPER[task], color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.text(max(x) * .01, PAPER[task] * 1.12, f"Paper after 200 epochs (~600,000 steps): {PAPER[task]} {unit}", va="bottom", fontsize=9)
    ax.plot(x, y, color=BLUE, lw=2, marker="o", ms=4.5, mec=SURFACE, mew=1.5)
    # Below the end point: the curve before it is higher, so a label above would sit on the line
    ax.annotate(f"{y[-1]:.2f} {unit} at step {x[-1]:,}", (x[-1], y[-1]), xytext=(0, -10), textcoords="offset points",
                ha="right", va="top", fontweight="bold")
    ticks = [t for t in (.01, .02, .05, .1, .2, .5, 1, 2, 5, 10) if lo <= t <= hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0, max(x) * 1.02)
    ax.set_xlabel("Training step (batch of 12)")
    ax.set_ylabel(f"Validation mean error ({unit}, log scale)")
    ax.set_title(name)
fig.savefig(OUT, dpi=130)
print("saved", OUT)
