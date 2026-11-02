"""Figure for roi_test.py: where each ROI version ends up, and how far its edges drift from the material.

    python roi_figure.py <roi_test.json> <out_png>
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(sys.argv[1]))
SURFACE, INK, MUTED, GRID, AXIS, QUIET = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#b4bfc9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "axes.linewidth": 1,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.titlesize": 12,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})
SERIES = (("before", "Committed notebook", ORANGE, "-"), ("sign_only", "Sign flipped only", AQUA, "-"),
          ("fixed", "Fixed notebook", BLUE, "-"), ("fixed_flow", "Fixed, optical flow instead of the network", QUIET, "--"))
truth = np.array(R["truth"])
n = R["frames"]

fig = plt.figure(figsize=(11.5, 5.4), layout="constrained")
gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.6])
ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

# Left: the last frame with the starting ROI, where the material went, and where two versions put the ROI
img = np.array(Image.open(os.path.join(HERE, "roi_work", "images", f"frame_{n:03d}.png")))
top, bot = 280, 900
ax0.imshow(img[top:bot], cmap="gray", vmin=0, vmax=255, extent=(0, img.shape[1], bot, top))


def box(roi, color, ls, lw):
    x1, x2, y1, y2 = roi
    ax0.add_patch(Rectangle((y1, x1), y2 - y1, x2 - x1, fill=False, ec=SURFACE, lw=lw + 2.5, ls=ls))
    ax0.add_patch(Rectangle((y1, x1), y2 - y1, x2 - x1, fill=False, ec=color, lw=lw, ls=ls))


box(R["runs"]["before"][-1], ORANGE, "-", 2)
box(R["runs"]["fixed"][-1], BLUE, "-", 2)
box(truth[-1], INK, (0, (4, 3)), 1.6)
ax0.set_xticks([])
ax0.set_yticks([])
for s in ax0.spines.values():
    s.set_visible(False)
ax0.set_title(f"Frame {n}: ROI positions", fontsize=11)
handles = [plt.Line2D([], [], color=c, lw=2, ls=ls) for c, ls in ((INK, (0, (4, 3))), (BLUE, "-"), (ORANGE, "-"))]
ax0.legend(handles, ["Material (truth)", "Fixed notebook", "Committed notebook"], loc="upper center",
           bbox_to_anchor=(.5, -.01), frameon=False, fontsize=9)

# Right: worst edge distance from the material, frame by frame
frames = np.arange(n + 1)
ends = []
for key, label, color, ls in SERIES:
    err = np.abs(np.array(R["runs"][key], float) - truth).max(1)
    ax1.plot(frames, err, color=color, lw=2, ls=ls, label=label)
    ends.append([err[-1], label, color])
ends.sort()
gap, prev = 7.0, -1e9
for e in ends:  # nudge the end labels apart so none overlap
    e.append(max(e[0], prev + gap))
    prev = e[3]
for val, label, color, y in ends:
    ax1.annotate(f"{label}: {val:.0f} px", (n, val), xytext=(n + 1, y), textcoords="data", va="center", fontsize=9.5,
                 color=INK, arrowprops=dict(arrowstyle="-", color=AXIS, lw=.8) if abs(y - val) > 1 else None)
ax1.spines[["top", "right"]].set_visible(False)
ax1.grid(axis="y", color=GRID, linewidth=1)
ax1.set_axisbelow(True)
ax1.set_xlim(0, n)
ax1.set_ylim(0, max(e[0] for e in ends) * 1.08)
ax1.set_xlabel("Frame (each step adds 0.4% vertical stretch and 0.4 px sideways drift)")
ax1.set_ylabel("Worst ROI edge distance from the material (px)")
ax1.set_title("The committed ROI drifts away from the material; the fixed ROI follows it")
ax1.legend(loc="upper left", frameon=False, fontsize=9)
fig.savefig(sys.argv[2], dpi=130, bbox_inches="tight", pad_inches=.15)
print("saved", sys.argv[2])
