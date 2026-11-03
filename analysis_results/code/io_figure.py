"""Inputs and outputs of the two run notebooks on roi_test.py's 40-frame tensile test, from pipeline_test.py's run.

    python io_figure.py <pipeline_work_dir> <out_png>

<pipeline_work_dir> is roi_work/pipeline_net (trained networks) or roi_work/pipeline_flow (optical flow). Row 1 is
what goes in: the frames, the ROI and the two-channel tensor each network receives. Rows 2 and 3 are the saved
outputs for the last frame pair, and the accumulated fields, next to the truth.
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np, scipy.io as sio
from PIL import Image
import roi_test as T

WORK, OUT = sys.argv[1], sys.argv[2]
SURFACE, INK, MUTED, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9.5, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.titlesize": 9.5, "axes.titlelocation": "left"})
DIV = LinearSegmentedColormap.from_list("div", ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#ec9a8e", "#e34948", "#8f1f1f"])
n = T.N_FRAMES
frame = lambda k: np.array(Image.open(os.path.join(T.FRAMES, f"frame_{k:03d}.png")).convert("L"))
pos = sio.loadmat(os.path.join(WORK, "displacement", f"result_{n}_position1.mat"))
roi_used, roi_next = pos["roi_used"].ravel(), pos["position"].ravel()[:4]
dm = sio.loadmat(os.path.join(WORK, "displacement", f"result_{n}_disp_0_x.mat"))
sm = sio.loadmat(os.path.join(WORK, "strain", f"{n}strain.mat"))
d_pair, d_acc = dm["disp_1_x"], dm["disp_acc"]
s_pair, s_acc = sm["strain_xx_yy_xy"] * 100, sm["strain_acc"] * 100

x1, x2, y1, y2 = T.ROI0
h, w = x2 - x1, y2 - y1
XX, YY = np.meshgrid(y1 + np.arange(w, dtype=float), x1 + np.arange(h, dtype=float))
u_true, v_true = T.field(n, XX, YY)
# this pair's true increment, at the material points of the ROI used
du_pair, dv_pair = (a - b for a, b in zip(T.field(n, XX, YY), T.field(n - 1, XX, YY)))
eps = T.D_EPS * np.arange(n)
eyy_true, exx_true = np.sum(T.D_EPS / (1 + eps)) * 100, -np.sum(T.NU * T.D_EPS / (1 - T.NU * eps)) * 100
eyy_pair, exx_pair = T.D_EPS / (1 + eps[-1]) * 100, -T.NU * T.D_EPS / (1 - T.NU * eps[-1]) * 100


def off(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def box(ax, r, color, ls="-"):
    a, b, c, d = r
    ax.add_patch(Rectangle((c, a), d - c, b - a, fill=False, ec="white", lw=3.2))
    ax.add_patch(Rectangle((c, a), d - c, b - a, fill=False, ec=color, lw=1.8, ls=ls))


def field(ax, f, title, lim, unit, note=None):
    m = ax.imshow(f, cmap=DIV, vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set_title(title + (f"\n{note}" if note else "\n"), loc="left")
    ax.set_anchor("S")    # keep the image next to its colour bar; spare height goes above the title
    off(ax)
    cb = plt.colorbar(m, ax=ax, orientation="horizontal", fraction=.05, pad=.03, aspect=14)
    cb.set_label(unit, color=MUTED)
    cb.outline.set_visible(False)


nice = lambda v: float(np.ceil(v * 2) / 2) if v > 1 else float(np.ceil(v * 10) / 10)
fig = plt.figure(figsize=(15.5, 17.5), layout="constrained")
rows = fig.subfigures(3, 1, height_ratios=[1.1, 1, 1], hspace=.02)

# ---- Row 1: inputs
rows[0].suptitle("Inputs: the image sequence, the starting ROI and the trained weights", x=0, ha="left", fontweight="bold", fontsize=12)
ax = rows[0].subplots(1, 5, width_ratios=[.3, .3, .5, .5, .9])
f0, fn = frame(0), frame(n)
ax[0].imshow(f0, cmap="gray", vmin=0, vmax=255)
box(ax[0], T.ROI0, BLUE)
ax[0].set_title("images/frame_000.png\nstarting ROI")
ax[1].imshow(fn, cmap="gray", vmin=0, vmax=255)
truth_box = [x1 + T.field(n, (y1 + y2) / 2, x1)[1], x2 + T.field(n, (y1 + y2) / 2, x2)[1],
             y1 + T.field(n, y1, (x1 + x2) / 2)[0], y2 + T.field(n, y2, (x1 + x2) / 2)[0]]
box(ax[1], truth_box, INK, (0, (3, 2)))
box(ax[1], roi_next, BLUE)
ax[1].set_title(f"images/frame_{n:03d}.png\nROI after frame {n}")
a, b, c, d = roi_used
hn, wn = ((b - a) // 32 + 1) * 32, ((d - c) // 32 + 1) * 32
for k, axk, src in ((0, ax[2], frame(n - 1)), (1, ax[3], fn)):
    crop = np.array(Image.fromarray(src).crop((c, a, d, b)).resize((wn, hn))).astype(float)
    axk.imshow(crop / crop.max(), cmap="gray", vmin=0, vmax=1)
    axk.set_title(f"Network input, channel {k}\nframe {n - 1 + k} in the ROI, {hn}×{wn}")
for x in ax[:4]:
    off(x)
ax[4].axis("off")
lines = [("Image sequence", f"{n + 1} frames of {f0.shape[0]}×{f0.shape[1]} px in images/,\nread in file-name order. Pair i is frames i−1 and i."),
         ("Starting ROI", f"x1, x2, y1, y2 = {x1}, {x2}, {y1}, {y2}\n(top row, bottom row, left column, right column)"),
         ("Weights", "results/model/pretrained_displacementnet\nresults/model/pretrained_strainnet"),
         ("What each network gets", "Both frames cropped to the ROI, resized up to a multiple\nof 32 and divided by their own maximum: one 2-channel\ntensor per pair. StrainNet does not use DisplacementNet's\noutput.")]
yy = .97
for head, body in lines:
    ax[4].text(0, yy, head, fontweight="bold", va="top", transform=ax[4].transAxes)
    ax[4].text(0, yy - .045, body, va="top", transform=ax[4].transAxes, color=MUTED, linespacing=1.45)
    yy -= .07 + .045 * body.count("\n") + .05
ax[4].plot([0, .06], [yy - .01] * 2, color=BLUE, lw=2, transform=ax[4].transAxes)
ax[4].text(.08, yy - .01, "ROI the notebook tracks", va="center", transform=ax[4].transAxes)
ax[4].plot([0, .06], [yy - .07] * 2, color=INK, lw=1.8, ls=(0, (3, 2)), transform=ax[4].transAxes)
ax[4].text(.08, yy - .07, "Where the starting ROI's material really is", va="center", transform=ax[4].transAxes)

# ---- Row 2: DisplacementNet outputs
rows[1].suptitle(f"Diaplacement_run outputs: results/displacement/result_{n}_disp_0_x.mat", x=0, ha="left", fontweight="bold", fontsize=12)
ax = rows[1].subplots(1, 6)
lp = nice(np.percentile(np.abs(d_pair), 99))
rng = lambda a: f"{a.min():.1f} to {a.max():.1f}".replace("-", "−")
field(ax[0], d_pair[0], f"disp_1_x[0]: u, pair {n}", lp, "px", f"Truth: {rng(du_pair)} px")
field(ax[1], d_pair[1], f"disp_1_x[1]: v, pair {n}", lp, "px", f"Truth: {rng(dv_pair)} px")
lu, lv = nice(np.abs(u_true).max() * 1.15), nice(np.abs(v_true).max() * 1.15)
field(ax[2], d_acc[0], "disp_acc[0]: accumulated u", lu, "px", f"Mean error {np.abs(d_acc[0] - u_true).mean():.1f} px")
field(ax[3], u_true, "True accumulated u", lu, "px")
field(ax[4], d_acc[1], "disp_acc[1]: accumulated v", lv, "px", f"Mean error {np.abs(d_acc[1] - v_true).mean():.1f} px")
field(ax[5], v_true, "True accumulated v", lv, "px")

# ---- Row 3: StrainNet outputs
rows[2].suptitle(f"StrainNet_run outputs: results/strain/{n}strain.mat", x=0, ha="left", fontweight="bold", fontsize=12)
ax = rows[2].subplots(1, 5)
inner = (slice(h // 10, -h // 10), slice(w // 10, -w // 10))
mean = lambda a: f"{a[inner].mean():.2f}".replace("-", "−")
sp = nice(max(np.percentile(np.abs(s_pair[:2]), 99), abs(eyy_pair) * 1.5))
field(ax[0], s_pair[0], f"strain_xx_yy_xy[0]: εxx, pair {n}", sp, "%", f"Mean {mean(s_pair[0])}%, truth {exx_pair:.2f}%".replace("-", "−"))
field(ax[1], s_pair[1], f"strain_xx_yy_xy[1]: εyy, pair {n}", sp, "%", f"Mean {mean(s_pair[1])}%, truth {eyy_pair:.2f}%")
sa = nice(max(np.percentile(np.abs(s_acc), 99), eyy_true * 1.2))
field(ax[2], s_acc[0], "strain_acc[0]: accumulated εxx", sa, "%", f"Mean {mean(s_acc[0])}%, truth {exx_true:.1f}%".replace("-", "−"))
field(ax[3], s_acc[1], "strain_acc[1]: accumulated εyy", sa, "%", f"Mean {mean(s_acc[1])}%, truth {eyy_true:.1f}%")
field(ax[4], s_acc[2], "strain_acc[2]: accumulated εxy", sa, "%", f"Mean {mean(s_acc[2])}%, truth 0")

fig.savefig(OUT, dpi=110)
print("saved", OUT)
