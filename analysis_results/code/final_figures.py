"""Labelled PNG figures for analysis_results/figures, read directly from the result files."""
import csv, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
SURFACE, INK, MUTED, GRID, AXIS, BLUE, QUIET = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#2a78d6", "#b4bfc9"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "axes.linewidth": 1,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.titlesize": 12,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})
DIV = LinearSegmentedColormap.from_list("div", ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#ec9a8e", "#e34948", "#8f1f1f"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#f0f6fe", "#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])
E = json.load(open(os.path.join(HERE, "eval.json")))
D = json.load(open(os.path.join(HERE, "strain_diagnose.json")))
save = lambda fig, name: (fig.savefig(os.path.join(OUT, name), dpi=130), plt.close(fig))


def curve(task, scale):
    seen, drop = {}, None
    with open(os.path.join(HERE, "runs", f"{task}_val.csv")) as f:
        for r in csv.DictReader(f):
            seen[int(r["step"])] = float(r["val_mae"]) * scale
            if r["phase"] == "2" and drop is None:
                drop = int(r["step"])
    return np.array(sorted(seen)), np.array([seen[k] for k in sorted(seen)]), drop


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
for ax, task, scale, unit, title in ((axes[0], "disp", .5, "px", "DisplacementNet learned"), (axes[1], "strain", 1, "%", "StrainNet stayed at its starting point")):
    x, y, drop = curve(task, scale)
    tidy(ax)
    ax.axhline(y[0], color=MUTED, lw=1)
    ax.axvline(drop, color=AXIS, lw=1)
    ax.plot(x, y, color=BLUE, lw=2, marker="o", ms=6, mec=SURFACE, mew=2)
    ax.set_ylim(0, max(y) * 1.14)
    ax.set_xlim(0, max(x) * 1.03)
    ax.text(max(x) * 1.02, y[0] + max(y) * .015, f"Untrained: {y[0]:.2f} {unit}", color=MUTED, va="bottom", ha="right") if y[-1] < y[0] * .8 else ax.text(5, y[0] + max(y) * .015, f"Untrained: {y[0]:.2f} {unit}", color=MUTED, va="bottom")
    ax.text(drop - 6, max(y) * 1.13, "Rate 1e-3 to 1e-5", color=MUTED, ha="right", va="top")
    ax.annotate(f"{y[-1]:.2f} {unit}", (x[-1], y[-1]), xytext=(0, -16), textcoords="offset points", ha="right", fontweight="bold")
    ax.set_xlabel("Training step (batch of 12)")
    ax.set_ylabel(f"Validation mean error ({unit})")
    ax.set_title(title)
save(fig, "01_training_curves.png")


def barh(ax, rows, unit, title):
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    cols = [BLUE if r[2] else QUIET for r in rows][::-1]
    ax.barh(labels, vals, color=cols, height=.5)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * .012, i, f"{v:.2f} {unit}", va="center", fontweight="bold")
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(vals) * 1.2)
    ax.set_title(title)


t = E["test"]
dp, sp, ev, pre = t["disp_px"], t["strain_pct"], E["eval_bug"], E["preprocessing"]
fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), layout="constrained")
barh(axes[0], [("Farneback optical flow (classical)", dp["farneback"]["mean"], 0), ("DisplacementNet with learning-rate drop", dp["net_p2"]["mean"], 1),
               ("DisplacementNet, notebook settings only", dp["net_p1"]["mean"], 0), ("Predicting zero everywhere", dp["zero"]["mean"], 0)], "px", "Displacement error on 300 unseen pairs")
barh(axes[1], [("Differentiated Farneback flow", sp["from_farneback"]["mean"], 0), ("StrainNet", sp["strainnet_p2"]["mean"], 1),
               ("Predicting zero everywhere", sp["zero"]["mean"], 0), ("Differentiated DisplacementNet", sp["from_dispnet"]["mean"], 0)], "%", "Strain error on 300 unseen pairs")
save(fig, "02_test_results.png")

fig, axes = plt.subplots(2, 1, figsize=(10, 4.4), layout="constrained")
barh(axes[0], [("Fixed: fcn.eval() called", ev["eval"], 1), ("Committed: eval() never called", ev["train_mode"], 0),
               ("Eval after 150 frames in train mode", ev["polluted_eval"], 0)], "px", "Inference without fcn.eval(): displacement error")
barh(axes[1], [("Fixed: each image over its own maximum", pre["dim_second_norm_each"], 1), ("Committed StrainNet_run: image 2 over image 1's maximum", pre["dim_second_norm_first"], 0),
               ("Training's preprocessing: no normalisation", pre["dim_second_raw"], 0)], "px", "Light dims 20% between two frames: displacement error")
save(fig, "03_silent_bugs.png")


def panels(name, title_u="Horizontal u", title_v="Vertical v"):
    d = np.load(os.path.join(HERE, "fields", f"{name}.npz"))
    g, p, flow, a, b = d["g"], d["p"], d["flow"], d["a"], d["b"]
    err = np.abs(p - g)
    L, Em = float(np.ceil(np.abs(g).max() * 2) / 2), float(np.ceil(err.max() * 2) / 2)
    fig, ax = plt.subplots(2, 5, figsize=(13, 5.6), layout="constrained")
    for r, (img, nm, ttl) in enumerate(((a, "Reference image", title_u), (b, "Deformed image", title_v))):
        ax[r, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
        ax[r, 0].set_title(nm, fontsize=10.5, fontweight="normal")
        m1 = ax[r, 1].imshow(g[r], cmap=DIV, vmin=-L, vmax=L)
        ax[r, 1].set_title(f"{ttl}: true", fontsize=10.5, fontweight="normal")
        ax[r, 2].imshow(p[r], cmap=DIV, vmin=-L, vmax=L)
        ax[r, 2].set_title("DisplacementNet", fontsize=10.5, fontweight="normal")
        ax[r, 3].imshow(flow[r], cmap=DIV, vmin=-L, vmax=L)
        ax[r, 3].set_title("Farneback", fontsize=10.5, fontweight="normal")
        m2 = ax[r, 4].imshow(err[r], cmap=SEQ, vmin=0, vmax=Em)
        ax[r, 4].set_title(f"Network error, mean {err[r].mean():.2f} px", fontsize=10.5, fontweight="normal")
    for x in ax.flat:
        x.set_xticks([])
        x.set_yticks([])
        for s in x.spines.values():
            s.set_visible(False)
    fig.colorbar(m1, ax=ax[:, 1:4], orientation="horizontal", fraction=.05, aspect=60, label="Displacement (px, 128×128 input)")
    fig.colorbar(m2, ax=ax[:, 4], orientation="horizontal", fraction=.05, aspect=20, label="Absolute error (px)")
    return fig


save(panels("typical"), "04_displacement_median_sample.png")
save(panels("hard"), "05_displacement_hard_sample.png")

d = np.load(os.path.join(HERE, "fields", "typical.npz"))
st, ps, fd = d["st"][1], d["ps"][1], d["fromdisp"][1]
S = float(np.ceil(np.abs(st).max()))
fig, ax = plt.subplots(1, 3, figsize=(10.5, 5), layout="constrained")
for x, f, t_ in zip(ax, (st, ps, fd), ("True ∂u/∂x", "StrainNet", "Differentiated DisplacementNet")):
    m = x.imshow(f, cmap=DIV, vmin=-S, vmax=S)
    x.set_title(t_, fontsize=10.5, fontweight="normal")
    x.set_xticks([])
    x.set_yticks([])
    for s in x.spines.values():
        s.set_visible(False)
fig.colorbar(m, ax=ax, orientation="horizontal", fraction=.05, aspect=60, label="Strain (%)")
save(fig, "06_strain_median_sample.png")

h = np.array(D["overfit_hist"])
fig, ax = plt.subplots(figsize=(7.5, 3.8), layout="constrained")
tidy(ax)
ax.plot(np.arange(1, len(h) + 1), h, color=BLUE, lw=2)
ax.set_yscale("log")
ax.set_ylim(.01, 20)
ax.set_yticks([.01, .1, 1, 10])
ax.set_yticklabels(["0.01", "0.1", "1", "10"])
ax.minorticks_off()
ax.set_xlim(0, len(h) * 1.04)
ax.set_xlabel("Training step, same 12 images every step")
ax.set_ylabel("MSE, log scale")
ax.set_title(f"StrainNet can learn: loss fell {(1 - D['overfit_end'] / D['overfit_start']) * 100:.1f}% on 12 images")
save(fig, "07_strain_overfit_test.png")
print(sorted(os.listdir(OUT)))
