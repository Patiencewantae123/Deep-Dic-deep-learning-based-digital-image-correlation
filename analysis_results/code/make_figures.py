"""Render the saved fields to PNG heatmaps, with no text in the image so that labels stay in the page and work in both themes.

Diverging fields: blue <-> red through a neutral grey. Error maps: one-hue sequential. Speckles: greyscale.
"""
import json, os, sys
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
SCALE = 3


def hex2bgr(h):
    h = h.lstrip("#")
    return np.array([int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)], np.float32)


DIVERGING = [hex2bgr(c) for c in ("#104281", "#2a78d6", "#86b6ef", "#f0efec", "#ec9a8e", "#e34948", "#8f1f1f")]
SEQUENTIAL = [hex2bgr(c) for c in ("#f0f6fe", "#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b")]


def ramp(t, stops):
    t = np.clip(t, 0, 1) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    f = (t - i)[..., None]
    return np.array(stops)[i] * (1 - f) + np.array(stops)[i + 1] * f


def save(rgb, name):
    h, w = rgb.shape[:2]
    cv2.imwrite(os.path.join(FIG, name), cv2.resize(rgb.astype(np.uint8), (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST))


def div(field, lim, name):
    save(ramp((field / lim + 1) / 2, DIVERGING), name)


def seq(field, lim, name):
    save(ramp(field / lim, SEQUENTIAL), name)


def css_gradient(stops):
    return "linear-gradient(90deg," + ",".join("#%02x%02x%02x" % (int(s[2]), int(s[1]), int(s[0])) for s in stops) + ")"


limits = {"diverging_css": css_gradient(DIVERGING), "sequential_css": css_gradient(SEQUENTIAL)}
for name in ("typical", "hard"):
    d = np.load(os.path.join(HERE, "fields", f"{name}.npz"))
    a, b, g, p, flow, st, ps, fd = d["a"], d["b"], d["g"], d["p"], d["flow"], d["st"], d["ps"], d["fromdisp"]
    save(np.repeat(a[..., None], 3, 2), f"{name}_ref.png")
    save(np.repeat(b[..., None], 3, 2), f"{name}_def.png")
    L = float(np.ceil(np.abs(g).max() * 2) / 2)
    E = float(np.ceil(np.abs(p - g).max() * 2) / 2)
    for c, cn in enumerate("uv"):
        div(g[c], L, f"{name}_{cn}_gt.png")
        div(p[c], L, f"{name}_{cn}_pred.png")
        seq(np.abs(p[c] - g[c]), E, f"{name}_{cn}_err.png")
        div(flow[c], L, f"{name}_{cn}_flow.png")
    # Channels 1 and 2 are du/dcol and dv/drow, the two normal strains under the horizontal-u convention
    S = float(np.ceil(max(np.abs(st[1]).max(), np.abs(st[2]).max())))
    # Scaled to StrainNet's error: the differentiated field reaches 135% and would blank everything else
    SE = float(np.ceil(max(np.abs(ps[1] - st[1]).max(), np.abs(ps[2] - st[2]).max())))
    for c, cn in ((1, "exx"), (2, "eyy")):
        div(st[c], S, f"{name}_{cn}_gt.png")
        div(ps[c], S, f"{name}_{cn}_pred.png")
        div(fd[c], S, f"{name}_{cn}_fromdisp.png")
        seq(np.abs(ps[c] - st[c]), SE, f"{name}_{cn}_err.png")
        seq(np.abs(fd[c] - st[c]), SE, f"{name}_{cn}_fromdisp_err.png")
    limits[name] = dict(disp=L, disp_err=E, strain=S, strain_err=SE)
json.dump(limits, open(os.path.join(HERE, "figure_limits.json"), "w"), indent=1)
print(limits)
