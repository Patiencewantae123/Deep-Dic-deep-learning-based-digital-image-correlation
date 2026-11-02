"""Evaluate the trained models on the held-out test split and measure the impact of each silent bug.

Displacement is reported in pixels of the 128x128 network input (ground-truth units / 2, the same /2 Diaplacement_run uses).
Strain is reported in % (the notebooks' units), per channel and overall.
"""
import copy, json, os, warnings
import cv2, numpy as np, scipy.io as sio, torch, torch.nn.functional as F
from PIL import Image
from scipy.interpolate import RectBivariateSpline
from torchvision.transforms import ToTensor
from common import *

warnings.simplefilter("ignore")
DATA, RUNS = os.path.join(HERE, "data"), os.path.join(HERE, "runs")
os.chdir(DATA)
TEST = list(range(3301, 3601))
LEAK = list(range(1, 301))  # the notebooks' validation list is a subset of their training list
grid = np.arange(0, 256, 1)
fine = np.arange(1.5, 257.5, 4)
meta = {r[0]: r for r in json.load(open("meta.json"))}


def load(i):
    a = Image.open(f"imgs3/train_image_{i}_1.png").resize((128, 128))
    b = Image.open(f"imgs3/train_image_{i}_2.png").resize((128, 128))
    gt = sio.loadmat(f"gt3/train_image_{i}.mat")["Disp_field_1"].astype(np.float64)
    return ToTensor()(a), ToTensor()(b), gt, np.array(a), np.array(b)


def strain_gt(gt):
    """The notebook's label: 100 x np.gradient, resampled to 64x64, in its channel order."""
    u0, u1 = np.gradient(gt[:, :, 0] * 100)
    v0, v1 = np.gradient(gt[:, :, 1] * 100)
    return np.stack([RectBivariateSpline(grid, grid, c)(fine, fine) for c in (u0, u1, v0, v1)])


def disp_to_strain(pred256):
    """Differentiate a displacement field on the 128 grid (values in 256-grid px), mapping onto the 64x64 strain grid."""
    u0, u1 = np.gradient(pred256[0] * 100 / 2)
    v0, v1 = np.gradient(pred256[1] * 100 / 2)
    t = torch.from_numpy(np.stack([u0, u1, v0, v1]).astype(np.float32))[None]
    return F.avg_pool2d(t, 2)[0].numpy()


def stats(errs):
    """errs: list of per-sample absolute-error arrays. Paper metrics: mean of per-sample max, mean of per-sample mean."""
    return dict(mean=float(np.mean([e.mean() for e in errs])), mean_max=float(np.mean([e.max() for e in errs])))


disp = build_disp()
disp.load_state_dict(torch.load(os.path.join(RUNS, "disp_phase2.pth")))
disp.eval()
strain = build_strain()
strain.load_state_dict(torch.load(os.path.join(RUNS, "strain_phase2.pth")))
strain.eval()
disp1 = build_disp()
disp1.load_state_dict(torch.load(os.path.join(RUNS, "disp_phase1.pth")))
disp1.eval()
strain1 = build_strain()
strain1.load_state_dict(torch.load(os.path.join(RUNS, "strain_phase1.pth")))
strain1.eval()

# Training-set mean field is the best constant answer: a model that learned nothing scores this
mean_u = np.zeros((2,))
for i in range(1, 501):
    mean_u += load(i)[2].mean((0, 1))
mean_u /= 500

R = {}


def predict(model, a, b):
    with torch.no_grad():
        return model(torch.cat((a, b), 0)[None].float())[0].numpy()


def evaluate(ids, tag):
    e = {k: [] for k in ("zero", "mean", "farneback", "net_p1", "net_p2", "net_u", "net_v")}
    s = {k: [] for k in ("zero", "strainnet_p1", "strainnet_p2", "from_dispnet", "from_farneback")}
    per_kind, per_mag, ch = {}, [], [[], [], [], []]
    for i in ids:
        a, b, gt, a8, b8 = load(i)
        g = np.moveaxis(gt[::2, ::2, :], -1, 0) / 2
        p2, p1 = predict(disp, a, b) / 2, predict(disp1, a, b) / 2
        e["zero"].append(np.abs(g))
        e["mean"].append(np.abs(g - (mean_u / 2)[:, None, None]))
        e["net_p1"].append(np.abs(p1 - g))
        e["net_p2"].append(np.abs(p2 - g))
        e["net_u"].append(np.abs(p2[0] - g[0]))
        e["net_v"].append(np.abs(p2[1] - g[1]))
        flow = cv2.calcOpticalFlowFarneback(a8, b8, None, .5, 4, 21, 5, 7, 1.5, 0)
        fl = np.moveaxis(flow, -1, 0)
        e["farneback"].append(np.abs(fl - g))
        st = strain_gt(gt)
        s["zero"].append(np.abs(st))
        ps = predict(strain, a, b)
        s["strainnet_p2"].append(np.abs(ps - st))
        s["strainnet_p1"].append(np.abs(predict(strain1, a, b) - st))
        s["from_dispnet"].append(np.abs(disp_to_strain(p2 * 2) - st))
        s["from_farneback"].append(np.abs(disp_to_strain(fl * 2) - st))
        for c in range(4):
            ch[c].append(float(np.abs(ps[c] - st[c]).mean()))
        kind = meta[i][1]
        per_kind.setdefault(kind, []).append(float(np.abs(p2 - g).mean()))
        per_mag.append((float(np.abs(g).mean()), float(np.abs(p2 - g).mean())))
    out = dict(disp_px={k: stats(v) for k, v in e.items()}, strain_pct={k: stats(v) for k, v in s.items()},
               strain_channels=[float(np.mean(c)) for c in ch], per_kind={k: [float(np.mean(v)), len(v)] for k, v in per_kind.items()}, per_mag=per_mag)
    print(f"\n== {tag} ({len(ids)} samples)")
    for k, v in out["disp_px"].items():
        print(f"  disp  {k:<12} mean {v['mean']:.4f} px  mean-max {v['mean_max']:.3f} px")
    for k, v in out["strain_pct"].items():
        print(f"  strain {k:<15} mean {v['mean']:.4f} %  mean-max {v['mean_max']:.3f} %")
    print("  per speckle type:", {k: round(v[0], 3) for k, v in out["per_kind"].items()})
    return out


R["test"] = evaluate(TEST, "held-out test")
R["leaked"] = evaluate(LEAK, "training images used as validation (the notebook's leak)")


# ---- Silent bug: run notebooks never call fcn.eval(), and the module default is train mode
def mae(model, ids, prep=lambda a, b: (a, b)):
    v = []
    for i in ids:
        a, b, gt, *_ = load(i)
        g = np.moveaxis(gt[::2, ::2, :], -1, 0) / 2
        with torch.no_grad():
            p = model(torch.cat(prep(a, b), 0)[None].float())[0].numpy() / 2
        v.append(float(np.abs(p - g).mean()))
    return float(np.mean(v))


ids = TEST[:150]
base = mae(disp, ids)
m = copy.deepcopy(disp).train()
train_mode = mae(m, ids)  # also rewrites m's BatchNorm running stats, which is what the run loop does
m.eval()
after = mae(m, ids)
print(f"\n== eval() bug: eval mode {base:.4f} | train mode (as committed) {train_mode:.4f} | eval after 150 train-mode frames {after:.4f}")
R["eval_bug"] = dict(eval=base, train_mode=train_mode, polluted_eval=after)

# ---- Silent bug: inference normalises by the image max, training does not
norm_each = lambda a, b: (a / a.max(), b / b.max())  # Diaplacement_run
norm_first = lambda a, b: (a / a.max(), b / a.max())  # StrainNet_run
dim = lambda a, b: (a * .5, b * .5)  # a 50% under-exposed camera
dim_norm = lambda a, b: norm_each(*dim(a, b))
dim_second = lambda a, b: (a, b * .8)  # the light dims 20% between frames
R["preprocessing"] = dict(raw=base, norm_each=mae(disp, ids, norm_each), norm_first=mae(disp, ids, norm_first), dim_raw=mae(disp, ids, dim),
                          dim_norm=mae(disp, ids, dim_norm), dim_second_raw=mae(disp, ids, dim_second),
                          dim_second_norm_each=mae(disp, ids, lambda a, b: norm_each(*dim_second(a, b))),
                          dim_second_norm_first=mae(disp, ids, lambda a, b: norm_first(*dim_second(a, b))))
print("== preprocessing:", {k: round(v, 4) for k, v in R["preprocessing"].items()})

# ---- Label names: np.gradient returns d/d(row) first. A pure horizontal stretch u = 0.01 x makes this visible
yy, xx = np.mgrid[0:256, 0:256].astype(float)
f0, f1 = np.gradient(.01 * xx)
R["gradient_names"] = dict(named_dudx=float(np.abs(f0).max()), named_dudy=float(np.abs(f1).max()))
print("== u = 0.01x:  the channel named dudx =", R["gradient_names"]["named_dudx"], "  the channel named dudy =", R["gradient_names"]["named_dudy"])

# ---- Missing rescale: Diaplacement_run scales the ROI update by w0/wnew, the saved field by nothing
h0, w0 = 806 - 440, 215 - 25
hnew, wnew = (h0 // 32 + 1) * 32, (w0 // 32 + 1) * 32
R["rescale"] = dict(h0=h0, w0=w0, hnew=hnew, wnew=wnew, v_err_pct=(hnew / h0 - 1) * 100, u_err_pct=(wnew / w0 - 1) * 100)
print("== ROI rescale:", R["rescale"])

# ---- ROI tracker logic with a perfect predictor: only the loop's arithmetic is under test.
# Material moves +d px per frame. Compare 'first_image' as a fixed reference with consecutive pairs, and both sign choices.
d, frames, track = 3.0, 10, {}
for ref in ("fixed", "incremental"):
    for sign in (-1, 1):
        roi, acc, hist = 100.0, 0.0, []
        for n in range(1, frames + 1):
            seen = d * n if ref == "fixed" else d
            acc += seen
            roi = 100 + sign * acc
            hist.append(roi - (100 + d * n))
        track[f"{ref}_{'minus' if sign < 0 else 'plus'}"] = hist
R["roi_tracker"] = track
print("== ROI error after 10 frames:", {k: v[-1] for k, v in track.items()})

# ---- Fields for the report: pick the median sample and a hard one by error
scores = sorted((float(np.abs(predict(disp, *load(i)[:2]) / 2 - np.moveaxis(load(i)[2][::2, ::2, :], -1, 0) / 2).mean()), i) for i in TEST[:120])
picks = dict(typical=scores[len(scores) // 2][1], hard=scores[int(len(scores) * .95)][1])
os.makedirs(os.path.join(HERE, "fields"), exist_ok=True)
for name, i in picks.items():
    a, b, gt, a8, b8 = load(i)
    g = np.moveaxis(gt[::2, ::2, :], -1, 0) / 2
    p = predict(disp, a, b) / 2
    st, ps = strain_gt(gt), predict(strain, a, b)
    np.savez(os.path.join(HERE, "fields", f"{name}.npz"), a=a8, b=b8, g=g, p=p, st=st, ps=ps, fromdisp=disp_to_strain(p * 2),
             flow=np.moveaxis(cv2.calcOpticalFlowFarneback(a8, b8, None, .5, 4, 21, 5, 7, 1.5, 0), -1, 0))
R["picks"] = picks
print("== picks", picks)
json.dump(R, open(os.path.join(HERE, "eval.json"), "w"), indent=1)
