"""Test-set errors on 500 unseen pairs (3301-3800) for every model and baseline, with 95% confidence intervals.

    python eval_test500.py <out_json>

Same samples, units and metrics as evaluate.py and eval_long.py, extended from 300 to 500 pairs (run
gen_data.py 3800 3601 first). "mean" averages each sample's mean error, "mean_max" each sample's largest error;
"ci95" is the half-width of the 95% interval for "mean" (1.96 standard errors over samples). "paired" gives the
mean per-sample difference between two methods, with its own interval, which is tighter than comparing two means.
The first 300 pairs must reproduce eval.json and eval_long.json, which checks the regenerated data.
"""
import json, os, sys, warnings
import cv2, numpy as np, scipy.io as sio, torch, torch.nn.functional as F
from PIL import Image
from scipy.interpolate import RectBivariateSpline
from torchvision.transforms import ToTensor
from common import HERE, build_disp, build_strain

warnings.simplefilter("ignore")
torch.set_num_threads(int(os.environ.get("THREADS", "10")))
OUT = os.path.abspath(sys.argv[1])
RUNS = os.path.join(HERE, "runs")
os.chdir(os.path.join(HERE, "data"))
TEST = list(range(3301, 3801))
grid, fine = np.arange(0, 256, 1), np.arange(1.5, 257.5, 4)
meta = {r[0]: r for r in json.load(open("meta.json"))}


def load(i):
    a = Image.open(f"imgs3/train_image_{i}_1.png").resize((128, 128))
    b = Image.open(f"imgs3/train_image_{i}_2.png").resize((128, 128))
    gt = sio.loadmat(f"gt3/train_image_{i}.mat")["Disp_field_1"].astype(np.float64)
    return ToTensor()(a), ToTensor()(b), gt, np.array(a), np.array(b)


def strain_gt(gt):
    u0, u1 = np.gradient(gt[:, :, 0] * 100)
    v0, v1 = np.gradient(gt[:, :, 1] * 100)
    return np.stack([RectBivariateSpline(grid, grid, c)(fine, fine) for c in (u0, u1, v0, v1)])


def disp_to_strain(pred256):
    u0, u1 = np.gradient(pred256[0] * 100 / 2)
    v0, v1 = np.gradient(pred256[1] * 100 / 2)
    return F.avg_pool2d(torch.from_numpy(np.stack([u0, u1, v0, v1]).astype(np.float32))[None], 2)[0].numpy()


def model(build, name):
    m = build()
    m.load_state_dict(torch.load(os.path.join(RUNS, name), map_location="cpu"))
    return m.eval()


def predict(m, a, b):
    with torch.no_grad():
        return m(torch.cat((a, b), 0)[None].float())[0].numpy()


def stats(means, maxes):
    m = np.asarray(means)
    return dict(mean=float(m.mean()), mean_max=float(np.mean(maxes)), ci95=float(1.96 * m.std(ddof=1) / np.sqrt(len(m))))


D = {k: model(build_disp, f) for k, f in (("net_p1", "disp_phase1.pth"), ("net_p2", "disp_phase2.pth"), ("long", "long_disp_phase2.pth"))}
S = {k: model(build_strain, f) for k, f in (("strainnet_p1", "strain_phase1.pth"), ("strainnet_p2", "strain_phase2.pth"), ("long", "long_strain_phase2.pth"))}
mean_u = np.mean([load(i)[2].mean((0, 1)) for i in range(1, 501)], axis=0)   # training-set mean field, as evaluate.py

e = {k: ([], []) for k in ("zero", "mean", "farneback", "net_p1", "net_p2", "net_u", "net_v", "long", "long_u", "long_v")}
s = {k: ([], []) for k in ("zero", "strainnet_p1", "strainnet_p2", "long", "from_dispnet", "from_long_dispnet", "from_farneback")}
labels, long_strain, kinds = [], [], []


def add(d, k, err):
    d[k][0].append(float(err.mean()))
    d[k][1].append(float(err.max()))


for n, i in enumerate(TEST):
    a, b, gt, a8, b8 = load(i)
    g = np.moveaxis(gt[::2, ::2, :], -1, 0) / 2
    p = {k: predict(m, a, b) / 2 for k, m in D.items()}
    fl = np.moveaxis(cv2.calcOpticalFlowFarneback(a8, b8, None, .5, 4, 21, 5, 7, 1.5, 0), -1, 0)
    add(e, "zero", np.abs(g))
    add(e, "mean", np.abs(g - (mean_u / 2)[:, None, None]))
    add(e, "farneback", np.abs(fl - g))
    for k in D:
        add(e, k, np.abs(p[k] - g))
    add(e, "net_u", np.abs(p["net_p2"][0] - g[0]))
    add(e, "net_v", np.abs(p["net_p2"][1] - g[1]))
    add(e, "long_u", np.abs(p["long"][0] - g[0]))
    add(e, "long_v", np.abs(p["long"][1] - g[1]))
    st = strain_gt(gt)
    ps = {k: predict(m, a, b) for k, m in S.items()}
    add(s, "zero", np.abs(st))
    for k in S:
        add(s, k, np.abs(ps[k] - st))
    add(s, "from_dispnet", np.abs(disp_to_strain(p["net_p2"] * 2) - st))
    add(s, "from_long_dispnet", np.abs(disp_to_strain(p["long"] * 2) - st))
    add(s, "from_farneback", np.abs(disp_to_strain(fl * 2) - st))
    labels.append(st)
    long_strain.append(ps["long"])
    kinds.append(meta[i][1])
    if (n + 1) % 100 == 0:
        print(f"{n + 1} pairs", flush=True)

y, q = np.stack(labels), np.stack(long_strain)
med = np.median(y, axis=(0, 2, 3), keepdims=True)
res = dict(samples=len(TEST), ids=[TEST[0], TEST[-1]],
           disp_px={k: stats(*v) for k, v in e.items()}, strain_pct={k: stats(*v) for k, v in s.items()})
const = np.abs(y - med).mean(axis=(2, 3))   # per sample and channel
res["strain_pct"]["best_constant"] = stats(const.mean(axis=1), np.abs(y - med).max(axis=(1, 2, 3)))
res["strain_channels_long"] = dict(net_pct=np.abs(y - q).mean(axis=(0, 2, 3)).tolist(), best_constant_pct=const.mean(axis=0).tolist(),
                                   corr=[float(np.corrcoef(y[:, c].ravel(), q[:, c].ravel())[0, 1]) for c in range(4)])


def paired(d, k1, k2):
    diff = np.asarray(d[k1][0]) - np.asarray(d[k2][0])
    return dict(mean=float(diff.mean()), ci95=float(1.96 * diff.std(ddof=1) / np.sqrt(len(diff))), first_better_share=float(np.mean(diff < 0)))


res["paired"] = {"disp long - net_p2": paired(e, "long", "net_p2"), "disp long - farneback": paired(e, "long", "farneback"),
                 "strain long - zero": paired(s, "long", "zero"), "strain long - strainnet_p2": paired(s, "long", "strainnet_p2")}
res["per_kind_disp_long"] = {k: [float(np.mean([m for m, kk in zip(e["long"][0], kinds) if kk == k])), kinds.count(k)] for k in sorted(set(kinds))}
# first 300 pairs against the earlier 300-pair results
E, LE = json.load(open(os.path.join(HERE, "eval.json"))), json.load(open(os.path.join(HERE, "eval_long.json")))
first300 = lambda d, k: float(np.mean(d[k][0][:300]))
res["check_first300"] = dict(
    disp_net_p2=[first300(e, "net_p2"), E["test"]["disp_px"]["net_p2"]["mean"]], disp_farneback=[first300(e, "farneback"), E["test"]["disp_px"]["farneback"]["mean"]],
    disp_long=[first300(e, "long"), LE["disp_px"]["mean"]], strain_long=[first300(s, "long"), LE["strain_pct"]["mean"]],
    strain_zero=[first300(s, "zero"), E["test"]["strain_pct"]["zero"]["mean"]])
json.dump(res, open(OUT, "w"), indent=1)
for grp in ("disp_px", "strain_pct"):
    for k, v in res[grp].items():
        print(f"{grp:10s} {k:18s} {v['mean']:.4f} ± {v['ci95']:.4f}   mean-max {v['mean_max']:.3f}")
print(json.dumps({k: res[k] for k in ("strain_channels_long", "paired", "per_kind_disp_long", "check_first300")}, indent=1))
