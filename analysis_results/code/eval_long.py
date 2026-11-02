"""Test-set error of the longer runs (train_long.py), with evaluate.py's samples, units and metrics.

    python eval_long.py <out_json> [disp.pth] [strain.pth]

Displacement in pixels of the 128x128 input, strain in %; "mean" averages each sample's mean error and
"mean_max" each sample's largest error, as in the paper's Table 3.
"""
import json, os, sys, warnings
import numpy as np, scipy.io as sio, torch
from PIL import Image
from scipy.interpolate import RectBivariateSpline
from torchvision.transforms import ToTensor
from common import HERE, build_disp, build_strain

warnings.simplefilter("ignore")
torch.set_num_threads(int(os.environ.get("THREADS", "10")))
OUT = os.path.abspath(sys.argv[1])
RUNS = os.path.join(HERE, "runs")
DISP_W = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(RUNS, "long_disp_phase2.pth")
STRAIN_W = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else os.path.join(RUNS, "long_strain_phase2.pth")
os.chdir(os.path.join(HERE, "data"))
TEST = list(range(3301, 3601))
grid, fine = np.arange(0, 256, 1), np.arange(1.5, 257.5, 4)


def load(i):
    a = Image.open(f"imgs3/train_image_{i}_1.png").resize((128, 128))
    b = Image.open(f"imgs3/train_image_{i}_2.png").resize((128, 128))
    gt = sio.loadmat(f"gt3/train_image_{i}.mat")["Disp_field_1"].astype(np.float64)
    return ToTensor()(a), ToTensor()(b), gt


def strain_gt(gt):
    u0, u1 = np.gradient(gt[:, :, 0] * 100)
    v0, v1 = np.gradient(gt[:, :, 1] * 100)
    return np.stack([RectBivariateSpline(grid, grid, c)(fine, fine) for c in (u0, u1, v0, v1)])


def stats(errs):
    return dict(mean=float(np.mean([e.mean() for e in errs])), mean_max=float(np.mean([e.max() for e in errs])))


def model(build, path):
    m = build()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


disp, strain = model(build_disp, DISP_W), model(build_strain, STRAIN_W)
e_d, e_s, e_strain_ch = [], [], [[], [], [], []]
with torch.no_grad():
    for i in TEST:
        a, b, gt = load(i)
        x = torch.cat((a, b), 0)[None].float()
        g = np.moveaxis(gt[::2, ::2, :], -1, 0) / 2
        e_d.append(np.abs(disp(x)[0].numpy() / 2 - g))
        es = np.abs(strain(x)[0].numpy() - strain_gt(gt))
        e_s.append(es)
        for c in range(4):
            e_strain_ch[c].append(float(es[c].mean()))
res = dict(weights=dict(disp=os.path.basename(DISP_W), strain=os.path.basename(STRAIN_W)), samples=len(TEST),
           disp_px=stats(e_d), strain_pct=stats(e_s), strain_channels_pct=[float(np.mean(c)) for c in e_strain_ch])
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
