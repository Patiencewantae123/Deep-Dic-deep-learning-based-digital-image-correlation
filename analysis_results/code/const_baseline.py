"""Best single-constant prediction per strain channel on the test pairs, against StrainNet (eval_long.json).

    python const_baseline.py <eval_long.json> [strain.pth]

A network that has only learned the dataset's average beats predicting zero but not this baseline. Also records
how much StrainNet's output varies within and between test images, against the labels.
"""
import json, os, sys
import numpy as np, scipy.io as sio, torch
from PIL import Image
from scipy.interpolate import RectBivariateSpline
from torchvision.transforms import ToTensor
from common import HERE, build_strain

torch.set_num_threads(int(os.environ.get("THREADS", "10")))
OUT = os.path.abspath(sys.argv[1])
W = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(HERE, "runs", "long_strain_phase2.pth")
os.chdir(os.path.join(HERE, "data"))
grid, fine = np.arange(0, 256, 1), np.arange(1.5, 257.5, 4)
net = build_strain()
net.load_state_dict(torch.load(W, map_location="cpu"))
net.eval()
labels, preds = [], []
with torch.no_grad():
    for i in range(3301, 3601):
        a = ToTensor()(Image.open(f"imgs3/train_image_{i}_1.png").resize((128, 128)))
        b = ToTensor()(Image.open(f"imgs3/train_image_{i}_2.png").resize((128, 128)))
        gt = sio.loadmat(f"gt3/train_image_{i}.mat")["Disp_field_1"].astype(np.float64)
        u0, u1 = np.gradient(gt[:, :, 0] * 100)
        v0, v1 = np.gradient(gt[:, :, 1] * 100)
        labels.append(np.stack([RectBivariateSpline(grid, grid, c)(fine, fine) for c in (u0, u1, v0, v1)]))
        preds.append(net(torch.cat((a, b), 0)[None].float())[0].numpy())
y, p = np.stack(labels), np.stack(preds)            # (300, 4, 64, 64)
med = np.median(y, axis=(0, 2, 3), keepdims=True)
res = json.load(open(OUT))
res["strain_const"] = dict(
    label_mean_pct=y.mean(axis=(0, 2, 3)).tolist(),
    best_constant_pct=med.ravel().tolist(),
    best_constant_error_pct=np.abs(y - med).mean(axis=(0, 2, 3)).tolist(),
    net_error_pct=np.abs(y - p).mean(axis=(0, 2, 3)).tolist(),
    net_mean_pct=p.mean(axis=(0, 2, 3)).tolist(),
    # spread of the per-image average, and of values within an image, for labels and predictions
    label_between_std=y.mean(axis=(2, 3)).std(axis=0).tolist(), net_between_std=p.mean(axis=(2, 3)).std(axis=0).tolist(),
    label_within_std=y.std(axis=(2, 3)).mean(axis=0).tolist(), net_within_std=p.std(axis=(2, 3)).mean(axis=0).tolist(),
    corr=[float(np.corrcoef(y[:, c].ravel(), p[:, c].ravel())[0, 1]) for c in range(4)])
json.dump(res, open(OUT, "w"), indent=1)
for k, v in res["strain_const"].items():
    print(f"{k:24s}", np.round(v, 3))
