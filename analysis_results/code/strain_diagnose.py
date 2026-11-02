"""Why did StrainNet stay at the zero-output baseline? Three independent checks.

1. Did the trained model learn anything at all? Correlation between prediction and label on unseen samples.
2. Are the labels right? Finite differences of a good displacement estimate must correlate with them.
3. Can the pipeline learn? Overfit 12 fixed training samples: fresh model, the notebook's optimiser, same batch every step.
   If loss collapses, the code path is sound and the run was just too short. If it does not, something is broken.
"""
import json, os, time, warnings
import cv2, numpy as np, scipy.io as sio, torch, torch.nn as nn
from PIL import Image
from scipy.interpolate import RectBivariateSpline
from torchvision.transforms import ToTensor
from common import HERE, build_strain

warnings.simplefilter("ignore")
RUNS, DATA = os.path.join(HERE, "runs"), os.path.join(HERE, "data")
os.chdir(DATA)
grid, fine = np.arange(0, 256, 1), np.arange(1.5, 257.5, 4)


def sample(i):
    a = ToTensor()(Image.open(f"imgs3/train_image_{i}_1.png").resize((128, 128)))
    b = ToTensor()(Image.open(f"imgs3/train_image_{i}_2.png").resize((128, 128)))
    gt = sio.loadmat(f"gt3/train_image_{i}.mat")["Disp_field_1"].astype(np.float64) * 100
    u0, u1 = np.gradient(gt[:, :, 0])
    v0, v1 = np.gradient(gt[:, :, 1])
    st = np.stack([RectBivariateSpline(grid, grid, c)(fine, fine) for c in (u0, u1, v0, v1)])
    return torch.cat((a, b), 0).float(), torch.from_numpy(st).float()


out = {}
m = build_strain()
m.load_state_dict(torch.load(os.path.join(RUNS, "strain_phase2.pth")))
m.eval()
P, L, F_ = [], [], []
for i in range(3301, 3351):
    x, y = sample(i)
    with torch.no_grad():
        P.append(m(x[None])[0].numpy())
    L.append(y.numpy())
    a8 = (x[0].numpy() * 255).round().astype(np.uint8)
    b8 = (x[1].numpy() * 255).round().astype(np.uint8)
    fl = cv2.calcOpticalFlowFarneback(a8, b8, None, .5, 4, 21, 5, 7, 1.5, 0)
    u0, u1 = np.gradient(fl[..., 0] * 100)
    v0, v1 = np.gradient(fl[..., 1] * 100)
    F_.append(np.stack([cv2.resize(c, (64, 64), interpolation=cv2.INTER_AREA) for c in (u0, u1, v0, v1)]))
P, L, F_ = np.array(P), np.array(L), np.array(F_)
corr = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
out["pred_label_corr"] = [corr(P[:, c], L[:, c]) for c in range(4)]
out["flow_label_corr"] = [corr(F_[:, c], L[:, c]) for c in range(4)]
out["pred_std"] = float(P.std())
out["label_std"] = float(L.std())
print("trained model vs label, correlation per channel:", np.round(out["pred_label_corr"], 3), f"| pred std {out['pred_std']:.3f}, label std {out['label_std']:.3f}", flush=True)
print("optical-flow strain vs label, correlation per channel:", np.round(out["flow_label_corr"], 3), flush=True)

torch.manual_seed(0)
batch = [sample(i) for i in range(1, 13)]
x = torch.stack([b[0] for b in batch])
y = torch.stack([b[1] for b in batch])
fresh = build_strain()
opt = torch.optim.Adam(fresh.parameters(), lr=1e-3)
loss_func = nn.MSELoss()
fresh.train()
hist, t0 = [], time.time()
for step in range(1, 241):
    loss = loss_func(fresh(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    hist.append(loss.item())
    if step % 30 == 0:
        print(f"overfit step {step} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
out["overfit_start"], out["overfit_end"] = hist[0], float(np.mean(hist[-10:]))
out["overfit_hist"] = hist
print(f"overfit: {hist[0]:.3f} -> {out['overfit_end']:.3f}  ({(1 - out['overfit_end'] / hist[0]) * 100:.1f}% lower)", flush=True)
json.dump(out, open(os.path.join(HERE, "strain_diagnose.json"), "w"), indent=1)
