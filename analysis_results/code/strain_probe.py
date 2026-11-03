"""StrainNet's channel means on one frame pair of roi_test.py's tensile test, for different input crops.

    python strain_probe.py <out_json> [strain.pth]

Truth per frame pair: du/dcol = -0.12%, dv/drow = +0.4% (x100 in the network's units), cross terms 0.
"""
import json, os, sys
import numpy as np, torch
from PIL import Image
from torchvision.transforms import ToTensor
from common import HERE, build_strain
import roi_test as T

torch.set_num_threads(10)
net = build_strain()
W = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(HERE, "runs", "long_strain_phase2.pth")
net.load_state_dict(torch.load(W, map_location="cpu"))
net.eval()
a, b = (Image.open(os.path.join(T.FRAMES, f"frame_{k:03d}.png")).convert("L") for k in (10, 11))
x1, x2, y1, y2 = T.ROI0


def run(box, size, norm=True):
    t = [ToTensor()(im.crop(box).resize(size)) for im in (a, b)]
    if norm:
        t = [x / x.max() for x in t]
    with torch.no_grad():
        return net(torch.cat(t, 0)[None].float())[0].numpy().mean(axis=(1, 2))


near32 = lambda n: max(32, int(n / 32 + .5) * 32)          # the run notebooks' resize (paper, Section 2.1)
HN, WN = near32(x2 - x1), near32(y2 - y1)
NB = f"notebook: ROI {x2 - x1}x{y2 - y1} -> {HN}x{WN}"
cases = {NB: ((y1, x1, y2, x2), (WN, HN)),
         "same, no max normalisation": ((y1, x1, y2, x2), (WN, HN), False),
         "128x128 crop, native scale": ((y1, x1 + 100, y1 + 128, x1 + 228), (128, 128)),
         "192x192 crop, native scale": ((y1, x1 + 100, y1 + 192, x1 + 292), (192, 192)),
         "256x256 crop -> 128x128 (training's resize)": ((y1 - 25, x1 + 50, y1 + 231, x1 + 306), (128, 128))}
res = dict(weights=os.path.basename(W), frames=[10, 11], notebook_case=NB, channels=["du/drow", "du/dcol", "dv/drow", "dv/dcol"],
           truth_pct=[0, -T.NU * T.D_EPS * 100, T.D_EPS * 100, 0], mean_pct={k: run(*c).tolist() for k, c in cases.items()})
json.dump(res, open(sys.argv[1], "w"), indent=1)
for k, v in res["mean_pct"].items():
    print(f"{k:44s}", np.round(v, 2))
