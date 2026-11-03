"""Two of the paper's experimental checks, repeated on synthetic image pairs made like the training data.

    python paper_checks.py <out_json> [disp.pth] [strain.pth]

Noise floor (paper Section 3.3, Table 5): 21 stationary pairs, the same speckle image twice with independent
camera noise. The right answer is zero everywhere; the paper reports DisplacementNet 0.0094 px and StrainNet
0.0073% (mean over pairs of each pair's mean magnitude), with standard deviations 0.0065 px and 0.0045%.

Rigid translation (paper Figure 12): pairs moved by a known uniform vertical shift and not deformed. The paper
found DisplacementNet non-uniform here, up to 0.275 px from VIC-2D, because its training set has no pure rigid
motion; gen_data.py has none either. Strain should be zero.

Shifts and errors are in pixels of the 128x128 network input. One network is loaded at a time to save memory.
"""
import json, os, sys
import cv2, numpy as np, torch
from PIL import Image
from torchvision.transforms import ToTensor
from common import HERE, build_disp, build_strain
import gen_data as G

torch.set_num_threads(int(os.environ.get("THREADS", "3")))
OUT = os.path.abspath(sys.argv[1])
RUNS = os.path.join(HERE, "runs")
DISP_W = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(RUNS, "long_disp_phase2.pth")
STRAIN_W = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else os.path.join(RUNS, "long_strain_phase2.pth")
SHIFTS = [0.25, 0.5, 1.0, 1.5, 2.0]          # px at 128x128
N_NOISE, N_SHIFT = 21, 10                     # pairs per test


def pair(seed, shift):
    """Like gen_data.make: 512 speckle, warp, 256 crop, noise 0.001 on each image, then the loaders' 128 resize."""
    rng = np.random.default_rng(seed)
    img, kind = G.speckle(rng)
    zero = np.zeros_like(img)
    warped = G.warp(img, zero, zero + 2 * shift) if shift else img      # 1 px at 128 = 2 px at 512
    x0, y0 = rng.integers(G.MARGIN, G.SIZE - G.CROP - G.MARGIN + 1, 2)
    out = []
    for a in (img, warped):
        a = np.clip(a[y0:y0 + G.CROP, x0:x0 + G.CROP] + rng.normal(0, np.sqrt(.001), (G.CROP, G.CROP)), 0, 1)
        out.append(Image.fromarray((a * 255).round().astype(np.uint8)).resize((128, 128)))
    return out, kind


def tensor(a, b):
    return torch.cat((ToTensor()(a), ToTensor()(b)), 0)[None].float()


def flow(a, b):
    return np.moveaxis(cv2.calcOpticalFlowFarneback(np.array(a), np.array(b), None, .5, 4, 21, 5, 7, 1.5, 0), -1, 0)


noise_pairs = [pair(90000 + k, 0)[0] for k in range(N_NOISE)]
shift_pairs = {s: [pair(91000 + 100 * j + k, s)[0] for k in range(N_SHIFT)] for j, s in enumerate(SHIFTS)}
ms = lambda v: dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)))
res = dict(weights=dict(disp=os.path.basename(DISP_W), strain=os.path.basename(STRAIN_W)),
           paper=dict(noise_disp_px=[0.0094, 0.0065], noise_strain_pct=[0.0073, 0.0045], translation_max_diff_px=0.275))

# DisplacementNet (output in 256-grid px, so /2 for px at 128)
net = build_disp()
net.load_state_dict(torch.load(DISP_W, map_location="cpu"))
net.eval()
with torch.no_grad():
    pred = lambda a, b: net(tensor(a, b))[0].numpy() / 2
    nd = [np.hypot(*pred(a, b)).mean() for a, b in noise_pairs]
    nf = [np.hypot(*flow(a, b)).mean() for a, b in noise_pairs]
    res["noise_floor"] = dict(pairs=N_NOISE, disp_net_px=ms(nd), disp_farneback_px=ms(nf))
    rows = []
    for s, prs in shift_pairs.items():
        p = [pred(a, b) for a, b in prs]
        f = [flow(a, b) for a, b in prs]
        inner = (slice(8, -8), slice(8, -8))        # leave out the border, where the warp reflects
        rows.append(dict(shift_px=s,
                         net_mean_v=float(np.mean([q[1][inner].mean() for q in p])),
                         net_mean_abs_error=float(np.mean([np.abs(q[1][inner] - s).mean() + np.abs(q[0][inner]).mean() for q in p]) / 1),
                         net_spread_v=float(np.mean([q[1][inner].max() - q[1][inner].min() for q in p])),
                         flow_mean_v=float(np.mean([q[1][inner].mean() for q in f])),
                         flow_mean_abs_error=float(np.mean([np.abs(q[1][inner] - s).mean() + np.abs(q[0][inner]).mean() for q in f]))))
    res["translation"] = rows
del net

# StrainNet (output: 100 x [du/drow, du/dcol, dv/drow, dv/dcol]); report |exx|, |eyy|, |exy| averaged, in %
net = build_strain()
net.load_state_dict(torch.load(STRAIN_W, map_location="cpu"))
net.eval()


def strain_mag(a, b):
    with torch.no_grad():
        o = net(tensor(a, b))[0].numpy()
    return np.mean([np.abs(o[1]).mean(), np.abs(o[2]).mean(), np.abs(.5 * (o[0] + o[3])).mean()])


res["noise_floor"]["strain_net_pct"] = ms([strain_mag(a, b) for a, b in noise_pairs])
for row, (s, prs) in zip(res["translation"], shift_pairs.items()):
    row["strain_net_mean_abs_pct"] = float(np.mean([strain_mag(a, b) for a, b in prs]))
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
