"""Dataset statistics in the form of the paper's Appendix A (Figure A-1), for the synthetic training set.

    python dataset_stats.py <out_json> [first last]

Per sample: the largest displacement magnitude and its standard deviation over the image (px at 128x128),
and the largest equivalent strain and its standard deviation (%). Equivalent strain is
sqrt(2/3 (exx^2 + eyy^2 + 2 exy^2)) from the symmetric gradient, so rigid rotation does not count.
"""
import json, os, sys
import numpy as np, scipy.io as sio
from common import HERE

OUT = os.path.abspath(sys.argv[1])
first, last = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (1, 3000)
os.chdir(os.path.join(HERE, "data"))
rows = []
for i in range(first, last + 1):
    gt = sio.loadmat(f"gt3/train_image_{i}.mat")["Disp_field_1"].astype(np.float64)
    u, v = gt[::2, ::2, 0] / 2, gt[::2, ::2, 1] / 2            # px at 128x128
    mag = np.hypot(u, v)
    du_r, du_c = np.gradient(gt[:, :, 0])                         # on the 256 grid, dimensionless
    dv_r, dv_c = np.gradient(gt[:, :, 1])
    exx, eyy, exy = du_c, dv_r, .5 * (du_r + dv_c)
    eq = np.sqrt(2 / 3 * (exx ** 2 + eyy ** 2 + 2 * exy ** 2)) * 100
    rows.append((mag.max(), mag.std(), eq.max(), eq.std(), np.abs(exx).mean() * 100))
a = np.array(rows)
q = lambda c: {p: float(np.percentile(a[:, c], p)) for p in (5, 25, 50, 75, 95)}
res = dict(samples=len(a), max_disp_px=q(0), std_disp_px=q(1), max_eq_strain_pct=q(2), std_eq_strain_pct=q(3),
           mean_abs_exx_pct=q(4), hist_max_disp=np.histogram(a[:, 0], bins=np.arange(0, 12.5, .5))[0].tolist(),
           hist_max_strain=np.histogram(a[:, 2], bins=np.arange(0, 10.5, .5))[0].tolist())
json.dump(res, open(OUT, "w"), indent=1)
for k in ("max_disp_px", "std_disp_px", "max_eq_strain_pct", "std_eq_strain_pct", "mean_abs_exx_pct"):
    print(f"{k:20s}", {p: round(v, 2) for p, v in res[k].items()})
