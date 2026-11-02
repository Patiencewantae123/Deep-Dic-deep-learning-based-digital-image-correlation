"""Does Diaplacement_run's ROI follow the material? A synthetic tensile test with known motion answers it.

Frames: one speckle pattern (made like gen_data.py, at twice the size, then halved as training does), stretched
vertically about a fixed lower grip at row Y_GRIP, with Poisson contraction and a slow sideways drift. Material at x in
frame 0 sits at x + U_k(x) in frame k, the same forward convention as the paper (Section 2.2) and gen_data.py.

Runs, all with the same trained DisplacementNet:
  before     - cell 4 of Diaplacement_run.ipynb as committed (ROI = start - accumulated mean shift)
  sign_only  - the same cell with the four minus signs turned into plus signs
  fixed      - cell 4 of the repo's current Diaplacement_run.ipynb: plus sign, and each ROI edge follows the
               displacement measured along that edge, so the ROI can stretch (the paper's description)
  fixed_flow - the same edge update with Farneback optical flow in place of the network, to check the test itself
Truth is where the material inside the starting ROI actually is.

    python roi_test.py <weights.pth> <out_json> <committed_notebook> [threads]
"""
import json, os, shutil, sys, warnings
import cv2, numpy as np, scipy.io as sio, torch
from PIL import Image
from torchvision.transforms import ToTensor
from common import HERE, build_disp

warnings.simplefilter("ignore")
WORK = os.path.join(HERE, "roi_work")
FRAMES = os.path.join(WORK, "images")
H, W = 1000, 300                       # frame size in pixels, the scale the network sees
Y_GRIP, X_MID = 950.0, 120.0           # fixed grip row, specimen centre column
N_FRAMES, D_EPS, NU, DRIFT = 40, 0.004, 0.3, 0.4   # 40 increments of 0.4% strain, 0.4 px/frame sideways drift
ROI0 = (440, 806, 25, 215)             # x1, x2, y1, y2 = top row, bottom row, left col, right col (the notebook's)


def field(k, X, Y):
    """Total displacement from frame 0 to frame k at frame-0 positions, in frame pixels."""
    e = k * D_EPS
    return -NU * e * (X - X_MID) + DRIFT * k, e * (Y - Y_GRIP)


def speckle(rng, h, w):
    img = np.zeros((h, w), np.float32)
    S = 1 << 4
    n = int(3600 * h * w / 512 ** 2)
    for cx, cy, a, b, g in zip(rng.uniform(0, w - 1, n), rng.uniform(0, h - 1, n), rng.uniform(1.2, 6.8, n),
                               rng.uniform(1.2, 6.8, n), rng.uniform(.08, .98, n)):
        cv2.ellipse(img, (int(cx * S), int(cy * S)), (int(a * S), int(b * S)), 0, 0, 360, float(g), -1, cv2.LINE_8, 4)
    return cv2.GaussianBlur(img, (0, 0), 1.0)


def make_frames():
    rng = np.random.default_rng(7)
    h2, w2 = 2 * H, 2 * W
    ref = speckle(rng, h2, w2)
    Xt, Yt = np.meshgrid(np.arange(w2, dtype=np.float32), np.arange(h2, dtype=np.float32))
    os.makedirs(FRAMES, exist_ok=True)
    for k in range(N_FRAMES + 1):
        # forward warp on the doubled grid: find the frame-0 source of each target pixel by fixed-point iteration
        sx, sy = Xt.copy(), Yt.copy()
        for _ in range(8):
            u, v = field(k, sx / 2, sy / 2)
            sx, sy = Xt - 2 * u, Yt - 2 * v
        img = cv2.remap(ref, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        img = np.clip(img + rng.normal(0, np.sqrt(.001), img.shape), 0, 1)
        Image.fromarray((img * 255).round().astype(np.uint8)).resize((W, H)).save(os.path.join(FRAMES, f"frame_{k:03d}.png"))


def truth(k):
    """Where the starting ROI's edges are in frame k."""
    x1, x2, y1, y2 = ROI0
    mid_r, mid_c = (x1 + x2) / 2, (y1 + y2) / 2
    return (x1 + field(k, mid_c, x1)[1], x2 + field(k, mid_c, x2)[1],
            y1 + field(k, y1, mid_r)[0], y2 + field(k, y2, mid_r)[0])


def notebook_cell(name, nb, sign="as_written"):
    with open(nb, encoding="utf-8") as f:
        src = "".join(json.load(f)["cells"][4]["source"])
    src = src.replace("path_img = 'images'", f"path_img = {FRAMES!r}")
    out = os.path.join(WORK, f"results_{name}")
    shutil.rmtree(out, ignore_errors=True)
    src = src.replace("results_path = 'results/displacement'", f"results_path = {out!r}")
    if sign == "plus":
        for a in ("x100", "y100", "x200", "y200"):
            n = src.count(f"round({a}-d")
            assert n == 1, (a, n)
            src = src.replace(f"round({a}-d", f"round({a}+d")
    ns = dict(np=np, torch=torch, os=os, cv2=cv2, ToTensor=ToTensor, fcn=fcn, device=torch.device("cpu"),
              default_loader=lambda p: Image.open(p).convert("L"))
    exec(src, ns)
    rows = [list(ROI0)]
    for i in range(1, N_FRAMES + 1):
        p = sio.loadmat(os.path.join(out, f"result_{i}_position1.mat"))["position"].ravel()
        rows.append([int(p[0]), int(p[1]), int(p[2]), int(p[3])])
    return rows


def predict(img1, img2, roi):
    x1, x2, y1, y2 = [int(round(c)) for c in roi]
    h0, w0 = x2 - x1, y2 - y1
    hnew, wnew = (h0 // 32 + 1) * 32, (w0 // 32 + 1) * 32
    crops = [ToTensor()(im.crop((y1, x1, y2, x2)).resize((wnew, hnew))) for im in (img1, img2)]
    imgs = torch.cat([c / c.max() for c in crops], 0)[None].float()
    with torch.no_grad():
        p = fcn(imgs)[0].numpy() / 2
    return p[0] * w0 / wnew, p[1] * h0 / hnew       # u, v in frame pixels


def predict_flow(img1, img2, roi):
    x1, x2, y1, y2 = [int(round(c)) for c in roi]
    a, b = (np.array(im.crop((y1, x1, y2, x2))) for im in (img1, img2))
    flow = cv2.calcOpticalFlowFarneback(a, b, None, .5, 4, 21, 5, 7, 1.5, 0)
    return flow[..., 0], flow[..., 1]


def corners(predict=predict):
    """Each edge moves by the displacement measured along it (a 5% band), so the ROI stretches with the material."""
    roi = np.array(ROI0, float)
    rows = [list(ROI0)]
    load = lambda k: Image.open(os.path.join(FRAMES, f"frame_{k:03d}.png")).convert("L")
    for i in range(1, N_FRAMES + 1):
        u, v = predict(load(i - 1), load(i), roi)
        bh, bw = max(1, v.shape[0] // 20), max(1, u.shape[1] // 20)
        roi += [v[:bh].mean(), v[-bh:].mean(), u[:, :bw].mean(), u[:, -bw:].mean()]
        rows.append([int(round(c)) for c in roi])
    return rows


if __name__ == "__main__":
    WEIGHTS, OUT_JSON, COMMITTED_NB = sys.argv[1:4]
    torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 2)
    REPO = os.path.abspath(os.path.join(os.path.dirname(OUT_JSON), "..", ".."))
    if not os.path.exists(os.path.join(FRAMES, f"frame_{N_FRAMES:03d}.png")):
        make_frames()
    fcn = build_disp()
    fcn.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    fcn.eval()
    runs = {"before": notebook_cell("before", COMMITTED_NB), "sign_only": notebook_cell("sign_only", COMMITTED_NB, "plus"),
            "fixed": notebook_cell("fixed", os.path.join(REPO, "Diaplacement_run.ipynb")), "fixed_flow": corners(predict_flow)}
    true = [list(truth(k)) for k in range(N_FRAMES + 1)]
    res = dict(weights=os.path.basename(WEIGHTS), frames=N_FRAMES, roi0=ROI0, truth=true, runs=runs, summary={})
    names = ("top", "bottom", "left", "right")
    for name, rows in runs.items():
        err = np.array(rows, float) - np.array(true)
        centre = [(err[-1][0] + err[-1][1]) / 2, (err[-1][2] + err[-1][3]) / 2]
        res["summary"][name] = dict(final_roi=rows[-1], final_error_px=dict(zip(names, err[-1].round(1).tolist())),
                                    final_centre_error_px=dict(row=round(centre[0], 1), col=round(centre[1], 1)),
                                    worst_edge_error_px=float(np.abs(err).max().round(1)),
                                    final_height=rows[-1][1] - rows[-1][0], final_width=rows[-1][3] - rows[-1][2])
    t = true[-1]
    res["summary"]["truth"] = dict(final_roi=[round(c, 1) for c in t], final_height=round(t[1] - t[0], 1), final_width=round(t[3] - t[2], 1))
    json.dump(res, open(OUT_JSON, "w"), indent=1)
    for k, v in res["summary"].items():
        print(k, v)
