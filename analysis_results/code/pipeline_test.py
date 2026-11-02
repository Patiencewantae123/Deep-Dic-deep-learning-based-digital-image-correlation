"""Run the repo's Diaplacement_run (cell 4) and StrainNet_run (cell 3) on roi_test.py's 40-frame tensile test and
compare the accumulated displacement and strain with the known answer.

    python pipeline_test.py flow <out_json>                          optical flow stands in for both networks
    python pipeline_test.py net <out_json> <disp.pth> <strain.pth>   the trained networks

The flow mode checks the notebooks' bookkeeping (accumulation, resizing, strain components, signs) without network
error; the net mode measures what the trained networks give.
"""
import json, os, shutil, sys, warnings
import cv2, numpy as np, scipy.io as sio, torch, torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor
from common import HERE, build_disp, build_strain
import roi_test as T

warnings.simplefilter("ignore")
torch.set_num_threads(2)
MODE, OUT_JSON = sys.argv[1], sys.argv[2]
REPO = os.path.abspath(os.path.join(os.path.dirname(OUT_JSON), "..", ".."))
WORK = os.path.join(HERE, "roi_work", f"pipeline_{MODE}")


def flow_of(imgs):
    a, b = ((imgs[0, k].numpy() * 255).round().astype(np.uint8) for k in (0, 1))
    return cv2.calcOpticalFlowFarneback(a, b, None, .5, 4, 21, 5, 7, 1.5, 0)


class FlowDisp:
    """Network units: displacement x2, at the resized ROI's resolution, like DisplacementNet."""
    training = False

    def __call__(self, imgs):
        return torch.from_numpy(2 * np.moveaxis(flow_of(imgs), -1, 0))[None]


class FlowStrain:
    """StrainNet's units: 100 x [du/drow, du/dcol, dv/drow, dv/dcol] at half resolution."""
    training = False

    def __call__(self, imgs):
        fl = flow_of(imgs)
        g = np.stack([*np.gradient(fl[..., 0]), *np.gradient(fl[..., 1])]) * 100
        return F.avg_pool2d(torch.from_numpy(g.astype(np.float32))[None], 2)


def cell(nb, idx, subs):
    with open(os.path.join(REPO, nb), encoding="utf-8") as f:
        src = "".join(json.load(f)["cells"][idx]["source"])
    for a, b in subs:
        assert a in src, a
        src = src.replace(a, b)
    return src


if __name__ == "__main__":
    if MODE == "flow":
        disp_net, strain_net = FlowDisp(), FlowStrain()
    else:
        disp_net, strain_net = build_disp(), build_strain()
        disp_net.load_state_dict(torch.load(sys.argv[3], map_location="cpu"))
        strain_net.load_state_dict(torch.load(sys.argv[4], map_location="cpu"))
        disp_net.eval()
        strain_net.eval()
    shutil.rmtree(WORK, ignore_errors=True)
    dres, sres = os.path.join(WORK, "displacement"), os.path.join(WORK, "strain") + "/"
    base = dict(np=np, torch=torch, os=os, cv2=cv2, ToTensor=ToTensor, loadmat=sio.loadmat, savemat=sio.savemat,
                device=torch.device("cpu"), default_loader=lambda p: Image.open(p).convert("L"))
    exec(cell("Diaplacement_run.ipynb", 4, [("path_img = 'images'", f"path_img = {T.FRAMES!r}"),
                                            ("results_path = 'results/displacement'", f"results_path = {dres!r}")]),
         dict(base, fcn=disp_net))
    exec(cell("StrainNet_run.ipynb", 3, [("displacement_result_path = 'results/displacement'", f"displacement_result_path = {dres!r}"),
                                         ("path_img = 'images'", f"path_img = {T.FRAMES!r}"),
                                         ("result_path = 'results/strain/'", f"result_path = {sres!r}")]),
         dict(base, fcn=strain_net))

    n = T.N_FRAMES
    d = sio.loadmat(os.path.join(dres, f"result_{n}_disp_0_x.mat"))["disp_acc"]
    s = sio.loadmat(os.path.join(sres, f"{n}strain.mat"))["strain_acc"] * 100
    x1, x2, y1, y2 = T.ROI0
    h, w = x2 - x1, y2 - y1
    # truth on the starting ROI's grid: pixel (r, c) is the material point that started at (x1 + r, y1 + c)
    XX, YY = np.meshgrid(y1 + np.arange(w, dtype=float), x1 + np.arange(h, dtype=float))
    u_true, v_true = T.field(n, XX, YY)
    # incremental strains are measured in the current frame, so they add up to sum(de / (1 + e)), not n * de
    eps = T.D_EPS * np.arange(n)
    eyy_true = float(np.sum(T.D_EPS / (1 + eps)) * 100)
    exx_true = float(-np.sum(T.NU * T.D_EPS / (1 - T.NU * eps)) * 100)
    inner = (slice(h // 10, -h // 10), slice(w // 10, -w // 10))  # edges of a flow field are less reliable
    res = dict(mode=MODE, frames=n,
               disp_acc_mean_abs_error_px=dict(u=float(np.abs(d[0] - u_true).mean()), v=float(np.abs(d[1] - v_true).mean())),
               disp_acc_range_true_px=dict(u=[float(u_true.min()), float(u_true.max())], v=[float(v_true.min()), float(v_true.max())]),
               strain_acc_pct=dict(exx=float(s[0][inner].mean()), eyy=float(s[1][inner].mean()), exy=float(s[2][inner].mean())),
               strain_acc_true_pct=dict(exx=exx_true, eyy=eyy_true, exy=0.0),
               shapes=dict(disp_acc=list(d.shape), strain_acc=list(s.shape),
                           strain_xx_yy_xy=list(sio.loadmat(os.path.join(sres, f"{n}strain.mat"))["strain_xx_yy_xy"].shape)))
    json.dump(res, open(OUT_JSON, "w"), indent=1)
    print(json.dumps(res, indent=1))
