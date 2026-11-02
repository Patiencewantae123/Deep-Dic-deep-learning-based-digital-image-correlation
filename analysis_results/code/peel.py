"""Find the full failure chain: run the committed cells, record the first failure, add only the fix for it, repeat.

Harness-only edits (not fixes) shrink the data lists and EPOCH so that each attempt takes seconds:
they are applied in every stage, including stage 0, and change no logic.
"""
import json, os, re, shutil, sys, time, traceback, warnings
import cv2, numpy as np, torch
import gen_data
from common import REPO, HERE, build_disp, build_strain

warnings.simplefilter("ignore")
DATA = os.path.join(HERE, "data")
PEEL = os.path.join(HERE, "peel_work")


def cells(nb):
    with open(os.path.join(REPO, nb), encoding="utf-8") as f:
        return ["".join(c["source"]) if c["cell_type"] == "code" else "" for c in json.load(f)["cells"]]


def attempt(srcs):
    """Run cells in order in one namespace. Returns (cell, error type, message) or None if all pass."""
    ns = {"__name__": "__main__"}
    out = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        for i, s in enumerate(srcs):
            if not s.strip():
                continue
            try:
                exec(compile(s, f"cell-{i}", "exec"), ns)
            except BaseException as e:
                return i, type(e).__name__, str(e).split("\n")[0][:190]
        return None
    finally:
        sys.stdout.close()
        sys.stdout = out


def peel(name, harness, fixes):
    print(f"\n### {name}")
    base = [harness(i, s) for i, s in enumerate(cells(name))]
    rows = []
    active = []
    for stage, (label, fix) in enumerate(fixes + [(None, None)]):
        if os.path.isdir("results"):
            shutil.rmtree("results")
        srcs = list(base)
        for f in active:
            srcs = f(srcs)
        t = time.time()
        res = attempt(srcs)
        dt = time.time() - t
        cell, err, msg = res if res else (None, None, None)
        print(f" stage {stage}: " + (f"cell-{cell} {err}: {msg}" if res else f"PASS ({dt:.0f}s)"), flush=True)
        rows.append(dict(stage=stage, fixes_so_far=[f[0] for f in fixes[:stage]], cell=cell, error=err, message=msg, secs=round(dt, 1)))
        if label is None:
            break
        active.append(fix)
    return rows


def sub(cell, old, new, count=-1):
    def f(srcs):
        assert old in srcs[cell], old
        srcs = list(srcs)
        srcs[cell] = srcs[cell].replace(old, new, count)
        return srcs
    return f


def chain(*fs):
    def f(srcs):
        for g in fs:
            srcs = g(srcs)
        return srcs
    return f


def shrink(src):
    for a, b in (("range(4000)", "range(2)"), ("range(36000)", "range(12)"), ("EPOCH = 100 ", "EPOCH = 10  "),
                 ("len(test_set) < 4000", "len(test_set) < 2")):
        src = src.replace(a, b)
    return src


# ---------- DisplacementNet_train: cell-9 is several pasted copies of its own loop; keep exactly one of them
def repair_disp_cell9(srcs):
    s = srcs[9]
    head = s[:s.index("for epoch in range(EPOCH):")]
    rec_start = s.index("        # Recreate train_loader") if "        # Recreate train_loader" in s else None
    lines = s.split("\n")
    first_list = s[s.index("for epoch in range(EPOCH):"):s.index("        # CPU FIX: Removed .cuda()")]
    i = next(k for k, l in enumerate(lines) if l.strip() == "# Recreate train_loader for this epoch (assuming shuffling or updates)")
    j = next(k for k, l in enumerate(lines) if k > i and l.strip().startswith("print(f'Validation error epoch"))
    body = "\n".join(l[8:] for l in lines[i:j + 1])
    srcs = list(srcs)
    srcs[9] = head + first_list + body + "\n"
    srcs[9] = shrink(srcs[9])
    # Harness cap: this loop re-lists the folder itself, so range(36000) above does not bound it
    srcs[9] = srcs[9].replace("train_data = MyDataset(dataset=train_set)", "train_data = MyDataset(dataset=train_set[:12])")
    return srcs


os.chdir(DATA)
all_rows = {}
all_rows["DisplacementNet_train"] = peel("DisplacementNet_train.ipynb", lambda i, s: shrink(s), [
    ("Rebuild cell-9 as one training loop", repair_disp_cell9),
    ("Load images as 1-channel: .convert('L')", sub(5, ".convert('RGB')", ".convert('L')")),
])
print(" artefacts:", sorted(os.listdir("results")) if os.path.isdir("results") else None)
disp_files = sorted(os.listdir("results")) if os.path.isdir("results") else []
disp_model = sorted(os.listdir("results/model")) if os.path.isdir("results/model") else []
print(" results/model/ contents:", disp_model)

# ---------- StrainNet_train
loop_old_start = "    dataset_path = '.'  # or set this"
SPLINE = """f = interpolate.RectBivariateSpline(y, x, dudx); dudx_ = f(ynew, xnew)
        f = interpolate.RectBivariateSpline(y, x, dudy); dudy_ = f(ynew, xnew)
        f = interpolate.RectBivariateSpline(y, x, dvdx); dvdx_ = f(ynew, xnew)
        f = interpolate.RectBivariateSpline(y, x, dvdy); dvdy_ = f(ynew, xnew)
"""


def restore_strain_loop(srcs):
    s = srcs[9]
    a = s.index(loop_old_start)
    b = s.index("        # Move data to the device")
    srcs = list(srcs)
    srcs[9] = s[:a] + "    for step, (img, gt) in enumerate(train_loader):\n" + s[b:]
    return srcs


def replace_interp2d(srcs):
    s = srcs[5]
    a = s.index("f = interpolate.interp2d(x, y, dudx")
    b = s.index("st = np.stack([dudx_")
    srcs = list(srcs)
    srcs[5] = s[:a] + SPLINE + "        " + s[b:]
    return srcs


all_rows["StrainNet_train"] = peel("StrainNet_train.ipynb", lambda i, s: shrink(s), [
    ("Define log_result / model_result", sub(8, 'print(f"Directories', 'log_result = str(root_result / "log"); model_result = str(root_result / "model")\nprint(f"Directories')),
    ("Remove ':' from the timestamp (invalid in Windows filenames)", sub(7, "%H:%M:%S", "%H_%M_%S")),
    ("Define device", sub(8, 'print(f"Directories', 'device = "cuda" if torch.cuda.is_available() else "cpu"\nprint(f"Directories')),
    ("Restore the deleted `for step, (img, gt) in enumerate(train_loader):` line", restore_strain_loop),
    ("Set dataset_path = '.' in cell-4 (cell-9's rebuild never reaches the loader)", sub(4, "dataset_path = ''", "dataset_path = '.'")),
    ("Import scipy.interpolate", sub(5, "import scipy.io as sio", "import scipy.io as sio\nfrom scipy import interpolate")),
    ("Replace interp2d (removed from SciPy) with RectBivariateSpline", replace_interp2d),
    ("Cast validation gt to float and drop the extra dimension", sub(9, "gt = gt.unsqueeze(1).to(device)", "gt = gt.float().to(device)")),
])
print(" artefacts:", sorted(os.listdir("results/log")), sorted(os.listdir("results/model")))

# ---------- run notebooks: need weights and a small frame sequence
os.makedirs(PEEL, exist_ok=True)
torch.save(build_disp().state_dict(), os.path.join(PEEL, "pretrained_displacementnet"))
torch.save(build_strain().state_dict(), os.path.join(PEEL, "pretrained_strainnet"))
rng = np.random.default_rng(5)
base = np.vstack([gen_data.speckle(rng)[0], gen_data.speckle(rng)[0]])
for k in range(3):
    h, w = base.shape
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    frame = cv2.remap(base, X, 511.5 + (Y - 511.5) / (1 + .002 * k), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    cv2.imwrite(os.path.join(PEEL, f"frame_{k}.png"), (np.clip(frame, 0, 1) * 255).astype(np.uint8))
os.chdir(PEEL)
P = PEEL.replace("\\", "/")
all_rows["Diaplacement_run"] = peel("Diaplacement_run.ipynb", lambda i, s: s, [
    ("Move the model to CPU (no GPU here)", sub(3, "fcn = fcn.cuda()", "fcn = fcn.cpu()")),
    ("Define model_result", sub(3, "fnet = FeatureResNet()", f"model_result = '{P}/'\nfnet = FeatureResNet()")),
    ("Load weights with the correct device", sub(3, "'pretrained_displacementnet'))", "'pretrained_displacementnet', map_location='cpu'))")),
    ("Define img_num and result paths", chain(sub(4, "path_img = ''", f"path_img = '{P}'\nimg_num = 3"), sub(4, "results_path = ''", f"results_path = '{P}'"))),
    ("Replace placeholder image names with real paths", chain(sub(4, "'first_image'", "path_img + '/frame_0.png'"), sub(4, "'next_img'", "path_img + f'/frame_{i}.png'"))),
    ("Remove torch.cuda.FloatTensor", sub(4, "imgs.type(torch.cuda.FloatTensor)", "imgs.float()")),
    ("Remove .cuda() on the input", sub(4, "Variable(imgs).cuda()", "imgs")),
    ("Replace np.int (removed from NumPy) with int", sub(4, "np.int(", "int(")),
])
print(" artefacts:", sorted(f for f in os.listdir(".") if f.endswith(".mat")))
all_rows["StrainNet_run"] = peel("StrainNet_run.ipynb", lambda i, s: s, [
    ("Move the model to CPU (no GPU here)", sub(2, "fcn = fcn.cuda()", "fcn = fcn.cpu()")),
    ("Point at the weights file", sub(2, "torch.load('PATH_TO_PRETRAINED' + 'pretrained_strainnet')", f"torch.load('{P}/pretrained_strainnet', map_location='cpu')")),
    ("Define num_img and paths", chain(sub(3, "displacement_result_path = ''", f"displacement_result_path = '{P}'\nnum_img = 3"), sub(3, "result_path = ''", f"result_path = '{P}/'"))),
    ("Read the file name Diaplacement_run actually writes", sub(3, "'/result_'+str(i)+'position.mat'", "'/result_'+str(i)+'_position1.mat'")),
    ("Replace placeholder image names with real paths", chain(sub(3, "'first_img'", f"'{P}/frame_0.png'"), sub(3, "'second_img'", f"'{P}/frame_' + str(i) + '.png'"))),
    ("Use one input variable (imgs2 is built but imgs is used)", chain(sub(3, "imgs2 = torch.cat(", "imgs = torch.cat("), sub(3, "imgs2 = imgs.unsqueeze(0)", "imgs = imgs.unsqueeze(0)"), sub(3, "imgs2 = imgs.type(torch.cuda.FloatTensor)", "imgs = imgs.type(torch.cuda.FloatTensor)"), sub(3, "imgs2 = Variable(imgs).cuda()", "imgs = Variable(imgs).cuda()"))),
    ("Remove torch.cuda.FloatTensor", sub(3, "imgs.type(torch.cuda.FloatTensor)", "imgs.float()")),
    ("Remove .cuda() on the input", sub(3, "Variable(imgs).cuda()", "imgs")),
])
print(" artefacts:", sorted(f for f in os.listdir(".") if f.endswith(".mat")))
all_rows["_artefacts"] = dict(disp_results=disp_files, disp_model=disp_model)
json.dump(all_rows, open(os.path.join(HERE, "peel.json"), "w"), indent=1)
