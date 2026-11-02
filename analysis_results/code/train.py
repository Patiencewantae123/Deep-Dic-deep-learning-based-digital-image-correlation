"""Time-boxed training with the committed networks and the notebooks' MyDataset classes (patched only where they crash).

Phase 1 = the notebook configuration: Adam, lr 1e-3, batch 12, MSE, no schedule.
Phase 2 = the paper's step: same run continued at lr 1e-5 (the notebooks have no such step).
Split: 1-3000 train, 3001-3300 validation, 3301-3600 test; all from distinct speckle patterns.
"""
import csv, json, os, sys, time, warnings
import numpy as np, torch, torch.nn as nn
import torch.utils.data as data_utils
from common import *
from common import _cell

warnings.simplefilter("ignore")
task, minutes1, minutes2 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "runs")
os.makedirs(OUT, exist_ok=True)
os.chdir(DATA)


def dataset_class(task):
    ns = {}
    exec("import numpy as np, torch\nimport torch.utils.data as data_utils\nfrom torchvision.transforms import ToTensor\n"
         "from PIL import Image\nfrom scipy import interpolate\n", ns)
    exec("def default_loader(path):\n    return Image.open(path)", ns)
    if task == "disp":
        src = _cell("DisplacementNet_train.ipynb", 5).replace(".convert('RGB')", ".convert('L')")
    else:
        src = _cell("StrainNet_train.ipynb", 5)
        a, b = src.index("f = interpolate.interp2d(x, y, dudx"), src.index("st = np.stack([dudx_")
        src = src[:a] + "".join(f"{n}_ = interpolate.RectBivariateSpline(y, x, {n})(ynew, xnew)\n        "
                                for n in ("dudx", "dudy", "dvdx", "dvdy")) + src[b:]
    exec(src, ns)
    return ns["MyDataset"]


def triples(ids):
    return [(f"./imgs3/train_image_{i}_1.png", f"./imgs3/train_image_{i}_2.png", f"./gt3/train_image_{i}.mat") for i in ids]


torch.manual_seed(0)
np.random.seed(0)
MyDataset = dataset_class(task)
fcn = build_disp() if task == "disp" else build_strain()
train_loader = data_utils.DataLoader(MyDataset(dataset=triples(range(1, 3001))), batch_size=12, shuffle=True, num_workers=0, drop_last=True)
val_loader = data_utils.DataLoader(MyDataset(dataset=triples(range(3001, 3101))), batch_size=1)
optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
loss_func = nn.MSELoss()


def validate():
    fcn.eval()
    mse = mae = 0.0
    with torch.no_grad():
        for img, gt in val_loader:
            out = fcn(img.float())
            mse += loss_func(out, gt.float()).item()
            mae += (out - gt.float()).abs().mean().item()
    fcn.train()
    n = len(val_loader)
    return mse / n, mae / n


log = open(os.path.join(OUT, f"{task}_steps.csv"), "w", newline="")
vlog = open(os.path.join(OUT, f"{task}_val.csv"), "w", newline="")
lw, vw = csv.writer(log), csv.writer(vlog)
lw.writerow(["step", "phase", "epoch", "loss", "elapsed_s"])
vw.writerow(["step", "phase", "val_mse", "val_mae", "elapsed_s"])
t0 = time.time()
step = 0
data_time = model_time = 0.0
fcn.train()


def run_phase(phase, minutes, lr):
    global step, data_time, model_time
    for g in optimizer.param_groups:
        g["lr"] = lr
    end = time.time() + minutes * 60
    epoch = 0
    mse, mae = validate()
    vw.writerow([step, phase, mse, mae, round(time.time() - t0, 1)])
    vlog.flush()
    print(f"[{task}] phase {phase} start: val mse {mse:.4f} mae {mae:.4f}", flush=True)
    while time.time() < end:
        t_d = time.time()
        for img, gt in train_loader:
            t_m = time.time()
            data_time += t_m - t_d
            output = fcn(img.float())
            loss = loss_func(output, gt.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model_time += time.time() - t_m
            step += 1
            lw.writerow([step, phase, epoch, loss.item(), round(time.time() - t0, 1)])
            if step % 60 == 0:
                log.flush()
                mse, mae = validate()
                vw.writerow([step, phase, mse, mae, round(time.time() - t0, 1)])
                vlog.flush()
                print(f"[{task}] step {step} ep {epoch} {time.time() - t0:.0f}s train {loss.item():.4f} val mse {mse:.4f} mae {mae:.4f}", flush=True)
            t_d = time.time()
            if time.time() >= end:
                break
        epoch += 1
    mse, mae = validate()
    vw.writerow([step, phase, mse, mae, round(time.time() - t0, 1)])
    vlog.flush()
    torch.save(fcn.state_dict(), os.path.join(OUT, f"{task}_phase{phase}.pth"))
    print(f"[{task}] phase {phase} end: step {step} val mse {mse:.4f} mae {mae:.4f}", flush=True)


run_phase(1, minutes1, 1e-3)
run_phase(2, minutes2, 1e-5)
json.dump(dict(steps=step, data_s_per_step=data_time / step, model_s_per_step=model_time / step), open(os.path.join(OUT, f"{task}_timing.json"), "w"))
print(f"[{task}] done, {data_time / step:.3f}s data + {model_time / step:.2f}s model per step", flush=True)
