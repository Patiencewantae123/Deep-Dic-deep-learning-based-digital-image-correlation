"""Continue the time-boxed runs of train.py for longer, to see how far more training closes the gap to the paper.

Same networks, dataset classes, split (1-3000 train, 3001-3100 validation), batch 12, MSE and Adam as train.py.
Starts from train.py's final weights, runs lr 1e-3 then lr 1e-5 (the paper's drop), and saves a checkpoint
every 30 minutes so stopping early keeps the progress. channels_last and the thread count only change speed.

    python train_long.py disp 6 120 25 <log_dir>
    python train_long.py strain 4 120 25 <log_dir>

A further run can continue from any saved weights, under its own name so earlier logs are kept:

    python train_long.py disp 6 420 45 <log_dir> max runs/long_disp_phase2.pth 3145
"""
import csv, os, sys, time, warnings
import numpy as np, torch, torch.nn as nn
import torch.utils.data as data_utils
from common import *
from train_common import dataset_class, triples

warnings.simplefilter("ignore")
task, threads, minutes1, minutes2 = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
torch.set_num_threads(threads)
REPO_OUT = os.path.abspath(sys.argv[5]) if len(sys.argv) > 5 else os.path.join(HERE, "long")
NAME = sys.argv[6] if len(sys.argv) > 6 else "long"
RESUME = os.path.abspath(sys.argv[7]) if len(sys.argv) > 7 else None
RUNS = os.path.join(HERE, "runs")
os.makedirs(REPO_OUT, exist_ok=True)
os.chdir(os.path.join(HERE, "data"))
VAL_EVERY = 120

torch.manual_seed(1)
np.random.seed(1)
MyDataset = dataset_class(task)
fcn = build_disp() if task == "disp" else build_strain()
fcn.load_state_dict(torch.load(RESUME or os.path.join(RUNS, f"{task}_phase2.pth")))
fcn = fcn.to(memory_format=torch.channels_last)
start_step = int(sys.argv[8]) if len(sys.argv) > 8 else sum(1 for _ in open(os.path.join(RUNS, f"{task}_steps.csv"))) - 1
train_loader = data_utils.DataLoader(MyDataset(dataset=triples(range(1, 3001))), batch_size=12, shuffle=True, num_workers=0, drop_last=True)
val_loader = data_utils.DataLoader(MyDataset(dataset=triples(range(3001, 3101))), batch_size=1)
optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
loss_func = nn.MSELoss()
cl = lambda t: t.float().contiguous(memory_format=torch.channels_last)


def validate():
    fcn.eval()
    mse = mae = 0.0
    with torch.no_grad():
        for img, gt in val_loader:
            out = fcn(cl(img))
            mse += loss_func(out, gt.float()).item()
            mae += (out - gt.float()).abs().mean().item()
    fcn.train()
    n = len(val_loader)
    return mse / n, mae / n


log = open(os.path.join(REPO_OUT, f"{NAME}_{task}_steps.csv"), "w", newline="")
vlog = open(os.path.join(REPO_OUT, f"{NAME}_{task}_val.csv"), "w", newline="")
lw, vw = csv.writer(log), csv.writer(vlog)
lw.writerow(["step", "phase", "epoch", "loss", "elapsed_s"])
vw.writerow(["step", "phase", "val_mse", "val_mae", "elapsed_s"])
t0 = time.time()
step = start_step
last_save = t0


def record(phase):
    mse, mae = validate()
    vw.writerow([step, phase, mse, mae, round(time.time() - t0, 1)])
    vlog.flush()
    log.flush()
    print(f"[{task}] step {step} phase {phase} {time.time() - t0:.0f}s val mse {mse:.4f} mae {mae:.4f}", flush=True)


def save(name):
    path = os.path.join(RUNS, f"{NAME}_{task}_{name}.pth")
    torch.save(fcn.state_dict(), path + ".tmp")
    os.replace(path + ".tmp", path)


def run_phase(phase, minutes, lr):
    global step, last_save
    for g in optimizer.param_groups:
        g["lr"] = lr
    end = time.time() + minutes * 60
    epoch = 0
    record(phase)
    while time.time() < end:
        for img, gt in train_loader:
            output = fcn(cl(img))
            loss = loss_func(output, gt.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            lw.writerow([step, phase, epoch, loss.item(), round(time.time() - t0, 1)])
            if step % VAL_EVERY == 0:
                record(phase)
            if time.time() - last_save > 1800:
                save("latest")
                last_save = time.time()
            if time.time() >= end:
                break
        epoch += 1
    record(phase)
    save(f"phase{phase}")


run_phase(1, minutes1, 1e-3)
run_phase(2, minutes2, 1e-5)
save("latest")
print(f"[{task}] done: {step - start_step} more steps, {step} in total, {(time.time() - t0) / 60:.0f} min", flush=True)
