"""Time one training step of DisplacementNet under different CPU settings (same maths, different speed)."""
import sys, time, warnings
import torch, torch.nn as nn
from common import build_disp

warnings.simplefilter("ignore")
threads, cl = int(sys.argv[1]), sys.argv[2] == "cl"
torch.set_num_threads(threads)
torch.manual_seed(0)
fcn = build_disp()
x, y = torch.rand(12, 2, 128, 128), torch.rand(12, 2, 128, 128)
if cl:
    fcn = fcn.to(memory_format=torch.channels_last)
    x = x.contiguous(memory_format=torch.channels_last)
opt = torch.optim.Adam(fcn.parameters(), lr=1e-3)
fcn.train()
times = []
for k in range(5):
    t = time.time()
    loss = nn.functional.mse_loss(fcn(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
    times.append(time.time() - t)
print(f"threads={threads} channels_last={cl}: {sum(times[1:]) / 4:.2f} s/step (first {times[0]:.2f})", flush=True)
