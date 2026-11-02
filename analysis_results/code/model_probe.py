"""Measure the committed networks: shapes, parameters, compute, valid input sizes, speed."""
import json, os, time, warnings
import torch, torch.nn as nn
from common import *

warnings.simplefilter("ignore")
out = {}


def flops_and_shapes(model, x):
    total, rows, hooks = [0], [], []

    def hook(mod, inp, outp):
        name = names[mod]
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
            k = mod.kernel_size[0] * mod.kernel_size[1]
            # MACs scale with the output grid for a conv, with the input grid for a transposed conv
            grid = outp if isinstance(mod, nn.Conv2d) else inp[0]
            total[0] += k * mod.in_channels * mod.out_channels * grid.shape[2] * grid.shape[3]
        if name in KEEP:
            rows.append((name, tuple(outp.shape)))

    names = {m: n for n, m in model.named_modules()}
    for m in model.modules():
        hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        model.eval()(x)
    for h in hooks:
        h.remove()
    return total[0], rows


KEEP = {"pretrained_net.conv_f", "pretrained_net.maxpool", "pretrained_net.layer1", "pretrained_net.layer2",
        "pretrained_net.layer3", "pretrained_net.layer4", "pretrained_net.conv_pre", "conv3_2", "conv4", "conv5",
        "conv6", "conv7", "conv8", "conv9", "convadd", "conv10"}

for label, make in (("DisplacementNet", build_disp), ("StrainNet", build_strain)):
    m = make()
    params = sum(p.numel() for p in m.parameters())
    unused = sum(p.numel() for n, p in m.named_parameters() if n.split(".")[1:2] in (["conv1"], ["fc"]))
    macs, rows = flops_and_shapes(m, torch.zeros(1, 2, 128, 128))
    print(f"\n== {label}: params={params:,}  never-used={unused:,}  MACs@128={macs / 1e9:.2f}G")
    for n, s in rows:
        print(f"   {n:<26}{s}")
    x = torch.rand(1, 2, 128, 128)
    with torch.no_grad():
        zero = m.eval()(x)
    print("   untrained output is exactly zero:", bool((zero == 0).all()))

    sizes = {}
    for h, w in [(128, 128), (256, 256), (160, 96), (384, 192), (64, 64), (32, 32), (100, 100), (128, 130), (96, 128)]:
        try:
            with torch.no_grad():
                sizes[f"{h}x{w}"] = list(m.eval()(torch.zeros(1, 2, h, w)).shape[2:])
        except RuntimeError as e:
            sizes[f"{h}x{w}"] = "ERROR: " + str(e).split("\n")[0][:70]
    print("   input -> output:", sizes)

    # conv10 is zero-initialised with no bias: what can the very first optimiser step reach?
    m.train()
    loss = nn.MSELoss()(m(torch.rand(2, 2, 128, 128)), torch.rand(2, m.conv10.out_channels, *(m(torch.zeros(2, 2, 128, 128)).shape[2:])))
    loss.backward()
    nonzero = [n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    none = [n for n, p in m.named_parameters() if p.grad is None]
    print(f"   step-1 params with non-zero grad: {nonzero}  | grad=None: {none}")

    out[label] = dict(params=params, unused=unused, macs=macs, shapes=rows, sizes=sizes, step1_nonzero=nonzero, grad_none=none)

# Train and run notebooks must describe the same network, or the pretrained weights cannot load
for a, b, name in ((build_disp, build_disp_run, "Displacement"), (build_strain, build_strain_run, "Strain")):
    ka, kb = a().state_dict(), b().state_dict()
    same = ka.keys() == kb.keys() and all(ka[k].shape == kb[k].shape for k in ka)
    print(f"{name}: train/run state_dicts identical: {same}")
    out[name + "_train_run_match"] = same

# Speed on this machine, with the notebook's batch size
torch.manual_seed(0)
for label, make, ch, sz in (("DisplacementNet", build_disp, 2, 128), ("StrainNet", build_strain, 4, 64)):
    m = make()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x, y = torch.rand(12, 2, 128, 128), torch.rand(12, ch, sz, sz)
    m.train()
    times = []
    for i in range(4):
        t = time.time()
        opt.zero_grad()
        nn.MSELoss()(m(x), y).backward()
        opt.step()
        times.append(time.time() - t)
    step = sum(times[1:]) / 3
    m.eval()
    with torch.no_grad():
        t = time.time()
        for _ in range(5):
            m(x[:1])
        single = (time.time() - t) / 5
    steps = 36000 // 12
    epoch_h = (steps * step + 4000 * single) / 3600
    print(f"{label}: train step (batch 12) {step:.2f}s | inference 1 image {single * 1000:.0f}ms | "
          f"1 epoch {epoch_h:.1f}h | 100 epochs {epoch_h * 100 / 24:.1f} days (model only, no data loading)")
    out[label].update(step_s=step, infer_s=single, epoch_h=epoch_h)

json.dump(out, open(os.path.join(HERE, "model_probe.json"), "w"), indent=1, default=str)
