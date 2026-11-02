"""Run the FIXED notebooks from the repo in fresh namespaces, then check behaviour, not just the absence of a crash.

The only harness change: lists are cut right after the split cell and EPOCH is lowered, so training takes seconds.
Works in a temp folder, so nothing lands in the repo.
"""
import glob, json, os, shutil, sys, tempfile, warnings
import numpy as np, scipy.io as sio, torch

REPO, HERE = sys.argv[1], os.path.dirname(os.path.abspath(__file__))
DATA, PEEL = os.path.join(HERE, "data"), os.path.join(HERE, "peel_work")
checks = []


def check(name, ok, detail=""):
    checks.append(dict(check=name, ok=bool(ok), detail=str(detail)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def run(nb, ns, after=None, sub=()):
    cells = json.load(open(os.path.join(REPO, nb), encoding="utf-8"))["cells"]
    for i, c in enumerate(cells):
        s = "".join(c["source"])
        if c["cell_type"] != "code" or not s.strip():
            continue
        for old, new in sub:
            s = s.replace(old, new)
        stdout, sys.stdout = sys.stdout, open(os.devnull, "w")
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                exec(compile(s, f"{nb}:cell-{i}", "exec"), ns)
        finally:
            sys.stdout.close()
            sys.stdout = stdout
        ns.setdefault("_warnings", []).extend(str(x.message)[:80] for x in w if x.category in (FutureWarning, DeprecationWarning))
        if after and i in after:
            after[i](ns)


def link_data(work):
    for f in ("imgs3", "gt3"):
        os.symlink(os.path.join(DATA, f), os.path.join(work, f), target_is_directory=True) if False else None
    shutil.copytree(os.path.join(DATA, "imgs3"), os.path.join(work, "imgs3"), ignore=lambda d, n: [x for x in n if int(x.split("_")[2]) > 400])
    shutil.copytree(os.path.join(DATA, "gt3"), os.path.join(work, "gt3"), ignore=lambda d, n: [x for x in n if int(x.split("_")[2].split(".")[0]) > 400])
    for f in ("train_dataset.txt", "validation_dataset.txt"):
        shutil.copy(os.path.join(REPO, f), work)


def cut(ns):
    tr, te = ns["train_set"], ns["test_set"]
    check("train and validation splits share no image", not (set(tr) & set(te)), f"{len(tr)} train, {len(te)} validation from 400 files on disk")
    ns["train_set"], ns["test_set"] = tr[:12], te[:2]


work = tempfile.mkdtemp(prefix="verify_")
link_data(work)
os.chdir(work)

print("\nDisplacementNet_train.ipynb")
ns = {"__name__": "__main__"}
run("DisplacementNet_train.ipynb", ns, {4: cut}, [("EPOCH = 200 ", "EPOCH = 10  "), ("milestones=[100]", "milestones=[5]")])
check("ran every cell", True)
check("learning rate dropped 100x at the milestone", abs(ns["optimizer"].param_groups[0]["lr"] - 1e-5) < 1e-12, ns["optimizer"].param_groups[0]["lr"])
check("checkpoint saved inside results/model/", glob.glob("results/model/param_all_2_9_*.pth"), os.listdir("results/model"))
logs = glob.glob("results/log/log_*.txt")
vals = glob.glob("results/log/validation_*.txt")
check("training log inside results/log/ with 10 rows", logs and len(open(logs[0]).readlines()) == 11)
check("validation log has 10 epochs", vals and len(open(vals[0]).readlines()) == 12)
check("nothing written beside results/log and results/model", sorted(os.listdir("results")) == ["log", "model"], os.listdir("results"))
check("output shape matches label", ns["output"].shape[1:] == ns["gt"].shape[1:], tuple(ns["output"].shape))
check("no deprecation warnings", not ns.get("_warnings"), ns.get("_warnings", [])[:2])
disp_state = ns["fcn"].state_dict()

print("\nStrainNet_train.ipynb")
shutil.rmtree("results")
ns = {"__name__": "__main__"}
run("StrainNet_train.ipynb", ns, {4: cut}, [("EPOCH = 200 ", "EPOCH = 10  "), ("milestones=[100]", "milestones=[5]")])
check("ran every cell", True)
check("learning rate dropped 100x at the milestone", abs(ns["optimizer"].param_groups[0]["lr"] - 1e-5) < 1e-12, ns["optimizer"].param_groups[0]["lr"])
check("checkpoint saved inside results/model/", glob.glob("results/model/param_all_strain2_9_*.pth"), os.listdir("results/model"))
check("training and validation logs written", len(glob.glob("results/log/log*")) == 1 and len(glob.glob("results/log/validation*")) == 1, os.listdir("results/log"))
vals = glob.glob("results/log/validation*")
check("validation log has 10 epochs", vals and len(open(vals[0]).readlines()) == 12)
check("labels reach the model as float32 at 64x64", ns["gt"].dtype == torch.float32 and ns["output"].shape[1:] == ns["gt"].shape[1:], tuple(ns["gt"].shape))
loaded = ns["MyDataset"](dataset=ns["test_set"])[0]
print("    (a sample label loads with shape", tuple(loaded[1].shape), ")")
check("no deprecation warnings", not ns.get("_warnings"), ns.get("_warnings", [])[:2])
strain_state = ns["fcn"].state_dict()

# Run notebooks: a 3-frame sequence with a 0.2%/frame vertical stretch, and the weights just trained
os.makedirs("images")
for k in range(3):
    shutil.copy(os.path.join(PEEL, f"frame_{k}.png"), os.path.join("images", f"frame_{k}.png"))
torch.save(disp_state, "results/model/pretrained_displacementnet")
torch.save(strain_state, "results/model/pretrained_strainnet")

print("\nDiaplacement_run.ipynb")
ns = {"__name__": "__main__"}
run("Diaplacement_run.ipynb", ns)
check("ran every cell", True)
check("model is in eval mode", not ns["fcn"].training)
files = sorted(os.listdir("results/displacement"))
check("one position and one field per frame pair", len(files) == 4, files)
d = sio.loadmat("results/displacement/result_2_disp_0_x.mat")["disp_1_x"]
p = sio.loadmat("results/displacement/result_2_position1.mat")["position"]
pn, h0, w0, hnew, wnew = ns["predict_np"], ns["h0"], ns["w0"], ns["hnew"], ns["wnew"]
check("saved u is rescaled by w0/wnew", np.allclose(d[0], pn[0] / 2 * w0 / wnew, rtol=1e-4), f"w0/wnew={w0}/{wnew}")
check("saved v is rescaled by h0/hnew", np.allclose(d[1], pn[1] / 2 * h0 / hnew, rtol=1e-4), f"h0/hnew={h0}/{hnew}")
check("field shape equals the resized ROI", d.shape == (2, hnew, wnew), d.shape)
check("position is six integers", p.shape == (1, 6) and np.issubdtype(p.dtype, np.integer), p.tolist())
check("no deprecation warnings", not ns.get("_warnings"), ns.get("_warnings", [])[:2])

print("\nStrainNet_run.ipynb")
ns = {"__name__": "__main__"}
run("StrainNet_run.ipynb", ns)
check("ran every cell", True)
check("model is in eval mode", not ns["fcn"].training)
files = sorted(os.listdir("results/strain"))
check("one strain file per frame pair", len(files) == 2, files)
s = sio.loadmat("results/strain/2strain.mat")["strain"]
check("strain has 4 channels at half the ROI size", s.shape == (4, ns["hnew"] // 2, ns["wnew"] // 2), s.shape)
check("no deprecation warnings", not ns.get("_warnings"), ns.get("_warnings", [])[:2])

os.chdir(HERE)
shutil.rmtree(work, ignore_errors=True)
fails = [c for c in checks if not c["ok"]]
print(f"\n{len(checks) - len(fails)} of {len(checks)} checks passed")
json.dump(checks, open(os.path.join(HERE, "verify.json"), "w"), indent=1)
sys.exit(1 if fails else 0)
