"""Execute each notebook's code cells, unmodified and in order, in a fresh namespace.

Every cell runs even after an earlier one fails, which mimics "Run All, continue on error".
Runs in a temp copy of the repo's list files, so nothing is written to the repo itself.
A cell that would train for real is bounded by a per-cell alarm thread.
"""
import json, os, sys, tempfile, traceback, warnings, shutil, threading, ctypes, time

REPO = sys.argv[1]
NOTEBOOKS = ["DisplacementNet_train.ipynb", "StrainNet_train.ipynb",
             "Diaplacement_run.ipynb", "StrainNet_run.ipynb"]
CELL_TIMEOUT = 60


def run_with_timeout(code, ns):
    """Run compiled code on this thread, raising TimeoutError into it after CELL_TIMEOUT."""
    tid = threading.get_ident()
    done = threading.Event()

    def alarm():
        if not done.wait(CELL_TIMEOUT):
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), ctypes.py_object(TimeoutError))

    threading.Thread(target=alarm, daemon=True).start()
    try:
        exec(code, ns)
    finally:
        done.set()


report = {}
for name in NOTEBOOKS:
    nb = json.load(open(os.path.join(REPO, name), encoding="utf-8"))
    work = tempfile.mkdtemp(prefix="asrun_")
    for f in ("train_dataset.txt", "validation_dataset.txt"):
        shutil.copy(os.path.join(REPO, f), work)
    os.chdir(work)
    ns = {"__name__": "__main__"}
    rows = []
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if not src.strip():
            continue
        t0 = time.time()
        try:
            code = compile(src, f"{name}:cell-{idx}", "exec")
        except SyntaxError as e:
            rows.append(dict(cell=idx, status="SYNTAX", error=f"{type(e).__name__}: {e.msg} (line {e.lineno})"))
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stdout, sys.stdout = sys.stdout, open(os.devnull, "w")
            try:
                run_with_timeout(code, ns)
                status, err = "ok", ""
            except BaseException as e:
                tb = traceback.extract_tb(e.__traceback__)[-1]
                status, err = "FAIL", f"{type(e).__name__}: {str(e)[:230]} [at {tb.name}, line {tb.lineno}]"
            finally:
                sys.stdout.close()
                sys.stdout = stdout
        rows.append(dict(cell=idx, status=status, error=err, secs=round(time.time() - t0, 2),
                         warnings=sorted({f"{w.category.__name__}: {str(w.message)[:90]}" for w in caught})))
    report[name] = rows
    print(f"\n### {name}")
    for r in rows:
        print(f"  cell-{r['cell']:<3} {r['status']:<7} {r.get('secs', '')!s:>6}  {r['error']}")
        for w in r.get("warnings", []):
            print(f"              warn: {w}")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    shutil.rmtree(work, ignore_errors=True)

json.dump(report, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "as_committed.json"), "w"), indent=1)
