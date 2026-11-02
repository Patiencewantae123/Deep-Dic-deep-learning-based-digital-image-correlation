"""Load the model definitions straight out of the repo's notebooks, so what runs here is the committed code."""
import json, os, warnings

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "committed_notebooks")  # frozen copy of HEAD, so edits to the repo cannot change a run in progress
HERE = os.path.dirname(os.path.abspath(__file__))


def _cell(nb, idx):
    with open(os.path.join(REPO, nb), encoding="utf-8") as f:
        return "".join(json.load(f)["cells"][idx]["source"])


def _load_classes(nb, cell_idx):
    ns = {}
    exec("import torch, torch.nn as nn\nimport numpy as np", ns)
    exec(_cell(nb, cell_idx), ns)
    return ns


def build(nb, cell_idx, num_classes):
    ns = _load_classes(nb, cell_idx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return ns["SegResNet"](num_classes, ns["FeatureResNet"]())


def build_disp():
    return build("DisplacementNet_train.ipynb", 2, 2)


def build_strain():
    return build("StrainNet_train.ipynb", 2, 4)


def build_disp_run():
    return build("Diaplacement_run.ipynb", 2, 2)


def build_strain_run():
    return build("StrainNet_run.ipynb", 1, 4)
