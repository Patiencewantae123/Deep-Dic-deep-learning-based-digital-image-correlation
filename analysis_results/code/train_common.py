"""The dataset pieces of train.py, importable without starting a run."""
from common import _cell


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
