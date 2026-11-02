"""Synthetic Deep-DIC style data, following Tables 1-2 of Yang et al. (2022), in the notebooks' on-disk layout:
    imgs3/train_image_{i}_1.png, imgs3/train_image_{i}_2.png  (8-bit grey, 256x256)
    gt3/train_image_{i}.mat with Disp_field_1 of shape 256x256x2, channel 0 = horizontal u, channel 1 = vertical v

Stands in for the authors' 40k dataset, which is not in the repo. Approximations, where the paper leaves things open:
  - griddata forward warp -> fixed-point inverse + cv2.remap (same x -> x+u(x) mapping, much faster)
  - stretch/shear act about the frame centre, which reproduces Table 2's effective maxima (5.1 and 3.8 px at 128x128)
  - Gaussian amplitude scaled so the 0.6 maximum gives Table 2's 4.6 px at 128x128
  - blur sigma = 1 px (the paper does not state it)
"""
import json, multiprocessing as mp, os, sys
import cv2, numpy as np, scipy.io as sio

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SIZE, CROP, MARGIN = 512, 256, 34
GAUSS_SCALE = 9.2 / 0.6  # amplitude 0.6 -> 9.2 px on the 256 grid = 4.6 px on the 128 grid


def speckle(rng):
    r = rng.random()
    kind = "normal"
    if r < .05: kind = "sparse"
    elif r < .35: kind = "large"
    elif r < .40: kind = "noisy"
    elif r < .45: kind = "lowcontrast"
    n = rng.integers(2800, 4501)
    hi_axis = 3.5 if kind == "sparse" else 6.8
    hi_grey = .68 if kind == "lowcontrast" else .98
    img = np.zeros((SIZE, SIZE), np.float32)

    def stamp(count, lo, hi):
        S = 1 << 4
        cs = rng.uniform(0, SIZE - 1, (count, 2))
        ax = rng.uniform(lo, hi, (count, 2))
        gs = rng.uniform(.08, hi_grey, count)
        for (cx, cy), (a, b), g in zip(cs, ax, gs):
            cv2.ellipse(img, (int(cx * S), int(cy * S)), (int(a * S), int(b * S)), 0, 0, 360, float(g), -1, cv2.LINE_8, 4)

    stamp(n, 1.2, hi_axis)
    if kind == "large":
        stamp(int(rng.integers(1, 6)), 6.5, 9.5)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    if kind == "noisy":
        img = img + rng.normal(0, np.sqrt(rng.uniform(.001, .01)), img.shape).astype(np.float32)
    return img, kind


def displacement(rng):
    X, Y = np.meshgrid(np.arange(SIZE, dtype=np.float32), np.arange(SIZE, dtype=np.float32))
    c = (SIZE - 1) / 2
    tx, ty = rng.uniform(-4, 4, 2)
    kx, ky = rng.uniform(.96, 1.04, 2)
    th = rng.uniform(-.01, .01)
    sx, sy = rng.uniform(-.03, .03, 2)
    u = (kx - 1) * (X - c) - th * (Y - c) + sx * (Y - c) + tx
    v = (ky - 1) * (Y - c) + th * (X - c) + sy * (X - c) + ty
    n_g = int(rng.integers(0, 3))
    for _ in range(n_g):
        A = rng.uniform(.003, .6, 2) * rng.choice([-1, 1], 2) * GAUSS_SCALE
        x0, y0 = rng.uniform(0, SIZE - 1, 2)
        sgx, sgy = rng.uniform(.06, .5, 2) * SIZE
        g = np.exp(-(X - x0) ** 2 / (2 * sgx ** 2) - (Y - y0) ** 2 / (2 * sgy ** 2))
        u += A[0] * g
        v += A[1] * g
    return u.astype(np.float32), v.astype(np.float32)


def warp(img, u, v):
    """I2(x + u(x)) = I1(x): material at x in the reference moves to x + u(x)."""
    Xt, Yt = np.meshgrid(np.arange(SIZE, dtype=np.float32), np.arange(SIZE, dtype=np.float32))
    sx, sy = Xt.copy(), Yt.copy()
    for _ in range(6):
        sx = Xt - cv2.remap(u, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        sy = Yt - cv2.remap(v, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.remap(img, sx, sy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def make(i):
    rng = np.random.default_rng(20261102 + i)
    I1, kind = speckle(rng)
    u, v = displacement(rng)
    I2 = warp(I1, u, v)
    x0, y0 = rng.integers(MARGIN, SIZE - CROP - MARGIN + 1, 2)
    crop = lambda a: a[y0:y0 + CROP, x0:x0 + CROP]
    noisy = lambda a: np.clip(crop(a) + rng.normal(0, np.sqrt(.001), (CROP, CROP)), 0, 1)
    png1, png2 = noisy(I1), noisy(I2)
    cv2.imwrite(os.path.join(OUT, "imgs3", f"train_image_{i}_1.png"), (png1 * 255).round().astype(np.uint8))
    cv2.imwrite(os.path.join(OUT, "imgs3", f"train_image_{i}_2.png"), (png2 * 255).round().astype(np.uint8))
    gt = np.stack([crop(u), crop(v)], axis=2).astype(np.float32)
    sio.savemat(os.path.join(OUT, "gt3", f"train_image_{i}.mat"), {"Disp_field_1": gt})
    return i, kind, float(np.abs(gt).mean()), float(np.abs(gt).max())


if __name__ == "__main__":
    n = int(sys.argv[1])
    os.makedirs(os.path.join(OUT, "imgs3"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "gt3"), exist_ok=True)
    with mp.Pool(8, initializer=cv2.setNumThreads, initargs=(1,)) as p:
        rows = p.map(make, range(1, n + 1), chunksize=16)
    json.dump(rows, open(os.path.join(OUT, "meta.json"), "w"))
    kinds = {k: sum(r[1] == k for r in rows) for k in {r[1] for r in rows}}
    print("samples", n, kinds)
    print("mean |disp| 256-grid px: %.2f, largest %.1f" % (np.mean([r[2] for r in rows]), np.max([r[3] for r in rows])))
