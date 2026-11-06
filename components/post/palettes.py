"""Palette references for grading (palettes.yaml): the reference's mean and colour covariance — we do not store pixels, only numbers.
A reference is a clip (frames from 20 to 80 % of its length, without the letterbox) or a folder of images; screen colour sRGB 0..1.
  .venv/bin/python -m components.post.palettes add <name> <clip.mp4 | image folder> [--note "what it is"] [--frames 16]
  .venv/bin/python -m components.post.palettes list
"""
import glob
import os
import sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "palettes.yaml")
W = 480


def _crop_bars(im):
    """Without the letterbox: rows that are almost entirely black, at the top and bottom."""
    y = im.mean(axis=(1, 2))
    rows = np.where(y > 0.02)[0]
    return im[rows[0]:rows[-1] + 1] if len(rows) > im.shape[0] // 3 else im


def frames_of(src, n=16):
    if os.path.isdir(src):
        fs = sorted(glob.glob(os.path.join(src, "*.png")) + glob.glob(os.path.join(src, "*.jpg")))
        ims = [cv2.imread(f)[..., ::-1].astype(np.float32) / 255 for f in fs]
    else:
        cap = cv2.VideoCapture(src); N = cap.get(cv2.CAP_PROP_FRAME_COUNT); ims = []
        for t in np.linspace(0.2, 0.8, n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(N * t)); ok, f = cap.read()
            if ok: ims.append(f[..., ::-1].astype(np.float32) / 255)
    out = []
    for im in ims:
        im = _crop_bars(im)
        out.append(cv2.resize(im, (W, max(8, round(im.shape[0] * W / im.shape[1]))), interpolation=cv2.INTER_AREA))
    return out


def describe(ims):
    """Numbers for the eye: brightness, saturation of the lit areas, "warm − cool" (b* of the lightest 10 % minus b* of the darkest 30 %)."""
    x = np.concatenate([i.reshape(-1, 3) for i in ims])
    y = x @ np.array([0.2126, 0.7152, 0.0722])
    u8 = (np.clip(x, 0, 1) * 255).astype(np.uint8).reshape(-1, 1, 3)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).reshape(-1, 3); lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    lit = hsv[:, 2] > 25
    nz = y > 0.04
    hi, lo = y >= np.percentile(y, 90), nz & (y <= np.percentile(y[nz], 30))
    return {"brightness": round(float(y.mean()), 3), "saturation": round(float(hsv[lit, 1].mean() / 255), 3),
            "warm_minus_cool": round(float((lab[hi, 2] - 128).mean() - (lab[lo, 2] - 128).mean()), 1)}


def add(name, src, note="", n=16):
    from components.post.look import stats_of
    ims = frames_of(src, n)
    if not ims: raise SystemExit(f"no frames: {src}")
    mu, cov = stats_of(ims)
    doc = (yaml.safe_load(open(PATH)) or {}) if os.path.exists(PATH) else {}
    doc.setdefault("palettes", {})[name] = {"note": note, "frames": len(ims), **describe(ims),
                                            "mean": [round(float(v), 5) for v in mu], "cov": [[round(float(v), 6) for v in r] for r in cov]}
    head = ("# Palette references for grading: mean and colour covariance (sRGB 0..1) — we do not store the reference's pixels.\n"
            "# Add: .venv/bin/python -m components.post.palettes add <name> <clip | image folder> --note '…'\n")
    with open(PATH, "w") as f:
        f.write(head); yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False, width=200)
    return doc["palettes"][name]


def main(argv):
    if not argv or argv[0] == "list":
        for k, v in ((yaml.safe_load(open(PATH)) or {}).get("palettes", {}) if os.path.exists(PATH) else {}).items():
            print(k, v.get("brightness"), v.get("saturation"), v.get("warm_minus_cool"), "—", v.get("note", ""))
        return
    if argv[0] == "add":
        import argparse
        ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("src"); ap.add_argument("--note", default="")
        ap.add_argument("--frames", type=int, default=16)
        a = ap.parse_args(argv[1:])
        r = add(a.name, a.src, a.note, a.frames)
        print(a.name, {k: r[k] for k in ("brightness", "saturation", "warm_minus_cool", "frames")})


if __name__ == "__main__":
    main(sys.argv[1:])
