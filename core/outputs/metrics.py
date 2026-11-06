"""Frame metrics as a core output (numbers the code can compute — the model judges on top of them, not instead).

    .venv/bin/python -m core.outputs.metrics <frame.png|folder> [--json]

For a frame (lit.png / preview png) → dict:
  sky_fraction     share of pixels that read as sky (bright, low-saturation, upper part) — 0 = no sky, 1 = only sky
  horizon_row      row (0..1 from the top) with the strongest vertical luminance change in the upper 70 % — where land meets sky
  overexposed      share of pixels with luminance > 0.98
  black            share of pixels with luminance < 0.02
  mean_luma        mean luminance 0..1
  contrast         std of luminance
  sharpness        variance of the Laplacian of luminance (higher = sharper); compare only at the same resolution
  id_coverage      from id.png next to the frame (raw/id.png): share of pixels with a non-zero id — how much of the frame is objects
  size             [w, h]
For a folder: every lit.png / *.png frame directly inside (and cam*/lit.png) — one record each.
Stdlib + numpy + PIL (the project venv). No Blender.
"""
import json
import os
import sys

import numpy as np
from PIL import Image


def _luma(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2], a


def frame_metrics(path):
    img = Image.open(path)
    L, rgb = _luma(img)
    h, w = L.shape
    sat = rgb.max(axis=-1) - rgb.min(axis=-1)
    rows = np.arange(h)[:, None] / max(h - 1, 1)
    sky = (L > 0.55) & (sat < 0.35) & (rows < 0.6)                     # bright, unsaturated, upper part
    sky_fraction = float(sky.mean())
    prof = L.mean(axis=1)
    top = max(int(h * 0.7), 2)
    grad = np.abs(np.diff(prof[:top]))
    horizon_row = float(int(grad.argmax()) / max(h - 1, 1)) if grad.size else None
    gy = np.diff(L, axis=0)[:, 1:-1]; gx = np.diff(L, axis=1)[1:-1, :]
    lap = (np.diff(L, n=2, axis=0)[:, 1:-1] + np.diff(L, n=2, axis=1)[1:-1, :])
    out = {"size": [int(w), int(h)], "sky_fraction": round(sky_fraction, 4), "horizon_row": round(horizon_row, 4) if horizon_row is not None else None,
           "overexposed": round(float((L > 0.98).mean()), 4), "black": round(float((L < 0.02).mean()), 4),
           "mean_luma": round(float(L.mean()), 4), "contrast": round(float(L.std()), 4), "sharpness": round(float(lap.var()), 6)}
    idp = os.path.join(os.path.dirname(path), "raw", "id.png")
    if not os.path.exists(idp):
        idp = os.path.join(os.path.dirname(path), "id.png")
    if os.path.exists(idp):
        ida = np.asarray(Image.open(idp).convert("L"))
        out["id_coverage"] = round(float((ida > 0).mean()), 4)
    return out


def folder_metrics(folder):
    recs = {}
    cands = []
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if name.endswith(".png") and os.path.isfile(p):
            cands.append(p)
        elif os.path.isdir(p) and os.path.exists(os.path.join(p, "lit.png")):
            cands.append(os.path.join(p, "lit.png"))
    for p in cands:
        try:
            recs[os.path.relpath(p, folder)] = frame_metrics(p)
        except Exception as e:                                             # noqa: BLE001 — a bad file is a record, not a crash
            recs[os.path.relpath(p, folder)] = {"error": f"{type(e).__name__}: {e}"}
    return recs


def main(argv):
    if not argv:
        print(__doc__); return 2
    target = argv[0]
    res = folder_metrics(target) if os.path.isdir(target) else {os.path.basename(target): frame_metrics(target)}
    if "--json" in argv:
        print(json.dumps(res, ensure_ascii=False, indent=1))
    else:
        for k, m in res.items():
            if "error" in m:
                print(f"{k}: {m['error']}"); continue
            print(f"{k}: {m['size'][0]}x{m['size'][1]} sky {m['sky_fraction']:.2f} horizon {m['horizon_row']} over {m['overexposed']:.3f} black {m['black']:.3f} "
                  f"luma {m['mean_luma']:.2f} contrast {m['contrast']:.2f} sharp {m['sharpness']:.4f}" + (f" id {m['id_coverage']:.2f}" if "id_coverage" in m else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
