"""Compositing a sketch frame (core/outputs/bl_sketch.py → cam<K>/_near.png, _far.png, sketch.json): the geometry levels are as in the previs
(core/outputs/previs.py: 1–99.5 % and an exponent of 0.85: Workbench gives a dark frame), but one set for all frames of the run — the animatic does not flicker;
the sky is a gradient of the node's tones from the top to the horizon (the horizon row follows the camera), the far terrain without shadow gets haze toward the horizon tone, the near part goes
on top → cam<K>/lit.png (the same file that editing, the sheet and post-processing take); the passes are deleted.
    .venv/bin/python -m core.outputs.sketch <cam<K> folder> [more …]"""
import json, os, sys

import numpy as np
from PIL import Image

PASSES = ("_near.png", "_far.png")
SPAN_MIN = 0.75                                         # levels: the upper one is no lower than the lower + 0.75 (dry tree crowns and rock used to turn white)


def _rgba(p):
    return np.asarray(Image.open(p).convert("RGBA"), np.float32) / 255.0


def levels(dirs, n=200_000):
    """The lower and upper geometry levels over all frames (1 % and 99.5 % of the pixel channels with alpha > 0.5; a sample of up to n per pass)."""
    vals = []
    for d in dirs:
        for f in PASSES:
            p = os.path.join(d, f)
            if not os.path.exists(p): continue
            a = _rgba(p); m = a[..., 3] > 0.5
            if m.any(): v = a[..., :3][m].ravel(); vals.append(v[:: max(1, len(v) // n)])
    if not vals: return 0.0, 1.0
    v = np.concatenate(vals); lo, hi = float(np.percentile(v, 1)), float(np.percentile(v, 99.5))
    return lo, max(hi, lo + SPAN_MIN)                                   # the gain is no more than 1/SPAN_MIN: in a green frame, gray does not go to white


def compose(d, lv=(0.0, 1.0)):
    J = json.load(open(os.path.join(d, "sketch.json")))
    lo, hi = lv; up = lambda c: np.clip((c - lo) / max(hi - lo, 1e-3), 0.0, 1.0) ** 0.85
    near = _rgba(os.path.join(d, "_near.png")); H, W = near.shape[:2]
    top, hor = np.asarray(J["sky_top"], np.float32), np.asarray(J["sky_horizon"], np.float32)
    yh = float(J["horizon"]) * H; y = np.arange(H, dtype=np.float32)[:, None, None]
    t = np.clip((yh - y) / max(yh, 1.0), 0.0, 1.0) ** 0.7
    sky = np.where(y > yh, hor * 0.9, hor + (top - hor) * t)            # below the horizon (a hole in the ground) — slightly darker than the horizon
    img = np.broadcast_to(sky, (H, W, 3)).copy()
    fp = os.path.join(d, "_far.png")
    if J.get("far") and os.path.exists(fp):
        far = _rgba(fp); a = far[..., 3:]
        img = (up(far[..., :3]) * (1 - J["haze"]) + hor * J["haze"]) * a + img * (1 - a)
    a = near[..., 3:]; img = up(near[..., :3]) * a + img * (1 - a)
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(os.path.join(d, "lit.png"))
    for f in PASSES:
        if os.path.exists(os.path.join(d, f)): os.remove(os.path.join(d, f))


if __name__ == "__main__":
    dirs, done, fail = sys.argv[1:], [], []
    lv = levels(dirs)
    for d in dirs:
        try: compose(d, lv); done.append(d)
        except Exception as e: fail.append(f"{d}: {type(e).__name__}: {e}")   # noqa: BLE001 — the frame was not composited: reported in the reply, the output decides
    print("__RESULT__" + json.dumps({"composed": len(done), "levels": [round(x, 3) for x in lv], "failed": fail}, ensure_ascii=False))
    sys.exit(1 if fail else 0)
