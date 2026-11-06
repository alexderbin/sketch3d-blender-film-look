"""A frame previs — a colored sketch with crisp outlines: the color from the library .blend (core/outputs/bl_color.py → color.png) and our outline
(line.png: silhouettes and edges). A marker sketch is a gray mass without outlines.
.venv/bin/python -m core.outputs.previs <out/<name>/cam<K>> [more folders…] [--line 0.85] [--lit — from lit.png into previs_lit.png, without levels]
Writes previs.png into every frame folder; if there is no color.png — skipped with a note.
With a far-terrain plan (the frame's far.json, core/outputs/bl_color.py for a scene with far.yaml) the geometry lies on a transparent background: the levels are as for the earlier frame
with a flat sky (the tone of the approved previs does not change), and under it is the plan's sky (sky.exr through AgX with the plan's exposure, core/light/far.py)."""
import json, os, sys
import numpy as np
from PIL import Image

args = [x for x in sys.argv[1:] if not x.startswith("--")]
K_LINE = float(next((x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--line=")), 0.85))   # how much the outline darkens
done, skipped = [], []
LIT = "--lit" in sys.argv                                            # a lit frame (bl_color light=…): no levels — night stays night
for d in args:
    cp, lp = os.path.join(d, "lit.png" if LIT else "color.png"), os.path.join(d, "line.png")
    if not (os.path.exists(cp) and os.path.exists(lp)): skipped.append(d); continue
    c = np.asarray(Image.open(cp).convert("RGB"), np.float32) / 255
    ap = os.path.join(d, "albedo.png")                              # a blend: the flat color holds the brightness, the studio one the volume
    if os.path.exists(ap) and not LIT: c = 0.5 * c + 0.5 * np.asarray(Image.open(ap).convert("RGB"), np.float32) / 255
    ln = np.asarray(Image.open(lp).convert("L").resize((c.shape[1], c.shape[0]), Image.BILINEAR), np.float32) / 255   # 1 is paper, 0 is line
    fj = os.path.join(d, "far.json"); F = json.load(open(fj)) if os.path.exists(fj) else None
    if F and F.get("mode") != ("lit" if LIT else "previs"): F = None   # another mode — left over from a previous run
    if F and not LIT:                                               # the far-terrain plan's sky under the geometry (on a transparent background)
        from core.light import far
        a = np.asarray(Image.open(cp).convert("RGBA"), np.float32)[..., 3:] / 255
        old = c * a + np.array(F["sky_bg"], np.float32) * (1 - a)   # the frame as it was with a flat sky — for the levels
        lo, hi = np.percentile(old, 1), np.percentile(old, 99.5)
        c = np.clip((c - lo) / max(hi - lo, 1e-3), 0, 1) ** 0.85
        c = far.previs_far(d, c, a, F)                              # the plan's sky and the hazes by the map bundle's depth
    elif not LIT:
        lo, hi = np.percentile(c, 1), np.percentile(c, 99.5)       # levels: Workbench gives a dark frame (the color is linear)
        c = np.clip((c - lo) / max(hi - lo, 1e-3), 0, 1) ** 0.85
    tp = os.path.join(d, "fog_t.png")                               # the far-terrain plan's haze: the outline melts away in it together with the mountain
    kl = K_LINE * (np.asarray(Image.open(tp).convert("L").resize((c.shape[1], c.shape[0])), np.float32) / 255 if F and os.path.exists(tp) else 1.0)
    out = c * (1 - kl * (1 - ln))[..., None]
    Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)).save(os.path.join(d, "previs_lit.png" if LIT else "previs.png")); done.append(d)
print("__RESULT__" + json.dumps({"previs": len(done), "skipped": skipped}, ensure_ascii=False))
