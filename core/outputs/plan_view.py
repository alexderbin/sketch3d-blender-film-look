"""A top-down scene diagram without rendering: object footprints by class, figures with a gaze arrow, cameras
as field-of-view wedges with numbers, the line of action of a pair, occluded cameras in red. For the caller (~600 tokens instead of frames)
and for a human: to check the plan and the camera placement before rendering.
.venv/bin/python -m core.outputs.plan_view scenes/<name> [out.png] [--cams=1,3] [--hero=m0] [--pair=m0,m1] [--size=800] [--no-labels]
Without --cams — all the scene's cameras; the line of action comes from --pair or from pairs in the cameras' intents. The reply is a __RESULT__ line."""
import json, math, os, re, sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

a = [x for x in sys.argv[1:] if not x.startswith("--")]; fl = dict((x[2:].split("=", 1) + [""])[:2] for x in sys.argv[1:] if x.startswith("--"))
SD = a[0]; OUT = a[1] if len(a) > 1 else os.path.join(SD, "plan.png"); S = int(fl.get("size", 800))
W_ = json.load(open(os.path.join(SD, "world.json"))); kind = {e["id"]: e["kind"] for e in W_["reg"]}
kit = json.load(open(os.path.join(SD, "figkit.json"))); named = kit.get("named", {})
cams = json.load(open(os.path.join(SD, "cameras.json"))) if os.path.exists(os.path.join(SD, "cameras.json")) else []
pick = [int(x) for x in fl["cams"].split(",")] if fl.get("cams") else list(range(1, len(cams) + 1))
FILL = {"wall": (150, 146, 138), "box": (176, 140, 96), "rock": (160, 160, 168), "rocks": (160, 160, 168), "tree": (120, 160, 100),
        "siege_tower": (130, 110, 90), "ridge": (214, 210, 200), "banner": (190, 70, 60)}
SKIP = {"ground", "figure", "member", "crowd", "skyline", "craters"}

V, used, cur = [], {}, None                                          # object footprints: the face vertices of a scene.obj object → an XY hull
for line in open(os.path.join(SD, "scene.obj")):                    # (the bridge writes the vertices at the start of the file, an object has only faces)
    if line.startswith("v "): V.append(line.split()[1:3])
    elif line.startswith("o "): cur = line[2:].strip(); cur = cur if kind.get(cur, "box") not in SKIP and not cur.startswith("figure_") else None
    elif line.startswith("f ") and cur: used.setdefault(cur, set()).update(int(t.split("/")[0]) for t in line.split()[1:])
V = np.array(V, np.float32); hulls = {}
for oid, ix in used.items():
    p = V[[i - 1 if i > 0 else len(V) + i for i in ix]]
    if len(p) >= 3: hulls[oid] = cv2.convexHull(p).reshape(-1, 2)
figs = {fid: (np.array(f["pos"][:2], float), math.radians(f.get("heading", 0.0)), f.get("height", 1.8)) for fid, f in named.items()}
camd = []
for k in pick:
    c = cams[k - 1]; M = np.array(c["matrix_world"], float); p, fwd = M[:3, 3], -M[:3, 2]
    w, h = c.get("resolution", [1536, 864]); half = math.atan(c.get("sensor_width_mm", 36.0) * (1 if w >= h else w / h) / 2 / c.get("lens_mm", 50.0))
    it = c.get("intent") or {}
    camd.append((k, p[:2], math.atan2(fwd[1], fwd[0]), half, it.get("visible", 1.0) < 0.85, it))
# box: figures, cameras and objects near them (distant relief parts and walls at the horizon do not inflate the diagram)
core = [f[0] for f in figs.values()] + [c[1] for c in camd]
core = np.array(core) if core else np.concatenate(list(hulls.values()))
lo, hi = core.min(0) - 3, core.max(0) + 3
for hpts in hulls.values():
    if np.all((hpts.min(0) > lo - 12) & (hpts.max(0) < hi + 12)): lo, hi = np.minimum(lo, hpts.min(0) - 1), np.maximum(hi, hpts.max(0) + 1)
span = max(hi - lo); ctr = (lo + hi) / 2; sc_ = (S - 60) / span
to_px = lambda q: (S / 2 + (q[0] - ctr[0]) * sc_, S / 2 - (q[1] - ctr[1]) * sc_)   # north is up
im = Image.new("RGB", (S, S), (247, 245, 239)); d = ImageDraw.Draw(im)
font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13); small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 11)
step = next(s for s in (1, 2, 5, 10, 20, 50, 100) if span / s <= 14)  # grid: no more than ~14 lines
for i in range(int(math.floor((ctr[0] - span) / step)), int(math.ceil((ctr[0] + span) / step)) + 1):
    x = to_px((i * step, 0))[0]; d.line([(x, 0), (x, S)], fill=(225, 222, 213) if i % 5 else (205, 201, 190))
for j in range(int(math.floor((ctr[1] - span) / step)), int(math.ceil((ctr[1] + span) / step)) + 1):
    y = to_px((0, j * step))[1]; d.line([(0, y), (S, y)], fill=(225, 222, 213) if j % 5 else (205, 201, 190))
areas = sorted(((cv2.contourArea(h), oid) for oid, h in hulls.items()), reverse=True)
big = {oid for _, oid in areas[:12]}                                  # labels only for the largest (so the diagram does not drown in names)
for oid, h in hulls.items():
    k_ = kind.get(oid, "box"); poly = [to_px(q) for q in h]
    if len(poly) >= 3: d.polygon(poly, fill=FILL.get(k_, (176, 140, 96)) if k_ != "ridge" else None, outline=(110, 92, 64) if k_ != "ridge" else (180, 175, 160))
    if oid in big and "--no-labels" not in sys.argv and k_ != "ridge":
        c = to_px(h.mean(0)); d.text(c, re.sub(r"_+$", "", oid)[:18], fill=(70, 59, 34), font=small, anchor="mm", stroke_width=2, stroke_fill=(250, 248, 242))
pairs = [tuple(fl["pair"].split(","))] if fl.get("pair") else sorted({tuple(sorted((c[5]["hero"], c[5]["pair"]))) for c in camd if c[5].get("pair")})
for A, Bf in pairs:                                                   # line of action (the 180° axis) — dashed
    if A in figs and Bf in figs:
        p0, p1 = np.array(to_px(figs[A][0])), np.array(to_px(figs[Bf][0])); n = int(np.linalg.norm(p1 - p0) / 10) + 1
        for i in range(0, n, 2): d.line([tuple(p0 + (p1 - p0) * i / n), tuple(p0 + (p1 - p0) * min(i + 1, n) / n)], fill=(60, 125, 58), width=2)
for k, p, ang, half, hidden, it in camd:                             # cameras: a field-of-view wedge up to 1/4 of the diagram
    L = span * 0.22; P = to_px(p); e1 = to_px(p + L * np.array([math.cos(ang - half), math.sin(ang - half)])); e2 = to_px(p + L * np.array([math.cos(ang + half), math.sin(ang + half)]))
    col = (214, 64, 52) if hidden else (90, 88, 82)
    d.polygon([P, e1, e2], fill=(214, 64, 52, 40) if hidden else None, outline=col); d.ellipse([P[0] - 5, P[1] - 5, P[0] + 5, P[1] + 5], fill=col)
    d.text((P[0] + 8, P[1] + 6), str(k) + (f" {it.get('intent', '')}" if it.get("intent") else ""), fill=col, font=font, stroke_width=2, stroke_fill=(250, 248, 242))
hero = fl.get("hero")
for fid, (p, hd, hgt) in figs.items():                                # figures: a circle, a gaze arrow, the name behind the back
    P = to_px(p); r = max(4, 0.35 * sc_); col = (214, 120, 40) if fid == hero else (47, 106, 168)
    d.ellipse([P[0] - r, P[1] - r, P[0] + r, P[1] + r], fill=(110, 168, 224) if fid != hero else (240, 170, 90), outline=col, width=2)
    tip = (P[0] + math.cos(hd) * (r + 9), P[1] - math.sin(hd) * (r + 9)); d.line([P, tip], fill=col, width=2)
    d.text((P[0] - math.cos(hd) * (r + 10), P[1] + math.sin(hd) * (r + 10)), fid, fill=col, font=font, anchor="mm", stroke_width=2, stroke_fill=(250, 248, 242))
x0, y0 = 14, S - 18; d.line([(x0, y0), (x0 + step * sc_, y0)], fill=(60, 60, 60), width=3); d.text((x0 + step * sc_ + 6, y0 - 7), f"{step} m", fill=(60, 60, 60), font=font)
d.text((S - 22, 12), "S", fill=(60, 60, 60), font=font); d.line([(S - 18, 30), (S - 18, 46)], fill=(60, 60, 60), width=2)
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True); im.save(OUT)
print("__RESULT__" + json.dumps({"plan": OUT, "objects": len(hulls), "figures": len(figs), "cameras": len(camd), "hidden": [c[0] for c in camd if c[4]],
                                 "span_m": round(float(span), 1), "grid_m": step, "tokens_est": round(S * S / 750)}, ensure_ascii=False))
