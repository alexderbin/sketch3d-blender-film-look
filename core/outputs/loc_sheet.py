"""The cameras are placed by the scene box: the center and extent of the "site" (figures and parts smaller than 60 m),
and the main axis follows the spread of the site.
A location sheet is five views of one scene: an establishing view along the main axis from the side of camera 1, a reverse view from the other end, two
side views, a plan from above (north up); in the free cell — supports seen in two or more views (openings do not move).
.venv/bin/python -m core.outputs.loc_sheet scenes/<name> [out/<name>/locsheet] [layer: previs for library scenes, otherwise toneline]
If a view is blocked closer than the midpoint of the way to the site by more than 30 % — the camera moves 0.65 times closer to the center, up to 3 times.
Output: <folder>/scene (links to the scene + its own cameras.json), cam1…cam5 (map bundles), sheet_<layer>.jpg, overlap.json."""
import core.paths        # noqa: E402  core parts on sys.path
import json, math, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = core.paths.ROOT
from core.model import scene_info as SI

SD = os.path.abspath(sys.argv[1]); name = os.path.basename(SD)
OUT = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(ROOT, "out", name, "locsheet")
LAYER = sys.argv[3] if len(sys.argv) > 3 else "previs" if os.path.exists(os.path.join(SD, "source.json")) else "toneline"   # previs — for library scenes
RES = [1280, 720]
BIG = ("ground", "terrain", "floor", "skyline", "sky", "water")


def look(title, pos, target, lens, up_hint=(0, 0, 1)):
    pos, target = np.array(pos, float), np.array(target, float)
    back = pos - target; back /= np.linalg.norm(back)
    right = np.cross(up_hint, back); right /= np.linalg.norm(right)
    up = np.cross(back, right)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = right, up, back, pos
    fwd = -back
    return {"name": title, "location": pos.round(3).tolist(), "yaw": round(math.degrees(math.atan2(fwd[1], fwd[0])), 3),
            "pitch": round(math.degrees(math.asin(max(-1, min(1, fwd[2])))), 3), "roll": 0.0, "lens_mm": lens, "sensor_width_mm": 36.0,
            "sensor_fit": "AUTO", "shift_x": 0.0, "shift_y": 0.0, "resolution": RES, "matrix_world": M.round(6).tolist()}


# the site: figures and parts smaller than 60 m; center, extent, main axis
kind, lo, hi, gz = SI.parts_of(SD); figs, _ = SI.figures_of(SD)
pts = [(lo[p] + hi[p]) / 2 for p in lo if kind.get(p) not in BIG and not p.startswith("figure_") and max(hi[p] - lo[p]) < 60]
pts += [np.array(f["pos"], float) for f in figs.values()]
LJ = os.path.join(SD, "location.json")                              # the site from the plan (components/site/terrain.py): without buildings — the site follows the plan's anchors
if not pts and os.path.exists(LJ):
    L_ = json.load(open(LJ)); pts = [np.array(L_["anchors"][a], float) for a in L_.get("public", []) if a in L_["anchors"]]
P = np.array(pts)[:, :2]; c = np.median(P, 0)
R = max(8.0, float(np.percentile(np.linalg.norm(P - c, axis=1), 80)) * 1.15)
ev, evec = np.linalg.eigh(np.cov((P - c).T)) if len(P) > 2 else (None, np.eye(2))
ax = evec[:, -1]
cams0 = json.load(open(os.path.join(SD, "cameras.json")))
if cams0 and np.dot(np.array(cams0[0]["matrix_world"])[:2, 3] - c, ax) < 0: ax = -ax       # the establishing view is from the side of camera 1
side = np.array([-ax[1], ax[0]])
z0 = np.median([f["pos"][2] for f in figs.values()]) if figs else gz
T = np.array([c[0], c[1], z0 + 1.5])
at = lambda d, h: np.array([c[0] + d[0], c[1] + d[1], z0 + h])
if os.path.exists(os.path.join(SD, "relief.npz")):                 # relief: the camera is no lower than 2 m above the ground beneath it
    from core.surface.ground import Ground
    G_ = Ground(SD); at0 = at; at = lambda d, h: (lambda p: np.array([p[0], p[1], max(p[2], float(G_.top(p[0], p[1])) + 2.0)]))(at0(d, h))
CAMS = [look("1 · establishing along the main axis", at(ax * R * 2.2, R * 0.5 + 2), T, 24),
        look("2 · reverse from the other end", at(-ax * R * 2.2, R * 0.5 + 2), T, 24),
        look("3 · from the left side", at(side * R * 2.2, R * 0.4 + 2), T, 24),
        look("4 · from the right side", at(-side * R * 2.2, R * 0.4 + 2), T, 24),
        look("5 · plan from above (north up)", at((0, 0), R * 2.9), T - [0, 0, 1.5], 24, up_hint=(0, 1, 0))]
sd = os.path.join(OUT, "scene")
from core.model import scene_copy; scene_copy.link(SD, sd)                         # the whole scene description (materials, sky node, site)
from core.outputs import maps
todo, moved, t_all = [str(k) for k in range(1, 6)], {}, 0.0
for rnd in range(4):                                                # the view is blocked by a wall or a drop near the camera — move the camera closer to the center
    json.dump(CAMS, open(os.path.join(sd, "cameras.json"), "w"), ensure_ascii=False)
    r = subprocess.run([os.path.join(ROOT, ".venv/bin/python"), *core.paths.python_target(core.paths.script("frames.py")), sd, OUT, ",".join(todo)], cwd=ROOT, capture_output=True, text=True)
    res = json.loads(next(l for l in r.stdout.splitlines() if l.startswith("__RESULT__"))[10:]) if "__RESULT__" in r.stdout else {"error": r.stderr[-400:]}
    if res.get("failed") or "error" in res:
        print("__RESULT__" + json.dumps({"error": res}, ensure_ascii=False)); sys.exit(1)
    t_all += res.get("s", 0); again = []
    for k in todo:
        z = maps.read_exr(os.path.join(OUT, f"cam{k}", "raw", "passes.exr"))["z"]
        c0 = CAMS[int(k) - 1]; pos = np.array(c0["location"]); tgt = T - ([0, 0, 1.5] if k == "5" else 0)
        if rnd < 3 and (z < 0.5 * np.linalg.norm(pos - tgt)).mean() > 0.30:          # an obstruction closer than the midpoint of the way to the site
            CAMS[int(k) - 1] = look(c0["name"], tgt + (pos - tgt) * 0.65, tgt, c0["lens_mm"], up_hint=(0, 1, 0) if k == "5" else (0, 0, 1))
            again.append(k); moved[k] = moved.get(k, 0) + 1
    if not again: break
    todo = again
res["s"] = round(t_all, 1)

# sheet assembly: the establishing view large, the reverse view and the plan on the right, the side views and the supports table below
FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
CW, CH, G, LAB = 640, 360, 12, 30
LAYOUT = {1: (0, 0, 2), 2: (2, 0, 1), 5: (2, 1, 1), 3: (0, 2, 1), 4: (1, 2, 1)}
sheet = Image.new("RGB", (3 * CW + 4 * G, 3 * (CH + LAB) + 4 * G), "white"); dr = ImageDraw.Draw(sheet)
seen = {}
for k, (col, row, span) in LAYOUT.items():
    fr = json.load(open(os.path.join(OUT, f"cam{k}", "frame.json")))
    for o in fr["objects"]:
        if o["class"] not in ("ground", "sky", "member"): seen.setdefault(o["id"], set()).add(k)
    for cid in fr.get("crowds", {}): seen.setdefault(cid, set()).add(k)
    w = span * CW + (span - 1) * G; h = round(w * 9 / 16)
    p_ = os.path.join(OUT, f"cam{k}", f"{LAYER}.png"); p_ = p_ if os.path.exists(p_) else os.path.join(OUT, f"cam{k}", "toneline.png")
    im = Image.open(p_).convert("RGB").resize((w, h))
    x, y = G + col * (CW + G), G + row * (CH + LAB + G)
    dr.text((x + 2, y + 2), fr["camera"]["name"], font=FONT, fill="black"); sheet.paste(im, (x, y + LAB))
    dr.rectangle([x - 1, y + LAB - 1, x + w, y + LAB + h], outline="black", width=2)
x, y = G + 2 * (CW + G), G + 2 * (CH + LAB + G)
shared = {k: sorted(v) for k, v in seen.items() if len(v) >= 2}
dr.text((x + 2, y + 2), "supports in two or more views:", font=FONT, fill="black")
for i, (k, v) in enumerate(sorted(shared.items(), key=lambda kv: -len(kv[1]))[:9]):
    dr.text((x + 8, y + LAB + 6 + i * 34), f"{k[:28]}: views {', '.join(map(str, v))}", font=FONT, fill="black")
rc = subprocess.run(["blender", "-b", "--factory-startup", "-P", core.paths.RUN, "--", os.path.join(core.paths.ROOT, "components/camera/bl_cover.py"), "--", sd, "cameras=1-5", "add=2"],
                    capture_output=True, text=True)                  # coverage: what share of the site's surfaces the views show
cov = next((json.loads(l.split("__RESULT__", 1)[1]) for l in rc.stdout.splitlines() if "__RESULT__" in l), None)
if cov:
    tip = "; ".join(f"+ azimuth {x['azimuth']}°, elevation {x['elevation']}° → {x['coverage']:.0%}" for x in cov["added"])
    dr.text((x + 2, y + LAB + 6 + 9 * 34 + 8), f"the views show {cov['coverage_given']:.0%} of the surfaces", font=FONT, fill=(120, 40, 30))
    if tip: dr.text((x + 8, y + LAB + 6 + 10 * 34 + 8), tip[:60], font=FONT, fill=(120, 40, 30))
out = os.path.join(OUT, f"sheet_{LAYER}.jpg"); sheet.save(out, quality=88)
info = {"sheet": out, "center": c.round(2).tolist(), "radius_m": round(R, 1), "axis": ax.round(3).tolist(), "moved_closer": moved, "landmarks": len(seen),
        "shared_2plus": len(shared), "s": res.get("s"), **({"coverage": cov["coverage_given"], "add_views": cov["added"]} if cov else {})}
json.dump({**info, "per_landmark": {k: sorted(v) for k, v in seen.items()}}, open(os.path.join(OUT, "overlap.json"), "w"), ensure_ascii=False, indent=1)
print("__RESULT__" + json.dumps(info, ensure_ascii=False))
