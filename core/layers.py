"""Scene layers — the contract between components (ARCHITECTURE.md, "Layers"). A layer file lies in the run's scene folder; the file
has a single writer (it declares the file in the passport's scene_files), readers use the functions from here. Components do not
read each other's plans (castle.json, ensemble.json, forest.json).
  occupancy.json — occupancy (Buildings): {"items": [{"kind", "poly": [[x, y], …]} | {"kind", "line": [[x, y(, z)], …], "w": m}],
                   "palette": {stone, roof, plaster, rock}}; kind: site (the castle platform), yard (a courtyard: level), wall, tower (xy, r),
                   building, house (walls: base, floor, eave, storeys, fachwerk; house — also h, door, type), plot (power, trade,
                   wealth), street, square, road, town (the boundary of the concept — not occupancy)
  anchors.json   — building anchors (Buildings): {"name": [x, y, z]}; relief anchors — the Site's location.json ("anchors")
  surfaces.json  — masonry surfaces (Buildings): {"items": [wall: {"tag": w<i>_out|w<i>_in, "wall", "side" (1 — outward,
                   2 — inward), "part": outer|inner, "strip" (core.surface.strip), "sid0", "gaps"} | tower: {"tag": t<i>, "tower",
                   "part": towers, "strip", "sid0", "openings"}], "palette": {stone, plaster, roof}}; the reader sets the choice and the seed
  canopy.json    — the forest canopy (Plants): {"trees": [{x, y, cr (crown radius), h, species, habit}, …]}
  influence.npz  — the settlement influence raster (Buildings): T (trade, uint8 0…255), x0, y0, step
  cameras.json   — cameras (Camera, take): a list of {name, matrix_world, lens_mm, …}; pinned: "<run>, <date>" — the entry was approved on
                   the sketch (core/services/stage.py pin, the source scene): the layer writer does not recompute it but builds the camera from it
A read without the layer gives an empty result (None, [], {}), not an error: a step decides itself whether it needs the layer; but
every read is accounted for (SEEN: layer → whether it existed and is non-empty) — a step's plan and assembly hand the accounting
to the manager, which writes "no layer …" into the step's line (the reads section — a step failure)."""
import json
import os

import numpy as np

FILES = {"occupancy": "occupancy.json", "anchors": "anchors.json", "surfaces": "surfaces.json", "canopy": "canopy.json",
         "cameras": "cameras.json", "influence": "influence.npz"}
SECTION = {"occupancy": "buildings", "anchors": "buildings", "surfaces": "buildings", "influence": "buildings",   # the scene model section
           "canopy": "layout", "cameras": "cameras"}                                                                # of the layer's writer
SEEN = {}                                                             # the process's read accounting: layer → whether it existed and is non-empty


def path(scene, name):
    return os.path.join(scene, FILES[name])


def write(scene, name, data):
    """Write a layer as a whole (the layer has a single writer; a write replaces the previous one)."""
    json.dump(data, open(path(scene, name), "w"), ensure_ascii=False)
    return path(scene, name)


def read(scene, name, default=None):
    p = path(scene, name)
    try: d = json.load(open(p)) if os.path.exists(p) else None
    except (OSError, ValueError): d = None
    full = bool(d) and not (isinstance(d, dict) and set(d) <= {"items", "trees", "palette"} and not (d.get("items") or d.get("trees")))
    SEEN[name] = SEEN.get(name, False) or full
    return default if d is None else d


# ── occupancy ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def palette(scene):
    """The building palette (the plan's colors stone, roof, plaster, rock): the Buildings occupancy, otherwise the surfaces layer; none — {}."""
    return (read(scene, "occupancy") or {}).get("palette") or (read(scene, "surfaces") or {}).get("palette") or {}


def occupancy(scene, kinds=None):
    """Occupancy elements; kinds — only these kinds."""
    items = (read(scene, "occupancy") or {}).get("items") or []
    return [it for it in items if not kinds or it.get("kind") in kinds]


def occupancy_mask(scene, box, res=0.5, kinds=None, pad=0.0):
    """Occupancy raster over the box box (x0, y0, x1, y1) with step res, m → (a uint8 mask: 1 — occupied; row i ↔ y0 + i·res,
    column j ↔ x0 + j·res). pad — a margin around the occupied area, m (a 6 m road with a 2 m margin — a 10 m strip)."""
    from PIL import Image, ImageDraw
    x0, y0, x1, y1 = box
    W, H = int(np.ceil((x1 - x0) / res)) + 1, int(np.ceil((y1 - y0) / res)) + 1
    im = Image.new("L", (W, H), 0); d = ImageDraw.Draw(im)
    px = lambda p: ((float(p[0]) - x0) / res, (float(p[1]) - y0) / res)
    for it in occupancy(scene, kinds):
        if len(it.get("poly") or []) >= 3:
            d.polygon([px(p) for p in it["poly"]], fill=1)
        if len(it.get("line") or []) >= 2:
            w = max(1, int(round(float(it.get("w", 3.0)) / res)))
            d.line([px(p) for p in it["line"]], fill=1, width=w)
            for p in (it["line"][0], it["line"][-1]):                 # the ends get a disk so that the street does not break off in a step
                q = px(p); d.ellipse([q[0] - w / 2, q[1] - w / 2, q[0] + w / 2, q[1] + w / 2], fill=1)
    m = np.asarray(im, np.uint8).copy()
    r = int(round(pad / res))
    if r > 0:                                                         # dilation by a disk of radius r
        out = m.copy()
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy > r * r: continue
                out |= np.roll(np.roll(m, dy, 0), dx, 1)
        m = out
    return m


BUILT = ("site", "yard", "wall", "tower", "building", "house", "plot", "street", "square", "road")   # everything built ("town" is the boundary of the concept)


def occupied_at(scene, X, Y, kinds=BUILT, pad=0.0, res=0.5):
    """Occupancy at points (X, Y — arrays of the same shape, m) → a bool array of the same shape; pad — a margin, m. No layer — everything is free."""
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    if X.size == 0 or not occupancy(scene, kinds): return np.zeros(X.shape, bool)
    box = (float(X.min()) - pad - res, float(Y.min()) - pad - res, float(X.max()) + pad + res, float(Y.max()) + pad + res)
    m = occupancy_mask(scene, box, res, kinds, pad); H, W = m.shape
    j = np.clip(np.round((X - box[0]) / res).astype(int), 0, W - 1); i = np.clip(np.round((Y - box[1]) / res).astype(int), 0, H - 1)
    return m[i, j].astype(bool)


def occupied(scene, box, res=0.5, kinds=None, pad=0.0):
    """→ a function (x, y) → whether occupied (by the mask over box; outside the box — free). No layer — always free."""
    if not occupancy(scene, kinds): return lambda x, y: False
    x0, y0, x1, y1 = box; m = occupancy_mask(scene, box, res, kinds, pad); H, W = m.shape
    def f(x, y):
        j, i = int(round((x - x0) / res)), int(round((y - y0) / res))
        return bool(0 <= i < H and 0 <= j < W and m[i, j])
    return f


# ── anchors ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def anchors(scene):
    """Scene anchors: of the relief (the Site's location.json) and of buildings (anchors.json) — {name: [x, y, z]}; buildings have their own names."""
    out = {}
    lp = os.path.join(scene, "location.json")
    if os.path.exists(lp):
        try: out.update((json.load(open(lp)).get("anchors") or {}))
        except (OSError, ValueError): pass
    out.update(read(scene, "anchors") or {})
    SEEN["anchors"] = SEEN.get("anchors", False) or bool(out)            # anchors exist if the site or the buildings have them
    return {k: [float(v) for v in p] for k, p in out.items() if isinstance(p, (list, tuple)) and len(p) >= 2}


def anchor(scene, name, default=None):
    return anchors(scene).get(name, default)


# ── masonry surfaces, canopy, cameras ──────────────────────────────────────────────────────────────────────────────────────
def surfaces(scene):
    return read(scene, "surfaces") or {}


def canopy(scene):
    """Canopy trees: [{x, y, cr, h, species, habit}, …]."""
    return (read(scene, "canopy") or {}).get("trees") or []


def influence(scene):
    """The settlement influence raster: {"T": float32 0…1, "x0", "y0", "step"} or None."""
    p = path(scene, "influence")
    SEEN["influence"] = SEEN.get("influence", False) or os.path.exists(p)
    if not os.path.exists(p): return None
    Z = np.load(p)
    return {"T": Z["T"].astype(np.float32) / 255.0, "x0": float(Z["x0"]), "y0": float(Z["y0"]), "step": float(Z["step"])}


def cameras(scene):
    """Scene cameras (cameras.json): a list of descriptions; positions — camera_points()."""
    c = read(scene, "cameras")
    return c if isinstance(c, list) else []


def camera_points(scene):
    """Scene camera positions: [(x, y, z), …]."""
    out = []
    for c in cameras(scene):
        M = c.get("matrix_world")
        if M: out.append(tuple(float(v) for v in np.asarray(M, float)[:3, 3]))
    return out
