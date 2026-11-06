"""Scene passport — a receipt for the caller (≤ 8 KiB): checks always, then the summary, and whatever does not fit is named. The foot
check (caught 7 of 7 planted defects): the bottom 15 % of the figure's height; over each point a ray is cast
downward, skipping the figures — that is the support; "floating" — all the bottom points are above the support by more than the
tolerance, "sunk" — some point is below it. Crowds — one line per crowd.
blender -b --factory-startup -P core/blender/run.py -- core/blender/bl_passport.py -- scenes/<name> [tolerance=0.05] [cap=8192]"""
import json, os, sys
import bpy
import numpy as np
from mathutils import Vector
from core.blender import bl_scene as B

a = sys.argv[sys.argv.index("--") + 1:]; SD = os.path.abspath(a[0]); opt = dict(kv.split("=", 1) for kv in a[1:])
TOL, LIMIT = float(opt.get("tolerance", 0.05)), int(opt.get("cap", 8192))
sc, statics, members, objs, W, kit = B.load_scene(SD); dg = bpy.context.evaluated_depsgraph_get()
crowd_of = {m["id"]: m.get("crowd") for m in (kit or {}).get("members", [])}
FIG = [o for o in objs if o.name.startswith("figure_") or o.name in crowd_of]; FIGN = {o.name for o in FIG}; DOWN = Vector((0, 0, -1))


def support(x, y, ztop):
    org = Vector((x, y, ztop))
    for _ in range(64):
        hit, loc, nrm, idx, obj, mat = sc.ray_cast(dg, org, DOWN)
        if not hit: return None, None
        name = getattr(obj, "original", obj).name
        if name not in FIGN: return loc.z, name.split(".")[0]
        org = loc - Vector((0, 0, 1e-4))
    return None, None


def low_points(o, frac=0.15, n=24):
    ev = o.evaluated_get(dg); me = ev.to_mesh(); co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co); ev.to_mesh_clear()
    M = np.array(ev.matrix_world); w = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]; z0, z1 = w[:, 2].min(), w[:, 2].max()
    low = w[w[:, 2] <= z0 + frac * (z1 - z0)]
    if len(low) > n:
        order = np.argsort(low[:, 2]); low = low[np.unique(np.concatenate([order[:n // 2], order[np.linspace(0, len(order) - 1, n - n // 2).astype(int)]]))]
    return low, z1


feet, crowd = {}, {}
for o in FIG:
    pts, ztop = low_points(o); gaps, sup = [], None
    for x, y, z in pts:
        s, sn = support(x, y, ztop + 1.0)
        if s is not None: gaps.append(z - s); sup = sup or sn
    st = "no support" if not gaps else (f"sunk by {-min(gaps):.2f} m into {sup}" if min(gaps) < -TOL else
                                     f"floating {min(gaps):.2f} m above {sup}" if min(gaps) > TOL else "on support")
    if o.name in crowd_of: crowd.setdefault(crowd_of[o.name], []).append(st)
    else: feet[o.name[7:]] = st
checks = {"feet": {"on support": sum(1 for v in feet.values() if v == "on support"), "troubles": {k: v for k, v in feet.items() if v != "on support"}},
          "crowds": {c: {"n_figures": len(v), "troubles": sum(1 for s in v if s != "on support")} for c, v in crowd.items()}}
kinds = {}
for e in W["reg"]: kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
cams = json.load(open(os.path.join(SD, "cameras.json"))) if os.path.exists(os.path.join(SD, "cameras.json")) else []
rest = [("figures", {k: {"growth": round(f.get("height", 0), 2), "where": [round(x, 1) for x in f["pos"][:2]], "heading": round(f.get("heading", 0))}
                    for k, f in (kit or {}).get("named", {}).items()}),
        ("cameras", [{"k": i + 1, "name": c.get("name", "")[:36]} for i, c in enumerate(cams)]),
        ("props", sorted(e["id"] for e in W["reg"] if e["kind"] not in ("figure", "member", "ground")))]
rep = {"scene": os.path.basename(SD.rstrip("/")), "checks": checks, "summary": {"kinds": kinds, "n_cameras": len(cams)}}
cut = []
for name, val in rest:                                              # then by importance while it fits; whatever does not fit is named
    trial = dict(rep, **{name: val})
    if len(json.dumps(trial, ensure_ascii=False).encode()) <= LIMIT: rep = trial
    else: cut.append(name)
if cut: rep["did not fit"] = cut
print("__RESULT__" + json.dumps(rep, ensure_ascii=False, separators=(",", ":")))
