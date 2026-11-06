"""A pipeline scene summary for the caller — text of 20–60 lines instead of reading scene.obj and figkit.json (rows of
"name | kind | where | size"; the scene files are parsed here directly, without Blender).
.venv/bin/python -m core.model.scene_info scenes/<name> [--json]
What the summary has: the scene box and ground level; parts by kind (how many, sizes); figures (where, heading, height); cameras (where, where they
look, lens, intent) and which figures are in each one's frame. parts_of() — part boxes for other tools.
First comes the scene model by section (core: core/model/scene_model.py); for a scene without a gray copy it is the only source."""
import json, math, os, sys
from collections import defaultdict
import numpy as np


def parts_of(SD):
    """Scene parts: the kind from world.json, the box from face vertices (the bridge writes the vertices before "o"), the ground level."""
    W = json.load(open(os.path.join(SD, "world.json")))
    kind = {e["id"]: e["kind"] for e in W.get("reg", [])}
    V, idx, cur = [], defaultdict(list), None
    with open(os.path.join(SD, "scene.obj")) as f:
        for line in f:
            if line.startswith("v "): V.append(line.split()[1:4])
            elif line.startswith("o "): cur = line[2:].strip()
            elif line.startswith("f ") and cur is not None: idx[cur] += [int(t.split("/")[0]) for t in line.split()[1:]]
    V = np.array(V, float); lo, hi = {}, {}
    for n, ii in idx.items():
        ii = np.array(ii); P = V[np.where(ii > 0, ii - 1, len(V) + ii)]; lo[n], hi[n] = P.min(0), P.max(0)
    ground = [p for p in lo if kind.get(p) in ("ground", "terrain", "floor")]
    gv = np.concatenate([V[np.array(idx[p]) - 1, 2] for p in ground]) if ground else V[:, 2]
    zs, cnt = np.unique(np.round(gv, 1), return_counts=True)            # ground level — the most frequent height of its vertices
    return kind, lo, hi, float(zs[cnt.argmax()])


def figures_of(SD):
    p = os.path.join(SD, "figkit.json")
    kit = json.load(open(p)) if os.path.exists(p) else {}
    return kit.get("named", {}), len(kit.get("members", []))


def summary(SD):
    from core.outputs import maps                                                     # cv2, OpenEXR, PIL — only in the .venv; parts_of() works in Blender too
    kind, lo, hi, gz = parts_of(SD)
    parts = sorted(lo)
    allo, alhi = np.min([lo[p] for p in parts], 0), np.max([hi[p] for p in parts], 0)
    by_kind = defaultdict(list)
    for p in parts:
        if not p.startswith("figure_"): by_kind[kind.get(p, "?")].append(p)
    figs, n_crowd = figures_of(SD)
    lp = os.path.join(SD, "lib.json")                               # who is who (written by the library bridge) — to choose the panel's hero
    desc = {k: v.get("desc", "") for k, v in json.load(open(lp)).get("subjects", {}).items()} if os.path.exists(lp) else {}
    cams = json.load(open(os.path.join(SD, "cameras.json"))) if os.path.exists(os.path.join(SD, "cameras.json")) else []
    out = {"scene": os.path.basename(os.path.abspath(SD)), "bbox_min": allo.round(1).tolist(), "bbox_max": alhi.round(1).tolist(),
           "ground_z": round(gz, 2), "parts": len(parts),
           "kinds": {k: {"n": len(v), "size_max": np.max([hi[p] - lo[p] for p in v], 0).round(1).tolist(), "ids": v[:6]} for k, v in sorted(by_kind.items())},
           "figures": {n: {"pos": np.round(f["pos"], 2).tolist(), "heading": round(f.get("heading") or 0), "height": round(f.get("height", 0), 2),
                           **({"desc": desc[n]} if n in desc else {})} for n, f in figs.items()},
           "crowd_members": n_crowd, "cameras": []}
    for i, c in enumerate(cams, 1):
        cam = maps.Cam(c); fw = cam.fwd
        inside = []
        for n, f in figs.items():
            u, v, z = cam.project(np.array([f["pos"], np.array(f["pos"]) + [0, 0, f.get("height", 1.8)]]))
            if np.all(z > 0) and ((0 <= u) & (u < cam.W) & (0 <= v) & (v < cam.H)).any():
                inside.append(f"{n} {int(abs(v[0] - v[1]) / cam.H * 100)}%")       # fraction of the frame height
        out["cameras"].append({"k": i, "name": c.get("name"), "pos": cam.pos.round(1).tolist(), "yaw": round(math.degrees(math.atan2(fw[1], fw[0]))),
                               "pitch": round(math.degrees(math.asin(np.clip(fw[2], -1, 1)))), "lens": c.get("lens_mm"), "res": c.get("resolution"),
                               "height_above_ground": round(float(cam.pos[2] - gz), 1), "intent": c.get("intent"), "figures_in_frame": inside})
    return out


if __name__ == "__main__":
    from core.model import scene_model
    M = scene_model.load(sys.argv[1])
    if not os.path.exists(os.path.join(sys.argv[1], "scene.obj")):
        print(json.dumps({"scene": M["scene"], "model": M}, ensure_ascii=False) if "--json" in sys.argv else scene_model.summary(M)); sys.exit()
    out = summary(sys.argv[1])
    out["model"] = {k: M[k] for k in ("explicit", "origin", "sections", "build")}
    if "--json" in sys.argv:
        print(json.dumps(out, ensure_ascii=False)); sys.exit()
    print(scene_model.summary(M))
    print(f"scene {out['scene']}: parts {out['parts']}, box {out['bbox_min']} … {out['bbox_max']}, ground z={out['ground_z']}")
    for k, v in out["kinds"].items():
        print(f"  {k}: {v['n']} pcs, largest {v['size_max'][0]}×{v['size_max'][1]}×{v['size_max'][2]} m — {', '.join(v['ids'])}{' …' if v['n'] > 6 else ''}")
    print(f"figures {len(out['figures'])}" + (f", crowd {out['crowd_members']}" if out["crowd_members"] else ""))
    for n, f in out["figures"].items():
        print(f"  {n}: {f['pos']} heading {f['heading']}° height {f['height']} m" + (f" — {f['desc']}" if f.get("desc") else ""))
    for c in out["cameras"]:
        it = f" intent '{c['intent'].get('intent')}' hero {c['intent'].get('hero')}" if isinstance(c["intent"], dict) else ""
        print(f"camera {c['k']} {c['name']}: {c['pos']} ({c['height_above_ground']} m above the ground), heading {c['yaw']}°, pitch {c['pitch']}°, "
              f"{c['lens']} mm, {c['res'][0]}×{c['res'][1]}{it}; in frame: {', '.join(c['figures_in_frame']) or '—'}")
