"""Run stages: pinning the approved cameras and comparing the stages. A tool for the caller when moving from one stage to
the next, not a gate.
    .venv/bin/python -m core.services.stage pin <run> [--names take_,orbit_,cam_] [--shots 0,2] [--scene <scene folder>]
    .venv/bin/python -m core.services.stage unpin <scene folder> [--names …] [--shots …]
    .venv/bin/python -m core.services.stage diff <run A> <run B> [--top 12]
pin — the run's camera records (<run>/scene/cameras.json; names are matched by prefix from --names, without it — all) →
cameras.json of the source scene (the run's context.json or --scene) with a pinned field "<run>, <date>". The manager
puts this file into the scene of every run; the writers of the camera layer (Camera, Time) do not recompute pinned
records — they build from them (core/layers.py, header). unpin — remove the pinned ones.
diff — the layers of two runs (the run's scene files: hashes), cameras by name (shift, gaze turn, lens, frame), sk_id
object boxes (objects.json of the last "frames" output: the sketch's stand-in against the preview's model) → the main
mismatches as lines and __RESULT__."""
import glob, hashlib, json, math, os, time

import numpy as np

import core.paths

ROOT = core.paths.ROOT
LAYERS = ("location.json", "relief.npz", "surface.json", "occupancy.json", "anchors.json", "surfaces.json", "influence.npz", "canopy.json",
          "figkit.json")                                          # decision layers: equal at all stages; cameras — by records


def _cams(path):
    return json.load(open(path)) if os.path.exists(path) else []


def _shots(arg):
    return {int(x) for x in str(arg).split(",") if x.strip()} if arg not in (None, "") else None


def _sel(c, names=(), shots=None):
    """Is the camera record selected: the name is matched by prefix from names (empty — all); shots — the shot numbers of the
    take (take_NNN: field shot.n)."""
    nm = str(c.get("name", ""))
    if names and not nm.startswith(names): return False
    if shots is None: return True
    sh = c.get("shot")
    return nm.startswith("take_") and isinstance(sh, dict) and sh.get("n") in shots


def _names(arg):
    return tuple(x for x in (arg or "").split(",") if x)


def _source(run, scene):
    if scene: return os.path.abspath(os.path.join(ROOT, scene))
    cp = os.path.join(run, "context.json")
    if not os.path.exists(cp): raise SystemExit(f"{run}: no context.json — give --scene")
    return json.load(open(cp))["scene"]


def pin(run, names=(), scene=None, shots=None):
    run = os.path.abspath(os.path.join(ROOT, run)); src = _source(run, scene)
    rec = [c for c in _cams(os.path.join(run, "scene", "cameras.json")) if _sel(c, names, shots)]   # a run of any stage
    if not rec: raise SystemExit(f"{run}: no camera records{' ' + ','.join(names) if names else ''}")
    mark = f"{os.path.relpath(run, ROOT)}, {time.strftime('%Y-%m-%d %H:%M')}"
    dst = os.path.join(src, "cameras.json"); have = _cams(dst); by = {c.get("name"): i for i, c in enumerate(have)}
    for c in rec:
        c = dict(c, pinned=mark)
        if c.get("name") in by: have[by[c["name"]]] = c
        else: have.append(c)
    json.dump(have, open(dst, "w"), ensure_ascii=False, indent=1)
    return {"pinned": len(rec), "scene": os.path.relpath(src, ROOT), "mark": mark}


def unpin(scene, names=(), shots=None):
    dst = os.path.join(os.path.abspath(os.path.join(ROOT, scene)), "cameras.json"); have = _cams(dst)
    keep = [c for c in have if not (c.get("pinned") and _sel(c, names, shots))]
    if keep: json.dump(keep, open(dst, "w"), ensure_ascii=False, indent=1)
    elif os.path.exists(dst): os.remove(dst)
    return {"unpinned": len(have) - len(keep), "left": len(keep)}


VOLATILE = {"s", "sec", "seconds", "elapsed"}                     # compute time in the layer (location.json: s — seconds per part) — not a decision


def _drop(d):
    if isinstance(d, dict): return {k: _drop(v) for k, v in d.items() if k not in VOLATILE}
    if isinstance(d, list): return [_drop(v) for v in d]
    return d


def _sig(p):
    """Layer signature: JSON — by content without the compute time (VOLATILE), anything else — by bytes."""
    h = hashlib.sha256()
    if p.endswith(".json"):
        try: h.update(json.dumps(_drop(json.load(open(p))), sort_keys=True, ensure_ascii=False).encode()); return h.hexdigest()[:16]
        except (OSError, ValueError): pass
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()[:16]


def _view(c):
    """Position and look direction of a camera record (the matrix's −z)."""
    M = np.asarray(c["matrix_world"], float)
    d = -M[:3, 2]; return M[:3, 3], d / max(np.linalg.norm(d), 1e-9)


def _objects(run):
    fs = sorted(glob.glob(os.path.join(run, "[0-9][0-9]_frames", "objects.json")))
    return json.load(open(fs[-1])) if fs else None


def diff(a, b, top=12):
    a, b = (os.path.abspath(os.path.join(ROOT, x)) for x in (a, b)); SA, SB = os.path.join(a, "scene"), os.path.join(b, "scene")
    out = {"layers": {}, "cameras": {}, "objects": {}}
    for f in LAYERS:
        pa, pb = os.path.join(SA, f), os.path.join(SB, f)
        if os.path.exists(pa) or os.path.exists(pb):
            out["layers"][f] = "equal" if os.path.exists(pa) and os.path.exists(pb) and _sig(pa) == _sig(pb) else \
                ("varied" if os.path.exists(pa) and os.path.exists(pb) else ("missing in B" if os.path.exists(pa) else "missing in A"))
    CA = {c.get("name"): c for c in _cams(os.path.join(SA, "cameras.json"))}; CB = {c.get("name"): c for c in _cams(os.path.join(SB, "cameras.json"))}
    rows = []
    aspect = lambda c: (lambda r: r[0] / max(r[1], 1))(c.get("resolution") or [16, 9])   # the resolution is lower in the sketch — the frame shape (aspect) is compared
    for n in sorted(set(CA) & set(CB)):
        (pa, da), (pb, db) = _view(CA[n]), _view(CB[n]); x, y = CA[n], CB[n]
        frame = (x.get("frame") != y.get("frame") or abs(aspect(x) - aspect(y)) > 0.005 * aspect(x)
                 or any(abs(float(x.get(k, 0)) - float(y.get(k, 0))) > 1e-4 for k in ("shift_x", "shift_y")))   # frame, aspect, shift
        rows.append((n, float(np.linalg.norm(pa - pb)), math.degrees(math.acos(max(-1.0, min(1.0, float(da @ db))))),
                     float(y.get("lens_mm", 0)) - float(x.get("lens_mm", 0)), frame))
    moved = [r for r in rows if r[1] > 0.005 or r[2] > 0.05 or abs(r[3]) > 1e-3 or r[4]]
    out["cameras"] = {"common": len(rows), "only_a": sorted(set(CA) - set(CB)), "only_b": sorted(set(CB) - set(CA)), "moved": len(moved),
                      "worst": [{"name": n, "shift_m": round(s, 3), "turn_deg": round(t, 2), "lens_mm": round(l, 2), "frame": fr}
                                for n, s, t, l, fr in sorted(moved, key=lambda r: -(r[1] + r[2]))[:top]]}
    OA, OB = _objects(a), _objects(b)
    if OA is not None and OB is not None:
        dd = []
        for k in sorted(set(OA) & set(OB)):
            ea, eb = np.subtract(OA[k]["max"], OA[k]["min"]), np.subtract(OB[k]["max"], OB[k]["min"])
            ca, cb = (np.add(OA[k]["max"], OA[k]["min"]) / 2), (np.add(OB[k]["max"], OB[k]["min"]) / 2)
            rel = float(np.max(np.abs(ea - eb) / np.maximum(np.maximum(ea, eb), 0.5)))   # relative size along the worst axis (for small objects the base is 0.5 m)
            dd.append((k, rel, float(np.linalg.norm(ca - cb)), OA[k].get("kind"), bool(OA[k].get("stand_in") or OB[k].get("stand_in")),
                       [round(float(x), 2) for x in ea], [round(float(x), 2) for x in eb]))
        bad = [r for r in dd if r[1] > 0.10 or r[2] > 0.5]           # size ±10 %, center within 0.5 m [estimate]
        out["objects"] = {"common": len(dd), "only_a": len(set(OA) - set(OB)), "only_b": len(set(OB) - set(OA)), "off": len(bad),
                          "worst": [{"sk_id": k, "size_rel": round(r, 3), "centre_m": round(c, 2), "kind": kd, "stand_in": si, "size_a": sa, "size_b": sb}
                                    for k, r, c, kd, si, sa, sb in sorted(bad, key=lambda r: -r[1])[:top]]}
    return out


def _print(d):
    L = d["layers"]; same = [f for f, v in L.items() if v == "equal"]; other = {f: v for f, v in L.items() if v != "equal"}
    print(f"layers: equal {len(same)}" + (f"; not equal: {', '.join(f'{f} ({v})' for f, v in other.items())}" if other else ""))
    C = d["cameras"]
    print(f"cameras: common {C['common']}, moved {C['moved']}" + (f", only in A: {len(C['only_a'])}" if C["only_a"] else "")
          + (f", only in B: {len(C['only_b'])}" if C["only_b"] else ""))
    for w in C["worst"]: print(f"    {w['name']}: {w['shift_m']} m, {w['turn_deg']}°" + (f", lens {w['lens_mm']:+} mm" if w["lens_mm"] else "")
                               + (", different frame, aspect or frame shift" if w["frame"] else ""))
    O = d["objects"]
    if O:
        print(f"sk_id boxes: common {O['common']}, out of tolerance {O['off']}" + (f", only in A: {O['only_a']}" if O["only_a"] else "")
              + (f", only in B: {O['only_b']}" if O["only_b"] else ""))
        for w in O["worst"]: print(f"    {w['sk_id']} ({w['kind']}{', stand-in' if w['stand_in'] else ''}): size {w['size_a']} → {w['size_b']} "
                                   f"({w['size_rel']:.0%}), center {w['centre_m']} m")
    else: print("sk_id boxes: no objects.json of the 'frames' output in one of the runs")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="run stages: pin | unpin | diff")
    ap.add_argument("cmd", choices=["pin", "unpin", "diff"]); ap.add_argument("a"); ap.add_argument("b", nargs="?")
    ap.add_argument("--names", default=""); ap.add_argument("--shots", default=None); ap.add_argument("--scene"); ap.add_argument("--top", type=int, default=12)
    x = ap.parse_args()
    if x.cmd == "pin":
        r = pin(x.a, _names(x.names), x.scene, _shots(x.shots)); print(f"camera records pinned: {r['pinned']} → {r['scene']}/cameras.json ({r['mark']})")
    elif x.cmd == "unpin":
        r = unpin(x.a, _names(x.names), _shots(x.shots)); print(f"pins removed: {r['unpinned']}, records left: {r['left']}")
    else:
        if not x.b: ap.error("diff: two runs are needed")
        r = diff(x.a, x.b, x.top); _print(r)
    print("__RESULT__" + json.dumps(r, ensure_ascii=False))
