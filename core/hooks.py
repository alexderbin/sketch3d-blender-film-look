"""Core extension points (ARCHITECTURE.md, "Step passport"): the core does not import components — a component declares in its
passport `hooks: {point: module:function}`, and the core calls the point's handlers in turn (by the component folder name). A
handler failure does not crash the core — a log line and the FAILED result, — unless the point is called strictly (strict).
Points and signatures:
  light            — after the world and the lamps of the light plan (core/light/bl_far.attach) and when a scene is opened by the
                     frames output (core/outputs/bl_frames_blend: what the .blend does not save): f(sc, P); repeatable
  maps_before      — before the scene is split into map parts (core/outputs/bl_parts.collect): f(sc) → a summary for the log
  maps_after       — after the maps (for those whose maps_before did not fail): f(sc)
  frame_near       — a frame with the scene light from the .blend (core/outputs/bl_color): ground clutter near the cameras, f(sc, scene, G, cams) → a summary
  frame_materials  — the same, strict: the frame's materials and weather, f(sc, scene, cams, opt, out) → {"weather", "sky"} | None
  weather          — weather presets of the frames output (core/outputs/frames.py), Python: f() → {name: {"sky": sky weather, …}}
The handler code is part of the code signature of the steps that pull in this module (manager: files()) — editing a handler rebuilds them."""
import glob
import importlib
import os

import core.paths

FAILED = object()                                                     # the result of a failed handler (not to be confused with None — "nothing to do")
_TABLE = None


def table():
    """{point: [(component, module, function)]} — from the passports components/*/component.yaml (once per process)."""
    global _TABLE
    if _TABLE is None:
        import yaml
        _TABLE = {}
        for pp in sorted(glob.glob(os.path.join(core.paths.ROOT, "components", "*", "component.yaml"))):
            try: hooks = (yaml.safe_load(open(pp, encoding="utf-8")) or {}).get("hooks") or {}
            except Exception as e: print("extension points:", os.path.relpath(pp, core.paths.ROOT), "—", e, flush=True); continue
            for point, ref in hooks.items():
                mod, _, fn = str(ref).partition(":")
                _TABLE.setdefault(point, []).append((os.path.basename(os.path.dirname(pp)), mod.strip(), fn.strip()))
    return _TABLE


def call(point, *a, strict=False, only=None, **kw):
    """The point's handlers in turn → [(component, result)]; only — only these components; a failure — a log line and FAILED
    (strict — the exception propagates: a point cannot do without a handler)."""
    out = []
    for comp, mod, fn in table().get(point, []):
        if only is not None and comp not in only: continue
        try: out.append((comp, getattr(importlib.import_module(mod), fn)(*a, **kw)))
        except Exception as e:
            if strict: raise
            print(f"point {point}: {comp} ({mod}:{fn}) — failure: {type(e).__name__} {str(e)[:200]}", flush=True); out.append((comp, FAILED))
    return out


def merged(point, *a, **kw):
    """The result dicts of a point's handlers merged into one dict (a later one wins)."""
    out = {}
    for _, r in call(point, *a, **kw):
        if isinstance(r, dict): out.update(r)
    return out


def files():
    """The files of the handler modules and of the passports that declare them (for the manager's step code signature)."""
    fs = set()
    for pts in table().values():
        for comp, mod, _ in pts:
            p = os.path.join(core.paths.ROOT, *mod.split(".")) + ".py"
            if os.path.isfile(p): fs.add(p)
            fs.add(os.path.join(core.paths.ROOT, "components", comp, "component.yaml"))
    return sorted(fs)
