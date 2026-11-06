"""Adapter to the mcp-for-blender code in place: their files are neither modified nor copied; the pipeline calls only the components
enabled in components.json (status on or patch); our patches wrap their calls, here. Inside Blender only.
Import: sys.path.insert(0, <donor folder>); import mcp4b as D. Their code location: the MCP4B_SRC variable or the default path
(projects/mcp-for-blender/src/mcp-for-blender)."""
import io
import json
import os
import sys
from contextlib import redirect_stdout

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("MCP4B_SRC") or os.path.join(HERE, "upstream")     # their files unmodified, commit 5a3618f
PKG = os.path.join(SRC, "src", "blender_mcp")
REG = {c["id"]: c for c in json.load(open(os.path.join(HERE, "components.json")))["components"]}
sys.dont_write_bytecode = True                       # no __pycache__ will appear in their folder
if PKG not in sys.path:
    sys.path.insert(0, PKG)
import blender_scripts as S                          # noqa: E402 — their file as is

_addon = _server = None
_gpu_ready = False


def need(cid):
    """Is the component enabled in the markup? Otherwise a clear refusal: what it is and why it is disabled."""
    c = REG.get(cid)
    if c is None or c["status"] not in ("on", "patch"):
        why = (c or {}).get("why") or (c or {}).get("when_do") or "not in the markup"
        raise RuntimeError(f"component {cid!r} is disabled ({(c or {}).get('status')}): {why}")


def active():
    return [c["id"] for c in REG.values() if c["status"] in ("on", "patch")]


def addon():
    """addon.py as a module: without register() — no socket, no panel, no telemetry."""
    global _addon
    if _addon is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("mcp4b_addon", os.path.join(SRC, "addon.py"))
        _addon = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_addon)
    return _addon


def server():
    global _server
    if _server is None:
        _server = addon().BlenderMCPServer()
    return _server


def run(script, **args):
    """Their observer script (LOOK, SCENE_SUMMARY, BOUNDS or custom text) as a function: arguments → dict."""
    need("run_script")
    code = getattr(S, script) if script.isupper() and hasattr(S, script) else script
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(S.build(code, args), {"bpy": bpy})
    return S.parse_result(buf.getvalue())


def _visible_meshes(names):
    out = []
    for n in names:
        o = bpy.data.objects.get(n)
        if o is None:
            continue
        for x in [o, *o.children_recursive]:
            if x.type == "MESH" and x.visible_get() and x.name not in out:
                out.append(x.name)
    return out


def look(path, mode="camera", target=None, mask=False, **args):
    """An image without a window (their LOOK). Patches: gpu.init(); the selection is cleared during the capture; the box covers visible meshes only;
    mask=True — view smoothing off (exact masks by object color)."""
    global _gpu_ready
    need("look")
    if not _gpu_ready:
        import gpu
        gpu.init()
        _gpu_ready = True
    sel = list(bpy.context.selected_objects)
    for o in sel:
        o.select_set(False)
    if target is None and mode in ("angles", "frames"):
        target = [o.name for o in bpy.context.scene.objects if o.parent is None]
    pref = bpy.context.preferences.system
    aa = pref.viewport_aa
    if mask:
        pref.viewport_aa = "OFF"
    try:
        return run("LOOK", mode=mode, target=_visible_meshes(target) if target else None, filepath=path, **args)
    finally:
        pref.viewport_aa = aa
        for o in sel:
            o.select_set(True)


def summary(**args):
    """Scene summary (their SCENE_SUMMARY). Patches: the scene is updated before the matrices are read; the ground field is removed — it measures
    from z = 0 (on a 40 m wall it reads "floating"); check support with our own ground_feet."""
    need("summary")
    bpy.context.view_layer.update()
    fields = [f for f in args.pop("fields", ["location", "size", "children", "hidden"]) if f != "ground"]
    r = run("SCENE_SUMMARY", fields=fields, **args)
    r["note"] = "the ground field is disabled by the adapter"
    return r


def bounds(names):
    need("bounds")
    bpy.context.view_layer.update()
    return run("BOUNDS", names=list(names))


def pick(cam, W, H, px, py):
    """Pixel (px, py) of the frame of camera cam, size W×H → the object and its position (their pick_viewport_object)."""
    need("pick")
    dg = bpy.context.evaluated_depsgraph_get()
    win = cam.calc_matrix_camera(dg, x=W, y=H)
    view = cam.matrix_world.inverted()
    return server().pick_viewport_object([list(r) for r in view], [list(r) for r in win], W, H, (px + 0.5) / W, (py + 0.5) / H)


def blend_version(path):
    need("blend_version")
    return addon()._polyhaven_blend_version(path)


def api(query):
    need("api")
    return server().bpy_api_lookup(query)


def node(bl_idname, **overrides):
    need("api")
    return server().describe_node_type(bl_idname, overrides or None)


def polyhaven(asset_id, asset_type, resolution="1k", file_format=None):
    need("polyhaven")
    return server().download_polyhaven_asset(asset_id, asset_type, resolution, file_format)


def guide(topic):
    """The hint text without the service header (--- title/summary ---)."""
    need("guides")
    text = open(os.path.join(PKG, "guides", f"{topic}.md"), encoding="utf-8").read()
    return text.split("\n---\n", 1)[1].lstrip() if text.startswith("---\n") else text


def call(cid):
    """For disabled components: an explicit check — raises a refusal with the reason from the markup."""
    need(cid)
