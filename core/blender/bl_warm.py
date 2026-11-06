"""Own operations of the warm Blender (core/blender/bl_serve.py): called by the python step — `import bl_warm; bl_warm.look(3)`. The
scene is already in memory.
look(K, path=None, size=640) — a view through scene camera K (Workbench, no textures — a flat color by class), PNG; the reply is the path.
cams() — scene cameras: number, name, where it stands; figs() — figures: id, place, heading."""
import json, math, os
import bpy
from core.blender import bl_scene as B

SD = os.environ.get("SKETCH_SCENE", "")


def cams():
    return [{"k": i + 1, "name": c.get("name", "")[:40], "at": [round(x, 1) for x in c.get("location", [0, 0, 0])]}
            for i, c in enumerate(json.load(open(os.path.join(SD, "cameras.json"))))]


def figs():
    kit = json.load(open(os.path.join(SD, "figkit.json"))); return {k: {"pos": [round(x, 2) for x in f["pos"]], "heading": f.get("heading")}
                                                                    for k, f in kit.get("named", {}).items()}


def look(k, path=None, size=640):
    """A view through camera K: the cameras are read from the file on every call (cam_intent may have added new ones)."""
    sc = bpy.context.scene; spec = json.load(open(os.path.join(SD, "cameras.json")))[int(k) - 1]
    old = bpy.data.objects.get("warm_cam")
    if old: bpy.data.objects.remove(old, do_unlink=True)
    sc.camera = B.make_camera(spec, "warm_cam", sc)
    W, H = spec.get("resolution", [1280, 720]); s = size / max(W, H)
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = int(W * s), int(H * s), 100
    sc.render.engine = "BLENDER_WORKBENCH"; d = sc.display.shading; d.light, d.color_type, d.show_cavity = "STUDIO", "OBJECT", True
    path = path or os.path.join(os.environ.get("SKETCH_SERVE", "."), f"look_{k}.png")
    B.render_to(sc, path); return path
