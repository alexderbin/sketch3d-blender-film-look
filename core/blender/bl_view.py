"""Windowless "eyes": a picture of the scene in a fraction of a second instead of the full map package (the LOOK of the
Blender helpers, core/ext/bl_helpers; checked: the view through a camera matches the pipeline frame by the id pass at 100 %).
blender -b --factory-startup -P core/blender/run.py -- core/blender/bl_view.py -- <bench/scenes/X | file.blend> <out.png> [camera K | viewport | frames] [key=value …]
  camera K  — through camera K from cameras.json (for a .blend — the file's active camera, K is not needed);
  viewport  — the 3D view; frames — a strip of animation frames (frames=1,10,20 or frame_count=8);
  the keys of LOOK: shading=solid|xray|wireframe, view=front|side|top|camera, max_size=768, target=name,name"""
import core.paths        # noqa: E402  core parts on sys.path
import json, os, sys
core.paths.ext("bl_helpers")
import bpy
a = sys.argv[sys.argv.index("--") + 1:]
src, out = a[0], os.path.abspath(a[1]); rest = a[2:]
mode = rest[0] if rest else "camera"; rest = rest[1:]
k = None
if mode == "camera" and rest and rest[0].isdigit(): k = int(rest[0]); rest = rest[1:]
args = {}
for kv in rest:
    key, _, v = kv.partition("=")
    args[key] = [int(x) for x in v.split(",")] if key == "frames" else (v.split(",") if key == "target" else (int(v) if v.isdigit() else v))
if src.endswith(".blend"):
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(src), load_ui=False)
else:
    from core.blender import bl_scene as B
    sc, statics, members, objs, W, kit = B.load_scene(os.path.abspath(src))
    if mode == "camera": B.make_camera(B.load_cameras(os.path.abspath(src))[(k or 1) - 1], f"cam{k or 1}", sc)
import bl_helpers as D
if mode in ("frames", "viewport") and "target" not in args:
    rigged = [o.name for o in bpy.context.scene.objects if o.type == "MESH" and o.parent and o.parent.type == "ARMATURE"]
    if rigged: args["target"] = rigged          # a take with a hero: the framing follows the hero, not the floor
if mode in ("frames", "viewport") and "view" not in args: args["view"] = "three_quarter"   # an orbit around the target; without it — the current 3D window view
res = D.look(out, mode=mode, **args)
print("__RESULT__" + json.dumps({"file": out, **({kk: vv for kk, vv in res.items() if kk in ("width", "height", "frames", "error", "views")} if isinstance(res, dict) else {})}, ensure_ascii=False))
