"""A Blender scene (.blend) → a pipeline scene scenes/<name>/ (scene.obj, world.json, cameras.json, figkit.json): then the usual
make_frame.sh / frames.py without changes. The bridge "a scene from the library graph → the frame map package" and a single figure
insertion for all skeleton kinds (a humanoid rig, a biped rig).
blender -b --factory-startup -P core/blender/run.py -- core/blender/bl_export_scene.py -- <file.blend> <scenes/name> [frame]
Scene parts are the top-level objects (their meshes together, with the pose at the frame); a figure is a part with a human skeleton:
the group figure_<id>, 18 COCO points from the bone table. The part kind (wall, cover, figure…) comes from the model's role in
the library graph, when one is present."""
import core.paths        # noqa: E402  core parts on sys.path
import json, math, os, sys
import bpy
from mathutils import Vector

ROOT = core.paths.ROOT
a = sys.argv[sys.argv.index("--") + 1:]
src, dst = os.path.abspath(a[0]), os.path.abspath(a[1]); frame = int(a[2]) if len(a) > 2 else None
bpy.ops.wm.open_mainfile(filepath=src, load_ui=False)
sc = bpy.context.scene
if frame is not None: sc.frame_set(frame)
bpy.context.view_layer.update()
os.makedirs(dst, exist_ok=True)

from core.assets.rig_joints import COCO   # noqa: E402
from core.outputs import bl_parts  # noqa: E402   the split into parts is shared with maps built directly from a .blend (core/outputs/bl_parts.py)

parts, reg, named, heights = bl_parts.collect(sc)
V = [v for p in parts for v in p["V"]]
with open(os.path.join(dst, "scene.obj"), "w") as f:
    f.write("# scene from " + os.path.basename(src) + " (core/blender/bl_export_scene.py)\n")
    f.write("".join(f"v {v.x:.4f} {v.y:.4f} {v.z:.4f}\n" for v in V))
    off = 0
    for p in parts:
        f.write(f"o {p['name']}\n" + "".join("f " + " ".join(str(off + i + 1) for i in q) + "\n" for q in p["F"]))
        off += len(p["V"])
groups = parts
# --- the scene camera and the world
cams = []
for cam in [sc.camera] if sc.camera else []:
    cd = cam.data
    cams.append({"name": cam.name, "lens_mm": cd.lens, "sensor_width_mm": cd.sensor_width, "sensor_height_mm": cd.sensor_height,
                 "sensor_fit": cd.sensor_fit, "shift_x": cd.shift_x, "shift_y": cd.shift_y,
                 "resolution": [sc.render.resolution_x, sc.render.resolution_y], "matrix_world": [list(r) for r in cam.matrix_world]})
json.dump(cams, open(os.path.join(dst, "cameras.json"), "w"), ensure_ascii=False, indent=1)
sun = next((o for o in sc.objects if o.type == "LIGHT" and o.data.type == "SUN"), None)
sun_dir = (sun.matrix_world.to_3x3() @ Vector((0, 0, 1))) if sun else None
world = {"reg": reg, "world": {"sun": {"azimuth": round(math.degrees(math.atan2(sun_dir.y, sun_dir.x)), 1) if sun else 220,
                                       "elevation": round(math.degrees(math.asin(max(-1, min(1, sun_dir.z)))), 1) if sun else 30},
                               "sky": {"top": 0.7, "horizon": 0.9}, "haze": {"distance": 700, "tone": 0.8}}, "ground_tone": 0.36, "sprites": []}
json.dump(world, open(os.path.join(dst, "world.json"), "w"), ensure_ascii=False)
json.dump({"coco": COCO, "templates": {}, "members": [], "named": named, "named_heights": heights}, open(os.path.join(dst, "figkit.json"), "w"), ensure_ascii=False)
# who is who for an image prompt: from the library job next to the .blend
# (job.json) — the kind by armour, the weapon, the pose by clip; figures m0, m1 … follow the job order.
# Without a job nothing is written (then the description is manual).
NOUN = {}
COLOUR = {"blue": (0.1, 0.2, 0.6), "red": (0.6, 0.08, 0.06), "yellow": (0.8, 0.65, 0.1), "green": (0.1, 0.4, 0.15), "black": (0.05, 0.05, 0.05),
          "white": (0.85, 0.85, 0.8), "grey": (0.4, 0.4, 0.42), "purple": (0.35, 0.1, 0.4), "orange": (0.8, 0.35, 0.05), "brown": (0.35, 0.22, 0.1)}


def describe(f):
    race = next((r for r in NOUN if f"/{r}/" in f.get("armour", "") + f.get("clip", "")), None)
    noun = NOUN.get(race, "a warrior"); pal = f.get("palette")
    if isinstance(pal, list) and pal and isinstance(pal[0], list):
        arm = f.get("armour", ""); kind_ = ("terminator armour" if "terminator" in arm else "power armour" if "power_armour" in arm
                                            else "carapace armour" if "carapace" in arm else "armour")
        c = pal[0]; noun += f" in {min(COLOUR, key=lambda k: sum((a - b) ** 2 for a, b in zip(COLOUR[k], c)))} {kind_}"
    w = [os.path.basename(x).rsplit(".", 1)[0] for x in f.get("weapons", [])]
    w = [" ".join(t for t in n.split("_") if t not in ("common", "rare", "epic", "legendary", "black", "rage")) for n in w]
    wp = ("a " + " and a ".join(w)) if w else "a weapon"
    c = os.path.basename(f.get("clip", "")).rsplit(".", 1)[0]
    act = ("aiming " + wp if "aim" in c else "firing " + wp if "fire" in c else "running" if "run" in c else "walking" if "walk" in c
           else "swinging " + wp if ("melee" in c or "attack" in c) else "falling" if ("die" in c or "death" in c) else "standing ready")
    return f"{noun} {act}" + (" from a crouch" if "crouch" in c else "")


json.dump({"blend": src, "ids": [r["id"] for r in reg]}, open(os.path.join(dst, "source.json"), "w"), ensure_ascii=False)   # the frame color comes from the .blend (bl_color.py);
# ids are the parts from the .blend; the ones appended later (append_parts: trees, ridges, banners) bl_color
# takes from scene.obj
if os.path.exists(os.path.join(dst, "edits.json")): os.remove(os.path.join(dst, "edits.json"))
job = os.path.join(os.path.dirname(src), "job.json")
if os.path.exists(os.path.join(dst, "lib.json")): os.remove(os.path.join(dst, "lib.json"))   # left from the previous export — the wrong squad
if os.path.exists(job):
    figs = (json.load(open(job)).get("renders") or [{}])[0].get("squad", {}).get("figures", [])
    if len(figs) == len(named):
        order = sorted(named, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
        json.dump({"_note": "from the library job (job.json): race, weapon, pose by clip", "subjects": {k: {"kind": "figure", "desc": describe(f)}
                   for k, f in zip(order, figs)}}, open(os.path.join(dst, "lib.json"), "w"), ensure_ascii=False, indent=1)
        for k, f in zip(order, figs):                               # creatures (non-humanoids) do not need a human skeleton in the pose map:
            if "/creature/" in f.get("armour", "") + f.get("clip", ""): named[k]["pose"] = False   # ControlNet would draw humans
        json.dump({"coco": COCO, "templates": {}, "members": [], "named": named, "named_heights": heights}, open(os.path.join(dst, "figkit.json"), "w"), ensure_ascii=False)
print("__RESULT__" + json.dumps({"parts": len(groups), "verts": len(V), "figures": len(named), "rigs": sorted({n["rig"] for n in named.values()}),
                                 "mirrored": sum(n["mirrored"] for n in named.values()),
                                 "kinds": {k: sum(r["kind"] == k for r in reg) for k in sorted({r["kind"] for r in reg})}, "cameras": len(cams)}, ensure_ascii=False))
