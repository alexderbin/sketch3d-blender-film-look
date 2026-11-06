"""Frames of a sketch25 scene in Blender: Workbench and Eevee with the sun and sky from the scene's world.
Run: blender -b -P core/blender/run.py -- core/blender/bl_look.py -- <scene_folder> <camera 1..> <output_folder>"""
import json
import os
import sys

import bpy
from core.blender import bl_scene as B

scene_dir, k, out = sys.argv[sys.argv.index("--") + 1:][:3]
k = int(k)
os.makedirs(out, exist_ok=True)
W = json.load(open(os.path.join(scene_dir, "world.json")))
from core.light import bl_far
from core.light import far                                                  # the sun and sky come from the scene's light node (far.yaml / world.json; far.sketch_world);
SW = far.sketch_world(bl_far.ensure_plan(scene_dir, os.path.join(out, "node.json"))); sun, sky = SW["sun"], SW["sky"]   # the plan is read in .venv (yaml)

sc = B.reset()
objs = B.import_obj(os.path.join(scene_dir, "scene.obj"))
objs.append(B.add_far_ground(objs))
B.assign_tones(objs, W["reg"])
B.make_camera(B.load_cameras(scene_dir)[k - 1], f"cam{k}", sc)
B.add_sun(sc, sun.get("azimuth", 220), sun.get("elevation", 30), strength=3.0)
B.sky_world(sc, sky.get("top", 0.78), sky.get("horizon", 0.94))

B.look_workbench(sc)
B.render_to(sc, os.path.join(out, f"cam{k}-workbench.png"))
B.look_eevee(sc)
B.render_to(sc, os.path.join(out, f"cam{k}-eevee.png"))
print("DONE")
