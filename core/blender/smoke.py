"""Headless check: a cube + ground + camera -> PNG in Workbench and Eevee.
Run: blender -b -P core/blender/run.py -- core/blender/smoke.py -- <output_folder>"""
import sys, os, math, time
import bpy

out = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "out/smoke"
os.makedirs(out, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.resolution_x, sc.render.resolution_y = 640, 360

bpy.ops.mesh.primitive_plane_add(size=10)
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
cam.location = (4, -4, 2.5)
cam.rotation_euler = (math.radians(70), 0, math.radians(45))
cam.data.lens, cam.data.sensor_width = 35, 36
sc.camera = cam
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.rotation_euler = (math.radians(50), 0, math.radians(30))
sc.collection.objects.link(sun)

print("Blender", bpy.app.version_string)
for name in ("BLENDER_WORKBENCH", "BLENDER_EEVEE"):
    sc.render.engine = name
    sc.render.filepath = os.path.join(out, name.lower() + ".png")
    t = time.time()
    try:
        bpy.ops.render.render(write_still=True)
        print(name, "ok", round(time.time() - t, 2), "s")
    except Exception as e:
        print(name, "FAIL", e)
