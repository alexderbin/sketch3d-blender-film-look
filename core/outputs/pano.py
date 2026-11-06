"""A 360° panorama of a lit scene (Eevee has no panoramic camera): 6 cube faces of 90° with a single Blender (the scene preparation is
core/outputs/bl_color.py light=scene, as for the pipeline frames), stitched → an equirectangular 2:1 (viewable in any 360 viewer) and
a 360° × [bottom, top]° band — a "big panorama" for choosing frames.
.venv/bin/python -m core.outputs.pano scenes/<name> <folder> <x,y,z> [face=1024] [samples=32] [band=-25,40] [light=far|scene|…] [sky_image=<image 2:1>[:rotation]]
The default light is the scene's light-and-sky node (light=far: sky, haze, far terrain, the scene's lamps under the node — core/light/bl_far.py) if the scene has a sky
described (far.yaml, far: in battle.yaml, a site); otherwise — the scene's own light. A face is a pipeline frame (bl_color.render_colour) → lit.png.
Output: <folder>/faces/*.png, pano_360.jpg (4N×2N), pano_band.jpg; the __RESULT__ line (Blender and stitching time)."""
import core.paths        # noqa: E402  core parts on sys.path
import json, math, os, subprocess, sys, time
import numpy as np
from PIL import Image
ROOT = core.paths.ROOT
a = sys.argv[1:]; scene, out, pos = a[0], a[1], a[2]; opt = dict(x.split("=", 1) for x in a[3:])
N = int(opt.get("face", 1024)); band = [float(v) for v in opt.get("band", "-25,40").split(",")]
os.makedirs(os.path.join(out, "faces"), exist_ok=True)
FACES = {"down": (0, 0, 0), "up": (math.pi, 0, 0), "north": (math.pi / 2, 0, 0), "south": (math.pi / 2, 0, math.pi),
         "east": (math.pi / 2, 0, -math.pi / 2), "west": (math.pi / 2, 0, math.pi / 2)}
from core.light import far
LIGHT = opt.get("light") or ("scene" if far.load_spec(scene).get("node") == "world.json" else "far")   # the node — if the sky is described
code = f"""import sys, os, math, json
sys.argv = [sys.argv[0], "--", {scene!r}, {os.path.join(out, 'faces')!r}, "1", "light={LIGHT}"] + ([f"samples={opt.get('samples')}"] if {bool(opt.get('samples'))} else []) + ([f"sky_image={opt.get('sky_image')}"] if {bool(opt.get('sky_image'))} else [])
sys.path.insert(0, {os.path.join(ROOT, 'core')!r}); import kernel
import bpy, bl_color as BC
from mathutils import Euler, Matrix, Vector
dirs = []
for i, (nm, rot) in enumerate({FACES!r}.items()):                      # a face is a pipeline frame: the scene's light or the node (sky, haze, far terrain)
    M = Matrix.LocRotScale(Vector(({pos})), Euler(rot).to_matrix(), None)
    spec = {{"matrix_world": [list(r) for r in M], "lens_mm": 18.0, "sensor_width_mm": 36.0, "sensor_fit": "HORIZONTAL", "resolution": [{N}, {N}]}}
    d = os.path.join({os.path.join(out, 'faces')!r}, nm); BC.render_colour(i + 1, spec, d, meter=False); dirs.append(d); print("FACE", nm, flush=True)
BC.finish_far(dirs)                                                     # light=far: processing of the faces (haze, sky, AgX) → lit.png
print("__RESULT__" + json.dumps({{"faces": 6, "light": {LIGHT!r}}}))"""
cp = os.path.join(out, "faces", "_pano_bl.py"); open(cp, "w").write(code)
t0 = time.time(); r = subprocess.run(["/usr/bin/time", "-l", "blender", "-b", "--factory-startup", "-P", core.paths.RUN, "--", cp], capture_output=True, text=True, cwd=ROOT)
open(os.path.join(out, "faces", "blender.log"), "w").write(r.stdout[-20000:] + r.stderr[-5000:])
peak = next((int(l.split()[0]) for l in r.stderr.splitlines() if "maximum resident set size" in l), 0)
if "__RESULT__" not in r.stdout: print("__RESULT__" + json.dumps({"error": "blender", "tail": (r.stdout + r.stderr)[-1500:]})); sys.exit(1)
t1 = time.time()


def rot(e):
    """Blender Euler XYZ → matrix (the camera looks along −Z, up is +Y)."""
    cx, sx, cy, sy, cz, sz = math.cos(e[0]), math.sin(e[0]), math.cos(e[1]), math.sin(e[1]), math.cos(e[2]), math.sin(e[2])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]); Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]); Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def project(lon, lat):
    """Directions (longitude clockwise from north, latitude) → color from the cube faces (the nearest point of a face)."""
    d = np.stack([np.sin(lon) * np.cos(lat), np.cos(lon) * np.cos(lat), np.sin(lat)], -1)
    out_ = np.zeros(lon.shape + (3,), np.uint8); best = np.full(lon.shape, -2.0)
    for nm, e in FACES.items():
        R = rot(e); c = d @ R                                          # in camera axes (R is orthogonal: Rᵀ·d)
        z = -c[..., 2]; ok = z > 1e-6; u = np.where(ok, c[..., 0] / np.maximum(z, 1e-6), 9); v = np.where(ok, c[..., 1] / np.maximum(z, 1e-6), 9)
        inside = ok & (np.abs(u) <= 1) & (np.abs(v) <= 1) & (z > best)
        im = np.asarray(Image.open(os.path.join(out, "faces", nm, "lit.png")).convert("RGB"))
        col = np.clip(((u + 1) / 2 * (N - 1)).round().astype(int), 0, N - 1); row = np.clip(((1 - v) / 2 * (N - 1)).round().astype(int), 0, N - 1)
        out_[inside] = im[row[inside], col[inside]]; best = np.where(inside, z, best)
    return out_


W, Hh = 4 * N, 2 * N
lon = (np.arange(W) + 0.5) / W * 2 * np.pi - np.pi; lat = np.pi / 2 - (np.arange(Hh) + 0.5) / Hh * np.pi
Lo, La = np.meshgrid(lon, lat); Image.fromarray(project(Lo, La)).save(os.path.join(out, "pano_360.jpg"), quality=90)
bh = int(W * math.radians(band[1] - band[0]) / (2 * math.pi)); la2 = np.radians(band[1]) - (np.arange(bh) + 0.5) / bh * np.radians(band[1] - band[0])
Lo2, La2 = np.meshgrid(lon, la2); Image.fromarray(project(Lo2, La2)).save(os.path.join(out, "pano_band.jpg"), quality=90)
print("__RESULT__" + json.dumps({"pano": os.path.join(out, "pano_360.jpg"), "band": os.path.join(out, "pano_band.jpg"), "size": [W, Hh],
                                 "s_blender": round(t1 - t0, 1), "peak_gb": round(peak / 2 ** 30, 2), "s_stitch": round(time.time() - t1, 1)}, ensure_ascii=False))
