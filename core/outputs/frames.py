"""Map bundles for several cameras of one scene: Blender loads the scene once (core/outputs/bl_pack.py with a camera list), and the map build
(core/outputs/pack.py) of each camera goes to the background as soon as its passes are ready. A warm Blender: 1.1–1.9 s per frame
against 4.1–5.5. A failure of any step gives a non-zero exit code.
Usage: .venv/bin/python -m core.outputs.frames <scene folder> <output folder> <K1,K2,…|a-b|all> [--par 3] [--no-color]
  [--light <node time dawn … night>|overcast|blaze|far|scene] [--weather rain|storm|snow|fog|…] [--hdri <Poly Haven id>] [--fire x,y,z;…]
  [--samples N] [--sky <sky image 2:1>[:rotation°]] [pack.py flags …]
For library scenes (source.json) — also the color from the .blend (core/outputs/bl_color.py) and the previs.png previs (core/outputs/previs.py).
--weather — the run's weather, one for all outputs: a materials preset (components/materials/mat_plan.py WEATHER: rain, storm, after_rain, snow,
fog, dust_storm, after_battle, mud, clear) — the node's sky (rain: a veil up to 4 km, drizzle at the ground) and in a lit frame the
states of the figures, the world, the ground, puddles and rain streaks; without --light the lit frame uses the scene's node time. Sky weather (overcast, storm…) —
only the node's sky."""
import core.paths        # noqa: E402  core parts on sys.path
import argparse, json, os, subprocess, sys, time
ROOT = core.paths.ROOT
ap = argparse.ArgumentParser(); ap.add_argument("scene"); ap.add_argument("out"); ap.add_argument("cams"); ap.add_argument("--par", type=int, default=3)
ap.add_argument("--no-color", action="store_true")                  # without color and previs (they exist only for library scenes, source.json)
ap.add_argument("--light", default="")                              # frame light: a node time (dawn … night) | overcast | blaze | far | scene → lit.png, previs_lit.png
ap.add_argument("--hdri", default=""); ap.add_argument("--fire", default="")   # Poly Haven sky; fire x,y,z;…
ap.add_argument("--samples", default="")                            # samples of a lit frame (bl_color samples=): 16–24 are enough for a video
ap.add_argument("--weather", default="")                            # the run's weather: a materials preset or the sky weather (see above)
ap.add_argument("--sky", default="")                                # image sky: an equirectangular 2:1 [:rotation°] (components/light)
a, extra = ap.parse_known_args()
scene, out = (p if os.path.isabs(p) else os.path.join(ROOT, p) for p in (a.scene, a.out))
cams = ([str(i + 1) for i in range(len(json.load(open(os.path.join(scene, "cameras.json")))))] if a.cams == "all" else
        [str(k) for k in range(int(a.cams.split("-")[0]), int(a.cams.split("-")[1]) + 1)] if "-" in a.cams else a.cams.split(","))   # all | a-b | K1,K2
os.makedirs(out, exist_ok=True); t0 = time.time(); log = open(os.path.join(out, "blender.log"), "w")
from core.light import far
from core import hooks                                              # the run's light-and-sky node — one for the sketch, maps, previs and light
WEATHER = hooks.merged("weather")                                   # materials weather presets (rain, storm, snow…) — the weather extension point
over = far.light_over(a.light)                                      # (--light day|… is the node time; overcast, blaze also set the weather)
if a.weather:                                                       # the run's weather: the node's sky from a materials preset or as is
    if a.weather not in WEATHER and a.weather not in far.WEATHER:
        sys.exit(f"--weather {a.weather}: no such weather; available are {', '.join(WEATHER)} or {', '.join(far.WEATHER)}")
    over["weather"] = WEATHER[a.weather]["sky"] if a.weather in WEATHER else a.weather
if a.sky: over["sky_image"] = a.sky                                # the image goes to the camera, the light to the node (the luminary follows the image)
wkv = [f"weather={a.weather}"] if a.weather in WEATHER else []   # for materials and rain — into the lit frame
lit = a.light or ("far" if wkv else "")                            # materials weather without --light — a lit frame at the node's time
node = os.path.join(out, "node.json"); far.plan_of(scene, over, out=node)
bl = subprocess.Popen(["blender", "-b", "--python-exit-code", "1", "-P", core.paths.RUN, "--", os.path.join(ROOT, "core/outputs/bl_pack.py"), "--", scene,
                       ",".join(cams) + ("," if len(cams) == 1 else ""), out], stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1, cwd=ROOT)
jobs, done = [], []
for line in bl.stdout:
    log.write(line)
    if not line.startswith("__FRAME__"): continue
    raw = line.split(" ", 1)[1].strip(); k = os.path.basename(os.path.dirname(raw))[3:]
    while sum(p.poll() is None for _, p in jobs) >= a.par: time.sleep(0.02)
    jobs.append((k, subprocess.Popen([os.path.join(ROOT, ".venv/bin/python"), "-m", "core.outputs.pack", scene, k, raw,
                                      os.path.dirname(raw), *extra], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)))
rc_bl = bl.wait(); fail = []
for k, p in jobs:
    msg = p.communicate()[0]
    if p.returncode: fail.append(k); sys.stderr.write(f"pack.py, camera {k}:\n{msg[-800:]}\n")
    else: done.append(k)
if rc_bl: fail.append("blender"); sys.stderr.write(f"Blender crashed, log {os.path.join(out, 'blender.log')}\n")
color = None
if done and not a.no_color and os.path.exists(os.path.join(scene, "source.json")):   # color from the library .blend + outline → previs.png
    r = subprocess.run(["blender", "-b", "--factory-startup", "-P", core.paths.RUN, "--", os.path.join(ROOT, "core/outputs/bl_color.py"), "--", scene, out, ",".join(done), f"far={node}"],
                       capture_output=True, text=True, cwd=ROOT)
    if "__RESULT__" in r.stdout:
        subprocess.run([os.path.join(ROOT, ".venv/bin/python"), "-m", "core.outputs.previs", *[os.path.join(out, f"cam{k}") for k in done]],
                       capture_output=True, text=True, cwd=ROOT); color = len(done)
    else: sys.stderr.write("color: bl_color.py crashed\n" + r.stderr[-400:])
    if not lit:                                                     # without light, an old lit frame must not replace the reference
        for k in done:
            for f in ("lit.png", "previs_lit.png"):
                if os.path.exists(os.path.join(out, f"cam{k}", f)): os.remove(os.path.join(out, f"cam{k}", f))
    if lit and color:                                               # light goes in a second pass, the approved previs stays
        kv = [f"light={lit}", f"far={node}"] + ([f"hdri={a.hdri}"] if a.hdri else []) + ([f"fire={a.fire}"] if a.fire else []) + ([f"samples={a.samples}"] if a.samples else []) + wkv
        r = subprocess.run(["blender", "-b", "--factory-startup", "-P", core.paths.RUN, "--", os.path.join(ROOT, "core/outputs/bl_color.py"), "--", scene, out, ",".join(done), *kv],
                           capture_output=True, text=True, cwd=ROOT)
        if "__RESULT__" in r.stdout:
            subprocess.run([os.path.join(ROOT, ".venv/bin/python"), "-m", "core.outputs.previs", *[os.path.join(out, f"cam{k}") for k in done], "--lit"],
                           capture_output=True, text=True, cwd=ROOT)
        else: sys.stderr.write("light: bl_color.py crashed\n" + r.stderr[-400:])
print("__RESULT__" + json.dumps({"cams": done, "failed": fail, **({"previs": color} if color else {}), "s": round(time.time() - t0, 1),
                                 "s_per_cam": round((time.time() - t0) / max(len(cams), 1), 2)}, ensure_ascii=False))
sys.exit(1 if fail or len(done) < len(cams) else 0)
