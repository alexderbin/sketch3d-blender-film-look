"""Warm Blender: the host from core/ext/warm_host (Engine and Server unmodified) with a pipeline scene in memory.
The caller and scripts send operations through core/blender/bl_call.py (inspect, execute with the steps props, rna, python…,
capture) — without launching Blender and loading the scene for every look. Own operations — the python step: `import bl_warm; bl_warm.look(…)`.
blender -b --factory-startup -P core/blender/run.py -- core/blender/bl_serve.py -- <bench/scenes/X> <state_folder> [simple=600]
Added on top of the host: the pipeline scene (bl_scene.load_scene) at startup; SystemExit and KeyboardInterrupt in a step are a
step error, the host stays alive (in the unmodified host, exit() in a python step kills the process with code 0); stopping — after idle time
(the machine is shared, memory) or by the file <folder>/stop. Writes <folder>/connection.json (port, token) and host.json (pid, scene)."""
import core.paths        # noqa: E402  core parts on sys.path
import json, os, sys, time
from pathlib import Path
core.paths.ext("warm_host", "addon")
sys.dont_write_bytecode = True
import bpy
from compact_blender.engine import Engine
from compact_blender.transport import Server
from core.blender import bl_scene as B

a = sys.argv[sys.argv.index("--") + 1:]; SD = os.path.abspath(a[0]); state = Path(a[1]).resolve(); state.mkdir(parents=True, exist_ok=True)
opt = dict(kv.split("=", 1) for kv in a[2:]); IDLE = float(opt.get("simple", 600))
t0 = time.monotonic(); B.load_scene(SD); os.environ["SKETCH_SCENE"] = SD; os.environ["SKETCH_SERVE"] = str(state)   # bl_warm knows the scene and the folder
eng = Engine(state / "exports", ("write", "render", "python")); _step = eng.step; last = [time.monotonic()]


def step(s):
    last[0] = time.monotonic()
    try:
        return _step(s)
    except (SystemExit, KeyboardInterrupt) as exc:
        raise RuntimeError(f"{type(exc).__name__} in step {s.get('op')} caught — the host is alive") from None


eng.step = step
srv = Server(eng, state / "connection.json")
(state / "host.json").write_text(json.dumps({"pid": os.getpid(), "scene": SD, "load_s": round(time.monotonic() - t0, 2), "idle_s": IDLE}))
print(f"__READY__ {os.getpid()} scene in {time.monotonic() - t0:.2f} s", flush=True)
try:
    while not (state / "stop").exists() and time.monotonic() - last[0] < IDLE:
        srv.poll(); time.sleep(0.005)
finally:
    srv.close()
    for f in ("connection.json", "host.json", "stop"): (state / f).unlink(missing_ok=True)
