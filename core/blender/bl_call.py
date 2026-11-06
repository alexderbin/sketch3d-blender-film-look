"""Client of the warm Blender (core/blender/bl_serve.py; the client comes from the warm host package, core/ext/warm_host): on the first call it starts the scene
host, then sends operations — without launching Blender and loading the scene for every look.
.venv/bin/python -m core.blender.bl_call scenes/<name> look <K> [size]        — a view through camera K → PNG (the path is in the reply)
.venv/bin/python -m core.blender.bl_call scenes/<name> inspect '<json params>' — the host's inspect: objects by name, pages
.venv/bin/python -m core.blender.bl_call scenes/<name> execute '<json steps>'      — the host's steps (props, rna, transform, python…)
.venv/bin/python -m core.blender.bl_call scenes/<name> py '<code>'                 — the python step (bpy and bl_warm are available)
.venv/bin/python -m core.blender.bl_call scenes/<name> status | stop
State — out/serve/<name>-<path digest>/ (connection.json, host.json, host.log); the host exits by itself after 10 min idle. The reply is the line
__RESULT__ (the host's reply as is, compact JSON) and the call time."""
import core.paths        # noqa: E402  core parts on sys.path
import json, os, socket, subprocess, sys, time
ROOT = core.paths.ROOT; core.paths.ext("warm_host", "src")
from compact_mcp.client import Client

SD = os.path.abspath(sys.argv[1]); CMD = sys.argv[2]; ARGS = sys.argv[3:]
import hashlib                                                       # the host folder: the scene name + a path digest (copies of the pages are all named "scene")
ST = os.path.join(ROOT, "out", "serve", f"{os.path.basename(SD.rstrip('/'))}-{hashlib.sha1(SD.encode()).hexdigest()[:6]}"); CONN = os.path.join(ST, "connection.json")
out = lambda r, **k: print("__RESULT__" + json.dumps({**r, **k}, ensure_ascii=False, separators=(",", ":")))


def alive():
    try:
        info = json.load(open(CONN)); socket.create_connection(("127.0.0.1", info["port"]), timeout=0.5).close(); return True
    except Exception:
        return False


if CMD == "stop":
    if alive(): open(os.path.join(ST, "stop"), "w").close()
    t = time.time()
    while alive() and time.time() - t < 3: time.sleep(0.1)
    if not alive():                                                 # the host folder holds views and the log, not needed after a stop
        import shutil; shutil.rmtree(ST, ignore_errors=True)
    out({"stopped": not alive()}); sys.exit(0)
t0 = time.time(); started = False
if not alive():                                                     # no host — start one (in its own process session: it lives on after the client exits)
    os.makedirs(ST, exist_ok=True)
    for f in ("connection.json", "stop"):
        if os.path.exists(os.path.join(ST, f)): os.remove(os.path.join(ST, f))
    subprocess.Popen(["blender", "-b", "--factory-startup", "-P", core.paths.RUN, "--", core.paths.script("bl_serve.py"), "--", SD, ST],
                     stdout=open(os.path.join(ST, "host.log"), "w"), stderr=subprocess.STDOUT, start_new_session=True)
    while not alive() and time.time() - t0 < 30: time.sleep(0.1)
    if not alive(): out({"error": "the host did not start", "log": open(os.path.join(ST, "host.log")).read()[-600:]}); sys.exit(1)
    started = True
t1 = time.time(); cl = Client(CONN)
if CMD == "status":
    r = {"host": json.load(open(os.path.join(ST, "host.json")))}
elif CMD == "look":
    code = f"import bl_warm\nprint(bl_warm.look({int(ARGS[0])}, {os.path.join(ST, f'look_{ARGS[0]}.png')!r}, {int(ARGS[1]) if len(ARGS) > 1 else 640}))"
    r = cl.call("execute", {"steps": [{"op": "python", "code": code}]})
elif CMD == "py":
    r = cl.call("execute", {"steps": [{"op": "python", "code": ARGS[0]}]})
else:
    r = cl.call(CMD, json.loads(ARGS[0]) if ARGS else {})
out(r if isinstance(r, dict) else {"result": r}, started=started, start_s=round(t1 - t0, 2), call_s=round(time.time() - t1, 3))
