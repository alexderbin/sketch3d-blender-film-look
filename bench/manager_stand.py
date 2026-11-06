"""Pipeline manager test stand WITHOUT Blender (level 1).
pipeline.run() is the real one; the step subprocess is replaced by a stub that writes a ".blend" as text
with a provenance chain (step{keys}<-[input]); "EMPTY" at the bottom of the chain means the step was built
without an input .blend. The project scene is read and never changed; everything is written to a temp folder
(STAND_DIR or mkdtemp). No Blender, no network.
    .venv/bin/python bench/manager_stand.py check   (level 1: C1–C14)  |  [s1 s2 s2b s3 s4 s5 s6 s7 code] — experiments (no arguments — all)
S1 cache (profile, deleted outputs) · S2 "checkpoints: last" + --from · S2b the same without --from
(control) · S3 failure and abort · S4 two runs into one folder · S5 "may_read" · S6 --from on an empty
folder · S7 step edit + from_step later than it · code — static audit of the cache signature: what the step
functions import that the passport's "code:" does not list (no steps are run)."""
import ast, contextlib, io, json, os, re, shutil, subprocess, sys, tempfile, time

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)                                       # entry point outside the roots: the core/components packages are visible without PYTHONPATH
from core.manager import pipeline  # noqa: E402

BASE = os.environ.get("STAND_DIR") or tempfile.mkdtemp(prefix="mgr-stand-")
CFG = {"plan_sleep": 0.0, "startup": 0.0, "work": 0.0, "fail": set(), "gives": {}}   # gives: {step: what the build function returns}
STEPS = ["relief", "building", "camera", "masonry", "textures", "light", "frames"]


class _R:
    def __init__(self, rc): self.returncode = rc


def fake_run(cmd, **kw):
    """Plan phase (plan_run.py) and build phase (blender … build.py -- step.json) — both read step.json, like the
    real ones."""
    argv = [str(c) for c in cmd]; out = kw.get("stdout"); sj = argv[-1]
    if any(a.endswith("plan_run.py") or a == "core.manager.plan_run" for a in argv):
        time.sleep(CFG["plan_sleep"]); S = json.load(open(sj))
        json.dump({"plan_from": S["keys"]}, open(S["plan"], "w"), ensure_ascii=False)
        return _R(0)
    time.sleep(CFG["startup"])                                    # Blender startup, import bpy — seconds
    S = json.load(open(sj)); time.sleep(CFG["work"])
    name = S["step"]; plan = json.load(open(S["plan"])) if S.get("plan") and os.path.exists(S["plan"]) else {}
    inp = S.get("input"); inp_txt = open(inp).read() if inp and os.path.exists(inp) else "EMPTY"
    if name in CFG["fail"]:
        if out: out.write("Traceback (most recent call last):\nRuntimeError: boom\n")
        return _R(1)
    txt = f"{name}{json.dumps(S['keys'], ensure_ascii=False, sort_keys=True)}|plan={json.dumps(plan, ensure_ascii=False, sort_keys=True)}<-[{inp_txt}]"
    if S.get("output"):
        os.makedirs(os.path.dirname(S["output"]), exist_ok=True); open(S["output"], "w").write(txt)
    else:
        open(os.path.join(S["out"], "frames.txt"), "w").write(txt)
    json.dump({"step": name, "new_objects": 1, "collections": {}, "gives": CFG["gives"].get(name, {})}, open(os.path.join(S["out"], "build.json"), "w"))
    return _R(0)


class _Sub:
    run = staticmethod(fake_run); STDOUT = subprocess.STDOUT; PIPE = subprocess.PIPE; CalledProcessError = subprocess.CalledProcessError


pipeline.subprocess = _Sub


def make_conv(path, keep="all", light="night", az=200, swap_cam_building=False):
    C = yaml.safe_load(open(os.path.join(ROOT, "core/manager/pipelines/castle_rock.yaml")))
    for e in C["steps"]:
        if isinstance(e, dict) and "light" in e: e["light"] = {"time": light}
        if isinstance(e, dict) and "camera" in e: e["camera"]["azimuth"] = az
    if keep != "all": C["checkpoints"] = keep
    if swap_cam_building:
        names = [list(e)[0] if isinstance(e, dict) else e for e in C["steps"]]; i, j = names.index("building"), names.index("camera")
        C["steps"][i], C["steps"][j] = C["steps"][j], C["steps"][i]
    yaml.safe_dump(C, open(path, "w"), allow_unicode=True, sort_keys=False); return path


def go(conv, folder, **kw):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = pipeline.run(conv, "bench/scenes/castle_rock", folder, kw.pop("profile", "preview"), **kw)
    except BaseException as e:                                    # noqa: BLE001
        return None, f"EXCEPTION {type(e).__name__}", buf.getvalue()
    return rc, None, buf.getvalue()


def rows(folder, n):
    J = os.path.join(folder, "journal.jsonl")
    return [json.loads(l) for l in open(J).read().splitlines() if l.strip()][-n:] if os.path.exists(J) else []


def show(title, rc, exc, folder, n=7):
    print(f"--- {title}: rc={rc} {exc or ''}")
    for r in rows(folder, n):
        print(f"    {r['step']:<9} {r['status']:<10} {'no .blend ' if r.get('no_blend') else ''}{('cache miss: ' + str(list(r['cache_miss']))) if r.get('cache_miss') else ''}")


def blend(folder, i, name):
    p = os.path.join(folder, f"{i:02d}_{name}", "scene.blend"); return open(p).read() if os.path.exists(p) else None


def chain(txt): return " <- ".join(re.findall(r"(light|textures|masonry|camera|building|relief)[{]", txt or "")) + (" <- EMPTY" if txt and "EMPTY" in txt else "")


def fresh(name):
    d = os.path.join(BASE, name); shutil.rmtree(d, ignore_errors=True); os.makedirs(d); return d


def child(label, folder, light, delay, work, startup):
    time.sleep(delay); CFG.update(plan_sleep=0.3, startup=startup, work=work)
    rc, exc, _ = go(make_conv(folder + "_" + label + ".yaml", light=light), folder); print(f"[{label}] rc={rc} {exc or ''}")


def dry_cached(conv, folder):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): pipeline.run(conv, "bench/scenes/castle_rock", folder, "preview", dry=True)
    line = [l for l in buf.getvalue().splitlines() if l.startswith(("from cache", "cached"))]; return line[0] if line else "everything will be run"


def audit_code_sig():
    """Depth 1: imports in the bodies of the step's "plan"/"build" functions (and at module top level) → is the
    file listed in the passport's "code:". Data: os.path.join(…, "library", …) in files from "code:" → is it
    covered by a "code:" path. Dynamic paths are not visible."""
    parts = ("model", "contract", "manager", "services", "outputs", "blender", "surface", "assets", "recipes")
    kdirs = [os.path.join(ROOT, "core", p) for p in parts] + [os.path.join(ROOT, "core", "assets", "models"), os.path.join(ROOT, "core", "assets", "mocap")]

    def names_in(node):
        out = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Import): out |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0: out.add(n.module.split(".")[0])
        return out

    def fn_imports(path, fn):
        tree = ast.parse(open(path, encoding="utf-8").read()); out = set()
        for n in tree.body:
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.Try, ast.If)): out |= names_in(n)
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.name == fn: out |= names_in(n)
        return out

    R = pipeline.registry(); found = 0
    for name, st in R.items():
        comp = st["component_dir"]
        dirs = [comp] + [os.path.join(comp, m) for m in sorted(os.listdir(comp)) if os.path.isdir(os.path.join(comp, m)) and m not in ("bench", "runs", "ext") and not m.startswith((".", "_"))] + kdirs
        listed = set(pipeline._code_sig(st)); d1 = set()
        for field in ("build", "plan"):
            f = pipeline._fn(st, field)
            if not f: continue
            for n in fn_imports(f[0], f[1]):
                for d in [os.path.dirname(f[0])] + dirs:
                    c = os.path.join(d, n + ".py")
                    if os.path.isfile(c): d1.add(os.path.relpath(c, ROOT)); break
        d1.discard("core/kernel.py"); miss = sorted(d1 - listed)
        data = set()
        for rel in listed:                                      # os.path.join(…, "library", "a", "b") in step code → library/a/b (comments and messages excluded)
            if not rel.endswith(".py"): continue
            try: tree = ast.parse(open(os.path.join(ROOT, rel), encoding="utf-8").read())
            except Exception: continue
            for n in ast.walk(tree):
                if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "join":
                    lit = [a.value if isinstance(a, ast.Constant) and isinstance(a.value, str) else None for a in n.args]
                    if "library" in lit:
                        tail = []
                        for v in lit[lit.index("library") + 1:]:
                            if v is None: break
                            tail.append(v)
                        if tail: data.add("/".join(["library"] + tail))
        data = sorted(x for x in data if not any(x.startswith(g.rstrip("*")) or g.startswith(x + "/") for g in (st.get("code") or [])))   # covered if "code:" names the path or a file inside it
        if miss or data:
            found += 1; print(f"  {name:<10} modules outside 'code:' (direct imports): {miss or '—'}\n             data library/… (os.path.join in the step code), not in 'code:': {data[:5] or '—'}")
    print(f"--- code: steps with signature gaps: {found} of {len(R)}")


def main(which):
    if "s1" in which:
        d = fresh("s1"); c = make_conv(os.path.join(d, "c.yaml")); f = os.path.join(d, "run")
        show("S1 first run", *go(c, f)[:2], f); show("S1 repeat", *go(c, f)[:2], f)
        show("S1 profile final (relief does not depend on the profile, but is rebuilt)", *go(c, f, profile="final")[:2], f)
        os.remove(os.path.join(f, "07_frames", "frames.txt"))      # only the output was deleted, the .step.json marker remains
        show("S1 output of step 'frames' deleted, marker remains", *go(c, f, profile="final")[:2], f)
        print("    frames.txt exists:", os.path.exists(os.path.join(f, "07_frames", "frames.txt")))
    if "s2" in which or "s2b" in which:
        d = fresh("s2"); c = make_conv(os.path.join(d, "c.yaml"), keep="last"); f = os.path.join(d, "run"); go(c, f)
        print("--- S2 'checkpoints: last': .blend files remain for:", [n for i, n in enumerate(STEPS[:6], 1) if blend(f, i, n) is not None])
        if "s2" in which:
            rc, exc, out = go(c, f, since="masonry", fresh=True); show("S2 --fresh --from masonry", rc, exc, f)
            print("    the manager said:", [l for l in out.splitlines() if "restart" in l][:1])
            print("    chain of step 'light' after this:", chain(blend(f, 6, "light")))
        if "s2b" in which:
            c2 = make_conv(os.path.join(d, "c2.yaml"), keep="last", az=210)
            show("S2b (control) camera edit WITHOUT --from: restart from 'relief' works", *go(c2, f)[:2], f)
    if "s3" in which:
        d = fresh("s3"); c = make_conv(os.path.join(d, "c.yaml")); f = os.path.join(d, "run"); CFG["fail"] = {"masonry"}
        show("S3 failure of 'masonry': the journal has rows only up to the failure, no steps after it", *go(c, f)[:2], f); CFG["fail"] = set()
        go(c, f); n0 = len(rows(f, 999))
        c2 = make_conv(os.path.join(d, "c2.yaml"), light="sunset")
        orig = fake_run
        def killing(cmd, **kw):                                    # kill the process after the .blend of step "light" is written, before the cache marker
            r = orig(cmd, **kw)
            if str(cmd[-1]).endswith("step.json") and "06_light" in str(cmd[-1]) and not any(str(a).endswith("plan_run.py") for a in cmd): raise KeyboardInterrupt
            return r
        _Sub.run = staticmethod(killing)
        rc, exc, _ = go(c2, f); _Sub.run = staticmethod(orig)
        print(f"--- S3 'sunset': killed after saving the .blend of step 'light', before the marker is written: {exc}; journal rows added: {len(rows(f, 999)) - n0}")
        mk = os.path.join(f, "06_light", ".step.json")
        print("    06_light/scene.blend:", (blend(f, 6, "light") or "")[:40], "| marker .step.json:", json.load(open(mk))["key"] if os.path.exists(mk) else "none (removed before the step)")
        show("S3 return to 'night': step 'light' must be executed again, not from cache", *go(c, f)[:2], f)
    if "s4" in which:
        d = fresh("s4"); f = os.path.join(d, "run")
        pa = subprocess.Popen([sys.executable, "-I", __file__, "child", "A", f, "night", "0.0", "3.0", "3.0"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        pb = subprocess.Popen([sys.executable, "-I", __file__, "child", "B", f, "sunset", "1.5", "0.2", "3.0"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        oa, ob = pa.communicate()[0], pb.communicate()[0]
        R = rows(f, 999); dup = {s: sum(1 for r in R if r["step"] == s and r["status"] == "done") for s in STEPS}
        print("--- S4 two processes into one folder (A 'night', B 'sunset'): rc A/B:", oa.strip(), ob.strip())
        print("    how many times a step is 'done' in total (2 = the same work done twice):", dup)
        print("    rows in journal.jsonl:", len(R), "(two interleaved halves, the rows carry no common run identification)")
        CFG.update(plan_sleep=0, startup=0, work=0)
        for lab, lt in (("A", "night"), ("B", "sunset")): print(f"    dry run, variant {lab} ({lt}):", dry_cached(make_conv(os.path.join(d, f"chk_{lab}.yaml"), light=lt), f))
    if "s5" in which:
        d = fresh("s5"); c = make_conv(os.path.join(d, "c.yaml"), swap_cam_building=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): rc = pipeline.run(c, "bench/scenes/castle_rock", os.path.join(d, "run"), "preview", dry=True)
        print("--- S5 'camera' BEFORE 'building' (camera: reads [], may_read [buildings]): dry rc =", rc, "— the contract stayed silent")
    if "s6" in which:
        d = fresh("s6"); c = make_conv(os.path.join(d, "c.yaml")); f = os.path.join(d, "run")
        show("S6 empty folder, --from masonry", *go(c, f, since="masonry")[:2], f)
        print("    chain of step 'masonry':", chain(blend(f, 4, "masonry")))
        show("S6 next ordinary full run", *go(c, f)[:2], f)
        print("    chain of step 'light':", chain(blend(f, 6, "light")))
    if "code" in which:
        audit_code_sig()
    if "s7" in which:
        d = fresh("s7"); c = make_conv(os.path.join(d, "c.yaml")); f = os.path.join(d, "run"); go(c, f)
        c2 = make_conv(os.path.join(d, "c2.yaml"), az=210)
        show("S7 edit of 'camera' (azimuth 210), from_step = 'masonry' (later than the edited step)", *go(c2, f, since="masonry")[:2], f)
        print("    chain of step 'light':", chain(blend(f, 6, "light")), "| azimuth 210 in the chain:", "210" in (blend(f, 6, "light") or ""))
        show("S7 next ordinary run with the same keys", *go(c2, f)[:2], f)
        print("    chain of step 'light':", chain(blend(f, 6, "light")), "| azimuth 210 in the chain:", "210" in (blend(f, 6, "light") or ""))


if __name__ == "__main__" and not (len(sys.argv) > 1 and sys.argv[1] == "check"):
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])); sys.exit(0)
    main(sys.argv[1:] or ["s1", "s2", "s2b", "s3", "s4", "s5", "s6", "s7", "code"])
    print(f"\n(stand temp folder: {BASE})")


# ── checks with expectations (level 1): .venv/bin/python -I manager_stand.py check → exit code 0/1 ───────────────────────────
def check():
    results = []
    def expect(name, cond, detail=""):
        results.append((name, bool(cond), detail)); print(f"  {'ok ' if cond else 'FAIL'} {name} {detail if not cond else ''}")
    st = lambda f, n: {r["step"]: r["status"] for r in rows(f, n)}
    print("--- check")
    d = fresh("c1"); c = make_conv(os.path.join(d, "c.yaml")); f = os.path.join(d, "run")
    rc, exc, _ = go(c, f); s = st(f, 7)
    expect("C1 first run: 7 done", rc == 0 and list(s.values()) == ["done"] * 7, f"{rc} {exc} {s}")
    rc, exc, _ = go(c, f); s = st(f, 7)
    expect("C1 repeat: 7 from cache", rc == 0 and set(s.values()) == {"cached"}, f"{rc} {exc} {s}")
    os.remove(os.path.join(f, "07_frames", "frames.txt")); rc, exc, _ = go(c, f); s = st(f, 7)
    expect("C2 output of 'frames' deleted → frames done, the rest from cache", s.get("frames") == "done" and s.get("light") == "cached"
           and os.path.exists(os.path.join(f, "07_frames", "frames.txt")), f"{s}")
    # C3: abort after the .blend of step "light" is saved, before the marker → on return "light" is rebuilt
    c2 = make_conv(os.path.join(d, "c2.yaml"), light="sunset"); orig = fake_run
    def killing(cmd, **kw):
        r = orig(cmd, **kw)
        if str(cmd[-1]).endswith("step.json") and "06_light" in str(cmd[-1]) and not any(str(a).endswith("plan_run.py") for a in cmd): raise KeyboardInterrupt
        return r
    _Sub.run = staticmethod(killing); go(c2, f); _Sub.run = staticmethod(orig)
    rc, exc, _ = go(c, f); s = st(f, 7)
    expect("C3 after the abort 'light' is done again, not from cache", s.get("light") == "done" and s.get("masonry") == "cached", f"{s}")
    # C4: busy folder — the second run is refused with code 101
    open(os.path.join(f, ".lock"), "w").write(str(os.getppid())); rc, exc, _ = go(c, f); os.remove(os.path.join(f, ".lock"))   # another process's live pid
    expect("C4 folder busy with a live process → rc 101", rc == 101, f"{rc} {exc}")
    rc, exc, _ = go(c, f); expect("C4 after .lock is removed the run proceeds", rc == 0, f"{rc} {exc}")
    # C5: empty folder, --from masonry → restart from relief, the masonry chain is complete
    d = fresh("c5"); c = make_conv(os.path.join(d, "c.yaml")); f = os.path.join(d, "run"); rc, exc, out = go(c, f, since="masonry")
    ch = chain(blend(f, 4, "masonry"))
    expect("C5 --from masonry on an empty folder → restart from relief, masonry <- camera <- building <- relief", rc == 0 and ch.startswith("masonry <- camera <- building <- relief"), f"{rc} {exc} {ch}")
    # C6: edit of "camera" + --from masonry, then an ordinary run → masonry rebuilt from the new camera
    d = fresh("c6"); c = make_conv(os.path.join(d, "c.yaml")); f = os.path.join(d, "run"); go(c, f)
    c2 = make_conv(os.path.join(d, "c2.yaml"), az=210); go(c2, f, since="masonry"); rc, exc, _ = go(c2, f); s = st(f, 7)
    expect("C6 after the camera rebuild masonry is not from cache (the input changed)", s.get("camera") == "done" and s.get("masonry") == "done"
           and "210" in (blend(f, 6, "light") or ""), f"{s}")
    # C7: code signature — closure over imports: editing a helper changes the signature
    td = os.path.join(ROOT, "components", "_stand_code_sig"); shutil.rmtree(td, ignore_errors=True); os.makedirs(td)   # package: the imports are package imports
    open(os.path.join(td, "__init__.py"), "w").write("")
    open(os.path.join(td, "a.py"), "w").write("from components._stand_code_sig import b\n"); open(os.path.join(td, "b.py"), "w").write("X = 1\n")
    fake = {"code": [os.path.relpath(os.path.join(td, "a.py"), ROOT)], "component_dir": td}
    s1 = pipeline._code_sig(fake); open(os.path.join(td, "b.py"), "w").write("X = 2\n"); pipeline._IMPORTS.clear(); s2 = pipeline._code_sig(fake)
    expect("C7 helper b.py (package import) is in the step's code signature and editing it changes the signature", "components/_stand_code_sig/b.py" in s1 and s1 != s2, f"{sorted(s1)}")
    shutil.rmtree(td, ignore_errors=True)
    # C9: a step with only the "plan" phase (Python, no Blender) between builds — the checkpoint .blend passes
    # through it
    if os.path.isdir(os.path.join(ROOT, "components", "battle")):   # the stand step is the battle plan (the battle component is not part of the public release)
        d = fresh("c9"); R0 = pipeline.registry
        def reg():
            R = R0(); R["events"] = dict({"name": "events", "component": "battle", "version": 1, "kind": "refines", "section": "actors",
                                                                    "reads": ["cameras"], "writes": [], "plan": "steps.py:plan_events", "cost": "light", "on_fail": "stop",
                                                                    "component_dir": os.path.join(ROOT, "components", "battle"), "source": "stand", "is_output": False})
            return R
        pipeline.registry = reg
        C = yaml.safe_load(open(os.path.join(ROOT, "core/manager/pipelines/castle_rock.yaml"))); names = [list(e)[0] if isinstance(e, dict) else e for e in C["steps"]]
        C["steps"].insert(names.index("masonry"), "events")
        c = os.path.join(d, "c.yaml"); yaml.safe_dump(C, open(c, "w"), allow_unicode=True, sort_keys=False); f = os.path.join(d, "run")
        rc, exc, out9 = go(c, f); s = st(f, 9); pipeline.registry = R0
        if rc == 100: print("    contract:", [l for l in out9.splitlines() if l.strip()][:4])
        expect("C9 plan-only step: done without .blend, masonry is built from the camera", rc == 0 and s.get("events") == "done" and not os.path.exists(os.path.join(f, "04_events", "scene.blend"))
               and chain(blend(f, 5, "masonry")).startswith("masonry <- camera <- building <- relief"), f"{rc} {exc} {s} chain={chain(blend(f, 5, 'masonry'))}")
    else:
        print("  skip C9 (no components/battle)")
    # C11: the step function's return value — what the passport declares in gives goes into the run context,
    # the rest is statistics in the journal row (stats)
    d = fresh("c11"); c = make_conv(os.path.join(d, "c.yaml")); f = os.path.join(d, "run")
    CFG["gives"] = {"building": {"building_plan": "/plan", "objects": 5}, "camera": {"cameras": 2}, "light": {"light_plan": "/lp", "lamps": 3}}
    rc, exc, _ = go(c, f); CFG["gives"] = {}
    r = {x["step"]: x for x in rows(f, 9)}; ctx = json.load(open(os.path.join(f, "context.json"))) if os.path.exists(os.path.join(f, "context.json")) else {}
    g = ctx.get("gives") or {}
    expect("C11 gives: building_plan and light_plan in the context, objects/lamps — journal stats",
           rc == 0 and g.get("building_plan") == "/plan" and g.get("light_plan") == "/lp" and "objects" not in g and "lamps" not in g
           and (r.get("building") or {}).get("stats") == {"objects": 5},
           f"{rc} {exc} gives={g} stats={(r.get('building') or {}).get('stats')}")
    # C12: a profile is a set of passport knobs (profiles: {final: {detail: all}} of masonry) under the pipeline
    # knobs; the preview has none
    d = fresh("c12"); c = make_conv(os.path.join(d, "c.yaml"))
    C = yaml.safe_load(open(c))
    for e in C["steps"]:
        if isinstance(e, dict) and "masonry" in e: e["masonry"] = {k: v for k, v in (e["masonry"] or {}).items() if k != "detail"}
    yaml.safe_dump(C, open(c, "w"), allow_unicode=True, sort_keys=False)
    rc1, exc1, _ = go(c, os.path.join(d, "fin"), profile="final"); k_fin = ({x["step"]: x for x in rows(os.path.join(d, "fin"), 9)}.get("masonry") or {}).get("keys")
    rc2, exc2, _ = go(c, os.path.join(d, "sk")); k_sk = ({x["step"]: x for x in rows(os.path.join(d, "sk"), 9)}.get("masonry") or {}).get("keys")
    expect("C12 profile final gives masonry detail: all, the preview does not", rc1 == 0 and rc2 == 0 and (k_fin or {}).get("detail") == "all" and "detail" not in (k_sk or {}),
           f"{rc1} {exc1} {rc2} {exc2} final={k_fin} preview={k_sk}")
    # C13: a command template without a value. a) {missing} — neither a knob nor an earlier gives: contract error
    #      before launch (code 100), nothing is built;
    #      b) {light_plan} — gives of step "light" is declared but the stub did not return it: the command is not
    #      launched, the step is failed with a reason
    R0 = pipeline.registry; orig13 = _Sub.run
    def probe(cmd):
        def reg():
            R = R0(); R["probe"] = {"name": "probe", "component": "core", "version": 1, "kind": "output", "section": "all", "reads": [], "writes": [],
                                    "command": cmd, "cost": "light", "on_fail": "stop", "is_output": True,
                                    "component_dir": os.path.join(ROOT, "core", "manager"), "source": "stand"}
            return R
        return reg
    calls = []
    def spy(cmd, **kw):
        if "pass" in [str(c) for c in cmd]: calls.append(cmd)
        return orig13(cmd, **kw)
    C = yaml.safe_load(open(os.path.join(ROOT, "core/manager/pipelines/castle_rock.yaml"))); C["outputs"] = (C.get("outputs") or []) + ["probe"]
    d = fresh("c13"); c = os.path.join(d, "c.yaml"); yaml.safe_dump(C, open(c, "w"), allow_unicode=True, sort_keys=False)
    pipeline.registry = probe(["{py}", "-c", "pass", "{missing}"]); _Sub.run = staticmethod(spy)
    rc, exc, out13 = go(c, os.path.join(d, "a"))
    expect("C13a template {missing} without a source — contract error before launch (code 100), nothing launched", rc == 100 and not calls and "missing" in out13,
           f"{rc} {exc} calls={len(calls)}")
    pipeline.registry = probe(["{py}", "-c", "pass", "{light_plan}"])
    rc, exc, _ = go(c, os.path.join(d, "b")); pipeline.registry = R0; _Sub.run = staticmethod(orig13)
    r = {x["step"]: x for x in rows(os.path.join(d, "b"), 9)}.get("probe") or {}
    expect("C13b gives declared but not returned — command not launched, step failed with a reason", r.get("status") == "failed" and not calls
           and any("no value" in t for t in r.get("tail") or []), f"{rc} {exc} status={r.get('status')} calls={len(calls)} tail={r.get('tail')}")
    # C14: sketch stage: masonry and textures — passport profiles {sketch: off}: they do not run, the .blend
    #      passes through; light — {sketch: plan} (passport override): only the "plan" phase, no .blend; frames —
    #      look: sketch, maps: false (outputs.yaml); a repeat — from cache; textures without masonry on the sketch
    #      stage — contract error before launch (code 100) naming the stage
    d = fresh("c14"); R0 = pipeline.registry
    def reg14(textures_on=False):
        def reg():
            R = R0(); R["light"] = dict(R["light"], profiles={"sketch": "plan"})
            if textures_on: R["textures"] = dict(R["textures"], profiles={})
            return R
        return reg
    pipeline.registry = reg14(); c = make_conv(os.path.join(d, "c.yaml")); f = os.path.join(d, "run")
    rc, exc, _ = go(c, f, profile="sketch"); r = {x["step"]: x for x in rows(f, 7)}; s = {k: v["status"] for k, v in r.items()}
    fp = os.path.join(f, "07_frames", "frames.txt"); fr = open(fp).read() if os.path.exists(fp) else ""; fk = (r.get("frames") or {}).get("keys") or {}
    expect("C14 sketch: masonry and textures off, light — plan only (no .blend), frames — from the camera, look: sketch, maps: false",
           rc == 0 and s.get("masonry") == "off" and s.get("textures") == "off" and s.get("light") == "done"
           and (r.get("light") or {}).get("profile_mode") == "plan" and not os.path.exists(os.path.join(f, "06_light", "scene.blend"))
           and chain(fr).startswith("camera <- building <- relief") and fk.get("look") == "sketch" and fk.get("maps") is False,
           f"{rc} {exc} {s} chain={chain(fr)} frames={fk}")
    rc, exc, _ = go(c, f, profile="sketch"); s = st(f, 7)
    expect("C14 sketch, repeat: from cache, the disabled ones — off", rc == 0 and set(s.values()) <= {"cached", "off"} and s.get("masonry") == "off", f"{rc} {exc} {s}")
    pipeline.registry = reg14(textures_on=True)
    rc, exc, out14 = go(c, os.path.join(d, "b"), profile="sketch"); pipeline.registry = R0
    expect("C14 textures without masonry on the sketch — contract error before launch (code 100), the cause is the stage", rc == 100 and "disabled at stage sketch" in out14,
           f"{rc} {exc} {[l for l in out14.splitlines() if 'textures' in l][:2]}")
    # C15: a stage in the recipe overrides the passport: stages: {sketch: {detail: cam}} on masonry — runs on the
    #      sketch; stages: {preview: off} on textures — they do not run on the preview; stages: {preview:
    #      {cameras: "1"}} on frames — the knob applies on the preview only
    d = fresh("c15"); C = yaml.safe_load(open(make_conv(os.path.join(d, "c.yaml"))))
    for e in C["steps"]:
        if isinstance(e, dict) and "masonry" in e: e["masonry"] = dict(e["masonry"] or {}, stages={"sketch": {"detail": "cam"}})
    C["steps"] = [{"textures": {"stages": {"preview": "off"}}} if e == "textures" or (isinstance(e, dict) and "textures" in e) else e for e in C["steps"]]
    C["outputs"] = [{"frames": dict((e.get("frames") or {}) if isinstance(e, dict) else {}, stages={"preview": {"cameras": "1"}})}
                    if (e == "frames" or (isinstance(e, dict) and "frames" in e)) else e for e in C["outputs"]]
    c = os.path.join(d, "c.yaml"); yaml.safe_dump(C, open(c, "w"), allow_unicode=True, sort_keys=False)
    rc1, exc1, _ = go(c, os.path.join(d, "sk"), profile="sketch"); s1 = {x["step"]: x for x in rows(os.path.join(d, "sk"), 7)}
    rc2, exc2, _ = go(c, os.path.join(d, "pv")); s2 = {x["step"]: x for x in rows(os.path.join(d, "pv"), 7)}
    expect("C15 recipe: masonry runs on the sketch (detail: cam), textures off on the preview, frames get cameras 1 on the preview, the sketch does not",
           rc1 == 0 and rc2 == 0 and (s1.get("masonry") or {}).get("status") == "done" and ((s1.get("masonry") or {}).get("keys") or {}).get("detail") == "cam"
           and (s1.get("textures") or {}).get("status") == "off" and (s2.get("textures") or {}).get("status") == "off"
           and ((s2.get("frames") or {}).get("keys") or {}).get("cameras") == "1" and "cameras" not in ((s1.get("frames") or {}).get("keys") or {}),
           f"{rc1} {exc1} {rc2} {exc2} sk={ {k: v['status'] for k, v in s1.items()} } pv={ {k: v['status'] for k, v in s2.items()} }")
    bad = [n for n, ok, _ in results if not ok]
    print(f"--- manager stand: {len(results) - len(bad)}/{len(results)} ok" + (f"; FAIL: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "check":
    sys.exit(check())
