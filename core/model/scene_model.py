"""The scene model (ARCHITECTURE.md; "the model is data"): a single scene description by sections on top of the files of the scenes/<name>/ folder.
scene.yaml in the scene folder is the explicit model (sections as data or links to files); without it the model is derived from the files
by the SECTIONS table of contents (all the project's scenes are readable without edits). Read a scene through load(), do not open the files directly.
    .venv/bin/python -m core.model.scene_model summary|check|files|write <scenes/name> […] [--json] [--copy] [--force]
  summary — a summary by sections; check — validation (the exit code = the number of errors); files — the scene description files (--copy — those that
  go into the run scene's copy); write — write scene.yaml as a table of contents (if it exists — only with --force).
The keys of the model and of scene.yaml are English.
In Blender the module is read via core/blender/run.py or build.py (they attach yaml from the venv)."""
import json, os, sys


VERSION = 1
# section (vocabulary canon) → (component folder, scene files as of today). The order is the default one (ARCHITECTURE.md); the real one comes from the passports.
SECTIONS = {
    "concept":       (None, ()),
    "site":          ("site", ("location.json", "relief.npz", "relief_w.npz", "surface.json", "cover.npz")),   # cover.npz refines grass
    "buildings":     ("buildings", ("buildings.yaml", "occupancy.json", "anchors.json", "surfaces.json", "influence.npz")),   # layers for neighbors (core.layers)
    "constructions": ("constructions", ("constructions.yaml",)),
    "layout":        ("layout", ("plan.yaml", "lib.json", "instances.json", "canopy.json")),   # canopy is a layer (core.layers)
    "actors":        ("actors", ("figkit.json", "battle.yaml")),
    "cameras":       ("camera", ("cameras.json", "cams.yaml")),
    "light":         ("light", ("far.yaml",)),
    "materials":     ("materials", ("materials.yaml",)),
    "time":          ("time", ()),
}
BUILD = ("source.json", "scene.obj", "world.json", "edits.json")    # a built scene: the .blend by link and the gray copy (bridge)
GREY = ("scene.obj", "world.json", "figkit.json", "instances.json")
NOCOPY = {"plan.yaml", "cameras.json", "cams.yaml"}                 # not copied into the run scene: the plan is a source, the caller places the cameras
SIDE = {"sheet.png"}                                                # sheets and backup copies are not part of the scene description
JSONS = {"location.json", "lib.json", "instances.json", "figkit.json", "cameras.json", "source.json", "world.json", "edits.json"}
YAMLS = {"buildings.yaml", "constructions.yaml", "plan.yaml", "battle.yaml", "cams.yaml", "far.yaml", "materials.yaml"}


def _yaml():
    try:
        import yaml
        return yaml
    except ImportError:
        return None


def _read(path):
    if path.endswith(".json"):
        return json.load(open(path))
    Y = _yaml()
    return Y.safe_load(open(path)) if Y else None


def load(scene_dir):
    """Scene model — a dict: scene, dir, explicit (scene.yaml exists), model, concept, fps, seed,
    sections {section: {component, files, data, source}}, build {blend, grey, edits, stepwise}, origin, notes."""
    SD = os.path.abspath(scene_dir)
    have = lambda f: os.path.exists(os.path.join(SD, f))
    M = dict(scene=os.path.basename(SD), dir=SD, explicit=False, model=VERSION, concept=None, fps=None,
             seed=0, sections={}, build={}, origin=None, notes=[])
    spec = {}
    if have("scene.yaml"):
        if _yaml():
            spec = _yaml().safe_load(open(os.path.join(SD, "scene.yaml"))) or {}
            M["explicit"] = True
            for k in ("model", "concept", "fps", "seed"):
                if k in spec: M[k] = spec[k]
        else:
            M["notes"].append("scene.yaml was not read (no yaml in this Python) — the model is derived from files")
    over = spec.get("sections") or {}
    for name, (comp, files_) in SECTIONS.items():
        o = over.get(name)
        sec = dict(component=comp, files=[f for f in files_ if have(f)], data=None, source=None)
        if isinstance(o, dict):
            ref = o.get("files") or ([o["file"]] if o.get("file") else None)
            if ref is not None: sec["files"] = list(ref)
            data = {k: v for k, v in o.items() if k not in ("file", "files")}
            sec["data"] = data or None
        elif isinstance(o, str):
            sec["files"] = [o]
        if name == "light":
            sec["source"] = _light_source(SD, sec)
        if sec["files"] or sec["data"] or sec["source"]:
            M["sections"][name] = sec
    for k in (spec.get("sections") or {}):
        if k not in SECTIONS:
            M["notes"].append(f"unknown section '{k}' in scene.yaml (sections: {', '.join(SECTIONS)})")
    bspec = spec.get("build") or {}
    blend = bspec.get("blend")
    if not blend and have("source.json"):
        try: blend = json.load(open(os.path.join(SD, "source.json"))).get("blend")
        except Exception: blend = None
    M["build"] = dict(blend=blend, grey=[f for f in GREY if have(f)], edits="edits.json" if have("edits.json") else None,
                           stepwise=bool(bspec.get("stepwise")))   # the scene is built by the stepwise pipeline (core/blender/build.py)
    M["origin"] = _origin(have)
    return M


def _light_source(SD, sec):
    """Where the scene's light and sky come from — the same order as in far.load_spec: far.yaml → far: in battle.yaml → the site's biome → world.json."""
    if sec["data"]: return "scene.yaml"
    if sec["files"]: return sec["files"][0]
    bp = os.path.join(SD, "battle.yaml")
    if os.path.exists(bp) and _yaml():
        try:
            if (_yaml().safe_load(open(bp)) or {}).get("far"): return "battle.yaml: far"
        except Exception: pass
    lp = os.path.join(SD, "location.json")
    if os.path.exists(lp):
        try: L = json.load(open(lp))
        except Exception: L = {}
        return f"site: biome {L.get('biome') or 'by the ground'}"
    return "world.json: default" if os.path.exists(os.path.join(SD, "world.json")) else None


def _origin(have):
    if have("battle.yaml"): return "battle (battle.py)"
    if have("plan.yaml"): return "plan in words (layout.py)"
    if have("location.json"): return "site (terrain.py scene)"
    if have("lib.json"): return "library (models/scene.py)"
    if have("source.json"): return ".blend via the bridge"
    if have("scene.obj"): return "gray scene sketch25"
    return "empty"


def files(M, copy=False):
    """Scene description files present in the folder (names). copy — for the run scene's copy (page, shot, take, sheet):
    everything except NOCOPY, plus scene.yaml."""
    out = []
    for sec in M["sections"].values():
        out += [f for f in sec["files"] if f not in out]
    for f in BUILD + GREY:
        if f not in out and os.path.exists(os.path.join(M["dir"], f)): out.append(f)
    if M["explicit"]: out.append("scene.yaml")
    return [f for f in out if not (copy and f in NOCOPY)]


def check(M):
    """Model check → [(level, text)]; the level is "error" or "note"."""
    out = [("error", z) if z.startswith("unknown") else ("note", z) for z in M["notes"]]
    SD = M["dir"]
    for name, sec in M["sections"].items():
        for f in sec["files"]:
            p = os.path.join(SD, f)
            if not os.path.exists(p):
                out.append(("error", f"{name}: no file {f}")); continue
            if f in JSONS or f in YAMLS:
                try:
                    d = _read(p)
                except Exception as e:
                    out.append(("error", f"{name}: {f} is unreadable — {type(e).__name__}: {str(e)[:80]}")); continue
                if f == "cameras.json":
                    if not isinstance(d, list) or not d:
                        out.append(("error", "cameras: cameras.json is not a list of cameras"))
                    else:
                        bad = [c.get("name", i) for i, c in enumerate(d) if not {"matrix_world", "resolution"} <= set(c)]
                        if bad: out.append(("error", f"cameras: without matrix_world or resolution — {bad[:4]}"))
                elif f in YAMLS and d is not None and not isinstance(d, dict):
                    out.append(("error", f"{name}: {f} is not a dict"))
    for f in ("world.json", "source.json", "edits.json"):
        p = os.path.join(SD, f)
        if os.path.exists(p):
            try: json.load(open(p))
            except Exception as e: out.append(("error", f"build: {f} is unreadable — {type(e).__name__}"))
    B = M["build"]
    if B["blend"] and not os.path.exists(B["blend"]):
        out.append(("note", f"build: the .blend is not in place ({os.path.relpath(B['blend'], os.path.dirname(os.path.dirname(SD)))}) — rebuild it with the builder"))
    if not B["blend"] and "scene.obj" not in B["grey"] and not B.get("stepwise"):
        out.append(("error", "build: no .blend (source.json) and no gray copy scene.obj"))
    if "world.json" in B["grey"] and "scene.obj" not in B["grey"]:
        out.append(("note", "build: world.json without scene.obj — no maps from the gray copy"))
    if "cameras" not in M["sections"]:
        out.append(("note", "cameras: no cameras.json"))
    return out


def summary(M):
    """A text summary by sections (10–30 lines)."""
    L = [f"scene {M['scene']}: model {'scene.yaml' if M['explicit'] else 'derived from the files'}, origin — {M['origin']}"
         + (f", fps {M['fps']}" if M["fps"] else "")]
    if M["concept"]: L.append("concept: " + " ".join(str(M["concept"]).split())[:300])
    for name, sec in M["sections"].items():
        what = ", ".join(sec["files"]) or "—"
        if sec["data"]: what += f"; data: {', '.join(sec['data'])}"
        if name == "light": what = f"{sec['source']}" + (f" ({what})" if sec["files"] and sec["source"] != sec["files"][0] else "")
        L.append(f"  {name:<14} [{sec['component'] or '—'}] {what}")
    B = M["build"]
    blend = os.path.relpath(B["blend"], os.path.dirname(os.path.dirname(M["dir"]))) if B["blend"] else "—"
    L.append(f"  {'build':<14} .blend {blend}{'' if not B['blend'] or os.path.exists(B['blend']) else ' (missing)'}; "
             f"gray copy: {', '.join(B['grey']) or '—'}" + ("; edits from edits.json" if B["edits"] else "") + ("; stepwise" if B["stepwise"] else ""))
    return "\n".join(L)


def write(M, force=False):
    """scene.yaml as a table of contents (keys in English): sections as links to the files that exist; build — a link to the .blend. → the path."""
    Y = _yaml(); p = os.path.join(M["dir"], "scene.yaml")
    if os.path.exists(p) and not force:
        raise SystemExit(f"{p} already exists — use --force to overwrite")
    doc = {"model": VERSION, "scene": M["scene"]}
    for k in ("concept", "fps"):
        if M[k]: doc[k] = M[k]
    if M["seed"]: doc["seed"] = M["seed"]
    doc["sections"] = {}
    for name, sec in M["sections"].items():
        d = dict(sec["data"] or {})
        if sec["files"]: d["files"] = sec["files"]
        if d: doc["sections"][name] = d
    b = {}
    if M["build"]["blend"]: b["blend"] = M["build"]["blend"]
    if M["build"]["stepwise"]: b["stepwise"] = True
    if b: doc["build"] = b
    open(p, "w").write("# The scene model (ARCHITECTURE.md): sections as data or as links to scene files; keys are in English\n"
                       + Y.safe_dump(doc, allow_unicode=True, sort_keys=False, width=120))
    return p


def pipeline_scenes():
    """Scenes referenced by the repository's pipelines (core, formats, productions, and the research/ and runs/ folders of components): the `scene:` folders."""
    import glob, re
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); out = set()
    pats = ["core/manager/pipelines/*.yaml", "formats/*/pipelines/*.yaml", "projects/*/pipelines/*.yaml", "research/**/*.yaml",
            "components/**/runs/**/*.yaml", "components/**/bench/**/*.yaml"]
    for f in {f for p in pats for f in glob.glob(os.path.join(root, p), recursive=True)}:
        if "/out/" in f or "/repack/vocab/" in f: continue
        try: txt = open(f, encoding="utf-8").read(20000)
        except (OSError, UnicodeDecodeError): continue
        if not re.search(r"^pipeline:", txt, re.M): continue
        m = re.search(r"^scene:\s*(\S+)", txt, re.M)
        if m and os.path.isdir(os.path.join(root, m.group(1))): out.add(m.group(1))
    return sorted(out)


def main(argv):
    if argv[:1] == ["check-pipelines"]:                              # level 2: scene models of all the repository's pipelines
        argv = ["check"] + pipeline_scenes() + argv[1:]
    if len(argv) < 2 or argv[0] not in ("summary", "check", "files", "write"):
        print(__doc__); return 2
    cmd, js = argv[0], "--json" in argv
    dirs = [a for a in argv[1:] if not a.startswith("--")]
    bad = 0
    for d in dirs:
        M = load(d)
        if cmd == "summary":
            print(json.dumps(M, ensure_ascii=False) if js else summary(M))
        elif cmd == "files":
            F = files(M, copy="--copy" in argv)
            print(json.dumps(F, ensure_ascii=False) if js else "\n".join(F))
        elif cmd == "write":
            print(write(M, force="--force" in argv))
        else:
            R = check(M); e = sum(1 for lv, _ in R if lv == "error"); bad += e
            if js: print(json.dumps({"scene": M["scene"], "n_errors": e, "check": R}, ensure_ascii=False))
            else:
                print(f"{M['scene']}: {'OK' if not e else f'n_errors {e}'}" + "".join(f"\n  {lv}: {t}" for lv, t in R))
    return min(bad, 100)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
