"""Pipeline manager: runs a YAML pipeline — component steps and outputs over the scene model.
The order and the contract check against the passports before launch, a copy of the run's scene, the run context, a cache keyed by the input hash, heavy work
under guard, the run journal. It works on wrappers of the current commands (the passport's "command" field); a step that is still glued together in the
render through output keys (light, weather) hands its keys over to the outputs (the passport's "output keys").
.venv/bin/python -m core.manager.pipeline <pipeline.yaml> [--scene bench/scenes/X] [--out out/dev/…] [--profile sketch|preview|final]
    [--dry] [--fresh] [--only step[,step]] [--from step] [--until step]
  → <folder>/journal.jsonl (a line per step), journal.md (the run summary), context.json; the exit code is the number of failures.
Passports: components/*/component.yaml (component steps), core/manager/outputs.yaml (core outputs), formats/*/format.yaml (thin
format steps; format pipelines — formats/<format>/pipelines/*.yaml)."""
import argparse, glob, hashlib, json, os, re, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
import core.paths     # noqa: E402
from core.model import scene_model  # noqa: E402
import yaml  # noqa: E402

ROOT = core.paths.ROOT
PY = os.path.join(ROOT, ".venv", "bin", "python")
PROFILES = ("sketch", "preview", "final")          # the canonical vocabulary (enums.profile); the Russian aliases of sketch|preview|final are accepted on input
VERSION = "pipeline-v2"      # v2: the key includes the import closure, the input .blend, the Blender version and the output signature; the label is written after the output check; --from takes the input from disk
# v1 notes below      # v1: a stepwise build in a single Blender ("build: stepwise" in a pipeline)


# ── passports ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def registry():
    """All steps and outputs: name → step passport (+ component, version, folder, source)."""
    R = {}
    srcs = (sorted(glob.glob(os.path.join(ROOT, "components", "*", "component.yaml"))) + [os.path.join(HERE, "outputs.yaml")]
            + sorted(glob.glob(os.path.join(ROOT, "formats", "*", "format.yaml"))))         # thin format steps — with the same passport
    bad = []
    for p in srcs:
        doc = yaml.safe_load(open(p)) or {}
        bad += _passport_errors(p, doc)
        for name, st in (doc.get("steps") or {}).items():
            st = dict(st or {})
            if name in R:
                raise SystemExit(f"passport {os.path.relpath(p, ROOT)}: step '{name}' already exists in {R[name]['source']}")
            R[name] = dict(st, name=name, component=doc.get("component", "core"), version=doc.get("version", 1),
                           component_dir=os.path.dirname(p), source=os.path.relpath(p, ROOT),
                           is_output=os.path.basename(p) == "outputs.yaml" or st.get("kind") == "output")   # an output reads whatever exists
    if bad: raise SystemExit("step passports:\n  " + "\n  ".join(bad))
    return R


def _passport_errors(p, doc):
    """A passport — the minimum a run needs: a step has plan, build or command; reads, writes, may_read are lists; keys is a dict."""
    out = []
    for name, st in (doc.get("steps") or {}).items():
        st = st or {}
        if not any(st.get(f) for f in ("plan", "build", "command", "output_keys")): out.append(f"{name}: no plan, build or command")
        out += [f"{name}: {f} is not a list" for f in ("reads", "writes", "may_read") if f in st and not isinstance(st[f], list)]
        if "keys" in st and not isinstance(st["keys"], (dict, type(None))): out.append(f"{name}: keys is not a dict")
    return [f"{os.path.relpath(p, ROOT)}: {e}" for e in out]


def _norm(entry):
    """An element of the pipeline's step list: "name" or {name: {keys}} → (name, keys)."""
    if isinstance(entry, str): return entry, {}
    (name, keys), = entry.items()
    return name, dict(keys or {})


# ── hashes ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _file_sig(p):
    """A file signature for the cache: the content (up to 256 MB) or the size and time (large files, .blend)."""
    if not os.path.exists(p): return "none"
    st = os.stat(p)
    if p.endswith(".blend") or st.st_size > 256 << 20: return f"{st.st_size}:{st.st_mtime_ns}"   # nanoseconds: two runs within one second — different inputs
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()[:16]


_IMPORTS = {}                 # path → (mtime, {project module files}) — the file's import analysis, once per run
_SIG_SKIP = ("core/paths.py", "core/contract/", "core/services/")   # infrastructure is not geometry: editing it does not rebuild steps


def _imports_of(path):
    """Project module files imported by a file (the core/components/formats packages; `import a.b`, `from a.b import c`)."""
    try: mt = os.stat(path).st_mtime
    except OSError: return set()
    hit = _IMPORTS.get(path)
    if hit and hit[0] == mt: return hit[1]
    mods = []
    try:
        import ast
        for n in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
            if isinstance(n, ast.Import): mods += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0: mods += [n.module] + [f"{n.module}.{a.name}" for a in n.names]
    except (SyntaxError, UnicodeDecodeError, OSError): pass
    files = set()
    for m in mods:
        if m.split(".")[0] not in ("core", "components", "formats"): continue
        base = os.path.join(ROOT, *m.split("."))
        for c in (base + ".py", os.path.join(base, "__init__.py")):
            if os.path.isfile(c): files.add(c); break
    _IMPORTS[path] = (mt, files)
    return files


_OWN_SKIP = {"runs", "out", "ext", "bench", "trials", "__pycache__", "templates", "pipelines"}


def _own_files(cdir):
    """Code and data of a component folder: .py, .yaml, .json up to 1 MB (excluding runs/, out/, ext/, bench/, trials/, pipelines/)."""
    out = []
    for cur, subs, fs in os.walk(cdir):
        subs[:] = [d for d in subs if d not in _OWN_SKIP and not d.startswith(".")]
        for f in fs:
            q = os.path.join(cur, f)
            if f.endswith((".py", ".yaml", ".yml", ".json")) and os.path.getsize(q) < 1 << 20: out.append(q)
    return out


def _code_sig(st, depth=4):
    """The step's code signature — assembled by the manager itself: the plan/build modules, the command script (a .py path or -m module), the files of the
    component folder and the closure of their package imports up to depth (the _SIG_SKIP infrastructure is excluded); the passport's "code" —
    only what cannot be found that way (library/ data, scripts called as a subprocess)."""
    seeds = set()
    for pat in st.get("code") or []: seeds |= set(glob.glob(os.path.join(ROOT, pat)))
    for f in ("plan", "build"):
        fn = _fn(st, f) if st.get("component_dir") else None
        if fn: seeds.add(fn[0])
    cmd = [str(a).replace("{root}", ROOT) for a in st.get("command") or []]
    for i, a in enumerate(cmd):
        if a.endswith(".py") and os.path.isfile(a): seeds.add(a)
        if a == "-m" and i + 1 < len(cmd):
            m = os.path.join(ROOT, *cmd[i + 1].split(".")) + ".py"
            if os.path.isfile(m): seeds.add(m)
    if st.get("component") != "core" and st.get("component_dir"): seeds |= set(_own_files(st["component_dir"]))
    out = {os.path.relpath(p, ROOT): _file_sig(p) for p in sorted(seeds)}

    def close(frontier):
        for _ in range(depth):
            nxt = []
            for f in frontier:
                for c in _imports_of(f):
                    rel = os.path.relpath(c, ROOT)
                    if rel not in out and not rel.startswith(_SIG_SKIP) and "/ext/" not in rel:
                        out[rel] = _file_sig(c); nxt.append(c)
            frontier = nxt
            if not frontier: break
    close([p for p in sorted(seeds) if p.endswith(".py")])
    if "core/hooks.py" in out:                                       # core extension points: component handlers from the passports
        from core import hooks as _H                                 # (core/hooks.py) — also the step's code, with their closure
        extra = [p for p in _H.files() if os.path.relpath(p, ROOT) not in out]
        for p in extra: out[os.path.relpath(p, ROOT)] = _file_sig(p)
        close([p for p in extra if p.endswith(".py")])
    return out


def _scene_sig(SD, reads, exclude=()):
    """The signature of the run's scene-model sections that the step reads (and writes — its input before the step; "all" — the whole model and the build);
    exclude — files written by the step itself ("scene files"): otherwise its output would enter its own key."""
    M = scene_model.load(SD)
    if "all" in reads: names = scene_model.files(M)
    else:
        names = [f for r in reads for f in (M["sections"].get(r) or {}).get("files", [])]
    sig = {f: _file_sig(os.path.join(SD, f)) for f in sorted(set(names) - set(exclude))}
    if "all" in reads and M["build"]["blend"]: sig[".blend"] = _file_sig(M["build"]["blend"])
    return sig


_BVER = None


def _blender_version(dry=False):
    """The Blender version string for the cache key (once per process); a dry run or no Blender — "none"."""
    global _BVER
    if _BVER is None:
        _BVER = "none"
        if not dry and shutil.which("blender"):
            try:
                o = subprocess.run(["blender", "--version"], capture_output=True, text=True, timeout=60).stdout
                _BVER = next((l.strip() for l in o.splitlines() if l.startswith("Blender")), "none")
            except Exception: pass                                    # noqa: BLE001 — a probe, not a step: any failure = "none"
    return _BVER


def _metrics_of(outdir):
    """Frame metrics in the step folder (core.outputs.metrics): {file: {sky_fraction, horizon_row, …}} or None if there are no frames."""
    try:
        from core.outputs import metrics as MX
    except Exception:                                                  # noqa: BLE001 — no numpy/PIL in this Python: no metrics
        return None
    out = {}
    for dp, dn, fn in os.walk(outdir):
        if dp[len(outdir):].count(os.sep) > 1: dn[:] = []; continue
        for f in fn:
            if f.endswith(".png") and ("lit" in f or dp == outdir):
                p = os.path.join(dp, f)
                try: out[os.path.relpath(p, outdir)] = MX.frame_metrics(p)
                except Exception as e: out[os.path.relpath(p, outdir)] = {"error": f"{type(e).__name__}: {e}"}   # noqa: BLE001
    return out or None


def _step_result(outdir):
    """What the step's functions returned (the plan — plan.json, the build — build.json → gives) — one dict; the build wins."""
    out = {}
    for f, k in (("plan.json", None), ("build.json", "gives")):
        p = os.path.join(outdir, f)
        try: d = json.load(open(p)) if os.path.exists(p) else None
        except (OSError, ValueError): d = None
        d = (d.get(k) if k and isinstance(d, dict) else d)
        if isinstance(d, dict): out.update(d)
    return out


def _layers_of(outdir):
    """Accounting of the scene layers a step read (core.layers.SEEN from plan.json and build.json): {layer: whether it existed and was non-empty}."""
    out = {}
    for f, k in (("plan.json", "_layers"), ("build.json", "layers")):
        p = os.path.join(outdir, f)
        try: d = json.load(open(p)) if os.path.exists(p) else {}
        except (OSError, ValueError): d = {}
        for n, v in ((d.get(k) if isinstance(d, dict) else None) or {}).items(): out[n] = out.get(n, False) or bool(v)
    return out


def _check_layers(st, outdir, row):
    """Layers that the step read but that did not exist (or were empty) go into the step line ("no layer …"); → those of them that come from the step's reads
    section (a required input): a step with them fails (otherwise it would silently run without its input)."""
    from core import layers as LY
    lay = _layers_of(outdir); miss = sorted(n for n, v in lay.items() if not v)
    if lay: row["layers"] = lay
    if miss: row["no_layer"] = miss
    return [n for n in miss if LY.SECTION.get(n) in (st.get("reads") or [])]


def _budget(st, profile, row):
    """The passport's "budget: {profile: sec | {sec: …, gb: …}}": exceeding the time or the memory peak — over_budget in the journal line."""
    bud = (st.get("budget") or {}).get(profile)
    if bud is None: return
    sec, gb = (bud, None) if isinstance(bud, (int, float)) else ((bud or {}).get("sec"), (bud or {}).get("gb"))
    over = []
    if sec and (row.get("sec") or 0) > sec: over.append(f"{row['sec']} s > {sec} s")
    if gb and (row.get("memory_gb") or 0) > gb: over.append(f"{row['memory_gb']} GB > {gb} GB")
    if over: row["over_budget"] = "; ".join(over)


def _outputs_of(outdir, limit=2000):
    """Files the step left in its folder (except service files) go into the label: if even one has vanished, the cache is a miss."""
    skip = {".step.json", "step.json", "step.log", "plan.json", "build.json", "scene.blend"}   # the checkpoint .blend is handled separately (without_blend)
    out = []
    for dp, dn, fn in os.walk(outdir):
        dn[:] = [d for d in dn if d != "scene"]                        # the "scene files" snapshot is not an output
        for f in fn:
            if f not in skip and not f.startswith("."):
                out.append(os.path.relpath(os.path.join(dp, f), outdir))
                if len(out) >= limit: return sorted(out)
    return sorted(out)


def _lock(folder):
    """One run per folder: a .lock with the pid; a live foreign pid — refusal (None), a dead one — takeover. → the release function."""
    p = os.path.join(folder, ".lock")
    if os.path.exists(p):
        try: pid = int(open(p).read().strip() or 0)
        except ValueError: pid = 0
        if pid and pid != os.getpid():
            try: os.kill(pid, 0); return None                     # alive — the folder is busy
            except OSError: pass                                      # dead — take it over
    open(p, "w").write(str(os.getpid()))
    def release():
        try:
            if os.path.exists(p) and open(p).read().strip() == str(os.getpid()): os.remove(p)
        except OSError: pass
    return release


def _keyfile_sig(st, keys):
    """Signature of the files named by the step's keys (the passport's "key files": a page description, a take): editing a description gives a new cache key."""
    vals = defaults(st); vals.update(keys)
    return {k: _file_sig(os.path.join(ROOT, str(vals[k]))) for k in st.get("key_files") or [] if vals.get(k)}


def _key(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ── step command ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def _argv(st, keys, ctx, outdir, extra):
    """The step command from the passport template: {py} {root} {scene} {out} {context} and the step keys {key}; keys with a "flag"
    (flags: {key: --x | x=}) go to the tail; the profile — the passport's command_profiles; extra — output keys from earlier steps."""
    vals = defaults(st); vals.update(keys)
    sub = dict(ctx["gives"], **vals, py=PY, root=ROOT, scene=ctx["run_scene"], out=outdir, context=ctx["path"])   # the earlier steps' gives too
    class _Keep(dict):
        def __missing__(self, k): return "{" + k + "}"                   # no value — the template as is; the manager will reject launching such a command
    argv = [str(a).format_map(_Keep(sub)) for a in st["command"]]
    used = {k for a in st["command"] for k in vals if "{" + k + "}" in str(a)}
    for k, flag in (st.get("flags") or {}).items():
        if k in vals and k not in used and vals[k] not in (None, "", False):
            argv += [f"{flag}{vals[k]}"] if str(flag).endswith("=") else ([flag] if vals[k] is True else [flag, str(vals[k])])
    argv += [str(a) for a in (st.get("command_profiles") or {}).get(ctx["profile"], [])]   # old path: command arguments by profile
    return argv + (extra if st.get("accepts_step_keys") else [])


def _glue(st, keys):
    """An "output keys" step (glued together in the render): the step's keys → the outputs' arguments."""
    out = []
    for k, flag in (st.get("output_keys") or {}).items():
        v = keys.get(k, (st.get("keys") or {}).get(k))
        if v not in (None, "", False):
            out += [f"{flag}{v}"] if str(flag).endswith("=") else [flag, str(v)]
    return out


# ── stepwise build (core/blender/build.py) ─────────────────────────────────────────────────────────────────────────────
def _fn(st, field):
    """The passport's "plan: file.py:function" / "build: file.py:function" → (module path, function); otherwise None (the field is a description)."""
    v = st.get(field)
    if not isinstance(v, str) or ":" not in v: return None
    f, fn = v.rsplit(":", 1)
    if not f.endswith(".py") or not fn.isidentifier(): return None
    path = os.path.normpath(os.path.join(st["component_dir"], f))
    return (path, fn) if os.path.exists(path) else None


def _run_build(st, sname, keys, ctx, outdir, blend_out, fn_build, name, env):
    """A build step: the "plan" phase in the project's Python (if there is one), the "build" phase in Blender over the previous step's checkpoint .blend.
    Heavy — under guard. → (process, command)."""
    spec = {"step": sname, "section": st.get("section") or sname, "module": fn_build[0] if fn_build else None, "function": fn_build[1] if fn_build else None,
            "plan": os.path.join(outdir, "plan.json"), "input": ctx["blend"], "output": blend_out, "is_output": bool(st.get("is_output")),
            "scene": ctx["run_scene"], "out": outdir, "profile": ctx["profile"], "keys": dict(defaults(st), **keys),   # passport defaults + pipeline keys
            "gives": ctx["gives"], "component": st["component"]}
    sp = os.path.join(outdir, "step.json")
    plan = _fn(st, "plan")
    if plan: spec.update(plan_module=plan[0], plan_function=plan[1])
    else: json.dump(dict(keys), open(spec["plan"], "w"), ensure_ascii=False)   # without a plan phase — the plan = the step's keys
    json.dump(spec, open(sp, "w"), ensure_ascii=False, indent=1)
    log = open(os.path.join(outdir, "step.log"), "w")
    if plan:
        cmd = [PY, "-m", "core.manager.plan_run", sp]; log.write(" ".join(cmd) + "\n"); log.flush()
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
        if p.returncode: log.close(); return p, cmd
        if fn_build is None:                                           # a step with only a "plan" phase (project Python, no Blender): the output is the plan and "gives"
            try: pl = json.load(open(spec["plan"]))
            except (OSError, ValueError): pl = {}
            gives = {k: v for k, v in (pl.items() if isinstance(pl, dict) else []) if isinstance(v, (str, int, float, bool))}
            json.dump({"step": sname, "section": spec["section"], "new_objects": 0, "sec": 0, "gives": gives, "plan": spec["plan"]},
                      open(os.path.join(outdir, "build.json"), "w"), ensure_ascii=False, indent=1)
            log.close(); return p, cmd
    cmd = ["blender", "-b", "--factory-startup", "--python-exit-code", "1", "-P", os.path.join(ROOT, "core", "blender", "build.py"), "--", sp]
    if st.get("cost") == "heavy":
        cmd = [PY, "-m", "core.services.guard", "heavy", "--wait", "1800", "--what", f"{name}:{sname}", "--", *cmd]
    log.write(" ".join(cmd) + "\n"); log.flush()
    p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env); log.close()
    return p, cmd


def gives_names(st):
    """Names a step hands to the following ones: the passport's gives field (a list of names or {name: description})."""
    g = st.get("gives")
    return set(g) if isinstance(g, (list, dict)) else set()


def _after_build(st, ctx, outdir, blend_out, row=None):
    """After a build step (and from cache): the checkpoint .blend is the next one's input. The step function's return value: names from the passport's gives go
    into the run context (data for the following steps and outputs), other scalars are step statistics for the journal line (stats)."""
    if blend_out: ctx["blend"] = blend_out
    bj = os.path.join(outdir, "build.json")
    if os.path.exists(bj):
        declared, stats = gives_names(st), {}
        for k, v in (json.load(open(bj)).get("gives") or {}).items():
            if not isinstance(v, (str, int, float, bool)): continue
            if k in declared: ctx["gives"][k] = v
            else: stats[k] = v
        if row is not None and stats: row["stats"] = stats


# ── allowed step keys: the passport (keys, values, description, command keys, open keys, synonyms) ─────────────────────────────
# keys:            all allowed keys of the step; the value is the default (null — the key is allowed, there is no default: the step decides itself)
# values:          {key: [list] | type | catalog:file[#path]}; a type is string, float, int, WxH, bool, point ("x,y"); a list may contain
#                  a "catalog:…" element (expanded by code). Catalog: .md — a table with `(`id`)` (library/terrain/CATALOG.md), .yaml — the keys
#                  of the section at the path "#a.b" (castles/castles.yaml#cultures), .py — a top-level name "#WEATHER" (dict keys or tuple elements)
# description:     {key: short string} where the meaning is not obvious
# synonyms:        {key: canonical key} (the Russian and English names of one knob); values and description are taken from the canonical one
# command_keys:    a list; keys of the command input only (they do not apply in the stepwise build)
# open_keys:       true — any keys are allowed (the knobs of a site preset); known ones are still checked by values
# hint:            one or two lines for the caller (order, links), without value lists — the code fills those in
BUILTIN_PLACEHOLDERS = {"py", "root", "scene", "out", "context"}


def profile_keys(st, profile):
    """Step knobs by profile from the passport: profiles: {sketch: {knob: value}, …} — cheaper in the sketch, full in the final."""
    p = (st.get("profiles") or {}).get(profile)
    return dict(p) if isinstance(p, dict) else {}


def profile_mode(st, profile):
    """The step's stage: the passport's profiles: {stage: off | plan | {knobs}} → "off" — the step does not run at the stage; "plan" — only the "plan" phase
    (data and layers for neighbors, no Blender); otherwise None — the step runs as usual."""
    p = (st.get("profiles") or {}).get(profile)
    if p is False or p == "off": return "off"                         # YAML 1.1: an unquoted off is read as false
    return "plan" if p == "plan" else None


def defaults(st):
    """Step key defaults from the passport: only those that are set (null — "allowed, no default")."""
    return {k: v for k, v in (st.get("keys") or {}).items() if v is not None}


def _placeholders(st):
    return {m for a in st.get("command") or [] for m in re.findall(r"\{([^{}]+)\}", str(a))} - BUILTIN_PLACEHOLDERS


_CAT = {}


def catalog_values(ref):
    """"path[#section]" (from the project root) → a list of values from the catalog (an md table of presets, a yaml section, a top-level name in a .py)."""
    if ref in _CAT: return _CAT[ref]
    path, _, frag = ref.partition("#")
    full = os.path.join(ROOT, path)
    if not os.path.exists(full): raise SystemExit(f"values catalog '{ref}': no file {path}")
    out = []
    if path.endswith(".md"):
        out = [m.group(1) for m in (re.match(r"^\|\s*.+?\s*\(`([a-z_0-9]+)`\)\s*\|", ln) for ln in open(full, encoding="utf-8")) if m]
    elif path.endswith((".yaml", ".yml")):
        d = yaml.safe_load(open(full, encoding="utf-8")) or {}
        for part in filter(None, frag.split(".")): d = (d or {}).get(part)
        out = list(d) if isinstance(d, (dict, list, tuple)) else []
    elif path.endswith(".py"):
        import ast
        for node in ast.parse(open(full, encoding="utf-8").read()).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == frag for t in node.targets):
                v = node.value
                out = [ast.literal_eval(k) for k in v.keys] if isinstance(v, ast.Dict) else list(ast.literal_eval(v))
    if not out: raise SystemExit(f"values catalog '{ref}': empty (the path or section was not found)")
    _CAT[ref] = [str(x) for x in out]
    return _CAT[ref]


def expand_values(spec):
    """The passport's "values" of a key → a list of allowed values (catalogs expanded) or a type string (string, float, int, WxH, bool, point)."""
    if isinstance(spec, str):
        return catalog_values(spec[len("catalog:"):]) if spec.startswith("catalog:") else spec
    out = []
    for x in spec or []:
        if isinstance(x, str) and x.startswith("catalog:"): out += catalog_values(x[len("catalog:"):])
        else: out.append(x)
    return out


def _vs(v):
    return str(v).lower() if isinstance(v, bool) else str(v)


def value_ok(v, spec):
    """A key's value against the expanded "values": a list — membership; a type — parsing; an unknown type or empty — allowed."""
    if isinstance(spec, list): return _vs(v) in {_vs(x) for x in spec}
    s = str(v).strip()
    try:
        if spec == "float": float(s)
        elif spec == "int": int(s)
        elif spec == "WxH": return re.fullmatch(r"\d+x\d+", s) is not None
        elif spec == "bool": return isinstance(v, bool) or s.lower() in ("true", "false")
        elif spec == "point": return len(s.split(",")) >= 2 and all(float(t) is not None for t in s.split(",")[:2])
    except ValueError:
        return False
    return True


def key_spec(st, blend=False):
    """Allowed step keys. blend — the stepwise input (there is a "build:" function and a "build: stepwise" pipeline): command keys do not apply.
    → {allowed: [..], command: [..], open: bool, defaults, values: {canonical key: list|type}, description, synonyms}."""
    cmd = set(st.get("command_keys") or [])
    base = (set(st.get("keys") or {}) | set(st.get("flags") or {}) | set(st.get("output_keys") or {}) | set(st.get("key_files") or [])
            | (set() if blend else _placeholders(st)) | set(st.get("values") or {}))     # the command's {key} — command input only
    allowed = base - cmd if blend else base | cmd
    allowed = [k for k in (st.get("keys") or {}) if k in allowed] + sorted(k for k in allowed if k not in (st.get("keys") or {}))   # passport order first, then the rest
    vals = {k: expand_values(v) for k, v in (st.get("values") or {}).items()}
    return {"allowed": allowed, "command": sorted(cmd), "open": bool(st.get("open_keys")), "defaults": defaults(st),
            "values": vals, "description": dict(st.get("description") or {})}


def check_placeholders(steps, R, blend_mode):
    """Pre-launch contract: for a step that will run as a command, each {name} template gets a value — a pipeline knob, a passport default,
    or the gives of an earlier step (declared or command_gives); otherwise the command would go out with "{name}" literally."""
    errs, have = [], set()
    for name, keys in steps:
        st = R.get(name)
        if not st: continue
        as_command = bool(st.get("command")) and not (blend_mode and _fn(st, "build"))   # as in the run: a plan step with a command runs as a command
        if as_command:
            ph = {m for a in st["command"] for m in re.findall(r"\{([A-Za-z_]\w*)\}", str(a))} - BUILTIN_PLACEHOLDERS
            miss = sorted(ph - set(keys) - set(defaults(st)) - have)
            if miss: errs.append(f"step '{name}': the command has no value for {{{', '.join(miss)}}} — the knob is not set and earlier steps do not provide it")
        have |= gives_names(st) | set(st.get("command_gives") or {})
    return errs


def check_keys(steps, R, blend_mode):
    """Pipeline step keys against the passports before launch: an unknown key and an invalid value are errors."""
    errs = []
    for name, keys in steps:
        st = R.get(name)
        if not st or not keys: continue
        use_build = blend_mode and bool(_fn(st, "build"))
        sp = key_spec(st, use_build)
        for k, v in keys.items():
            if k not in sp["allowed"] and not sp["open"]:
                why = " (a command-input key: does not apply in the stepwise build)" if use_build and k in sp["command"] else ""
                errs.append(f"step '{name}': unknown key '{k}'{why}; allowed: {', '.join(sp['allowed']) or 'no keys'}")
                continue
            spec = sp["values"].get(k)
            if spec is not None and v is not None and not value_ok(v, spec):
                shown = spec if isinstance(spec, str) else "|".join(map(str, spec))
                errs.append(f"step '{name}': key '{k}' = '{v}' — allowed: {shown}")
    return errs


# ── run ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
def check_order(steps, R, M, off=(), profile=None):
    """Pre-launch contract: the step exists in the passports; it reads sections that exist in the model or are written by a step before it. A step disabled at a stage (off)
    neither reads nor writes: a reader of its section with no other writer is an error naming the stage."""
    errs, have, dropped = [], set(M["sections"]) | {"all"}, {}
    for name, _ in steps:
        st = R.get(name)
        if not st:
            errs.append(f"step '{name}': not in the passports (available: {', '.join(sorted(R))})"); continue
        if name in off:
            for w in st.get("writes") or []: dropped.setdefault(w, name)
            continue
        for r in st.get("reads") or []:
            if r not in have and r not in scene_model.SECTIONS:
                errs.append(f"step '{name}': reads '{r}' — there is no such model section")
            elif r not in have and not st.get("is_output"):
                why = f" (its writer '{dropped[r]}' is disabled at stage {profile} — the passport's profiles)" if r in dropped else ""
                errs.append(f"step '{name}': reads '{r}', which is not in the scene model and nobody writes it before the step{why}")
        have |= set(st.get("writes") or [])
    return errs


def _scene_state(SD):
    """The run's scene files (top level, through links) → {name: signature}: a snapshot before and after the step."""
    out = {}
    if not os.path.isdir(SD): return out
    for f in sorted(os.listdir(SD)):
        q = os.path.join(SD, f)
        if os.path.isfile(q): out[f] = _file_sig(q)
    return out


def _snapshot(st, SD, outdir, before, row):
    """A step's contribution to the run's scene — everything it wrote or changed (not only what is declared in scene_files): copies in <step>/scene,
    names go into the label (everything is restored from cache). Undeclared items go into the step line: it shows who writes to the scene bypassing the passport."""
    after = _scene_state(SD); decl = list(st.get("scene_files") or [])
    out = sorted({f for f, sg in after.items() if before.get(f) != sg} | {f for f in decl if f in after})
    if out: os.makedirs(os.path.join(outdir, "scene"), exist_ok=True)
    for f in out: shutil.copy(os.path.join(SD, f), os.path.join(outdir, "scene", f))
    und = [f for f in out if f not in decl]
    if und: row["scene_undeclared"] = und
    if out: row["wrote"] = out
    return out


def _put(src, dst, link):
    """A file of the step's contribution to the run's scene: a copy (run) or a link (dry run — in a temporary copy, nothing is built)."""
    if os.path.lexists(dst): os.remove(dst)
    if link: os.symlink(src, dst)
    else: shutil.copy(src, dst)


def _drop_blend(p):
    """Delete the step's checkpoint .blend (and .blend1); → bytes freed."""
    n = 0
    for f in (p, p + "1"):
        if os.path.exists(f): n += os.path.getsize(f); os.remove(f)
    return n


def run(conv_path, scene=None, folder=None, profile=None, dry=False, fresh=False, only=None, since=None, until=None, redo_from=None, notes=None):
    """Run the pipeline. until — execute up to and including the step, then stop (combines with since); redo_from, notes are internal: a restart
    when a step needs a checkpoint .blend that was deleted by the "checkpoints: last" mode (the steps are rebuilt starting from index redo_from)."""
    C = yaml.safe_load(open(conv_path)) or {}
    if not (C.get("steps") or C.get("outputs")) and not C.get("scene"):
        print(f"{os.path.relpath(conv_path, ROOT)}: not a pipeline (no steps/outputs/scene) — it is a description for a format step, not a run"); return 100
    name = C.get("pipeline") or os.path.splitext(os.path.basename(conv_path))[0]
    scene = os.path.abspath(os.path.join(ROOT, scene or C["scene"]))
    profile = profile or C.get("profile", "preview")
    if profile not in PROFILES: raise SystemExit(f"profile {profile}: available are {', '.join(PROFILES)}")
    keep = str(C.get("checkpoints", "all")).strip()                   # step checkpoint .blends: all | last
    if keep not in ("all", "last"): raise SystemExit(f"checkpoints: '{keep}' — available are all | last")
    folder = os.path.abspath(os.path.join(ROOT, folder or C.get("out") or f"out/dev/runs/{name}/{os.path.basename(scene)}/{profile}"))   # a folder per stage:
    # the stage is in every step's key and there is one label per folder; two stages in one folder would rebuild everything one after the other
    R, M = registry(), scene_model.load(scene)
    warns = list(notes or [])                                          # old step names (synonyms), a restart — warnings go to the output and the journal
    steps = []
    rmodes = {}                                                        # the step's stage from the recipe overrides the passport
    for e in (C.get("steps") or []) + (C.get("outputs") or []):
        n, k = _norm(e)
        ov = (k.pop("stages", None) or {}).get(profile)                # recipe: stages: {stage: {knobs} | plan | off} — key frames,
        if ov is False or ov in ("off", "plan"): rmodes[n] = "plan" if ov == "plan" else "off"   # trials, masonry in a sketch close-up
        elif isinstance(ov, dict): rmodes[n] = "on"; k.update(ov)
        steps.append((n, k))
    steps = [(n, {**profile_keys(R.get(n) or {}, profile), **k}) for n, k in steps]   # profile knobs (passport: profiles) — under the pipeline's knobs
    if M.get("seed") is not None:                                      # the scene's seed — to steps with a seed knob where the pipeline did not set it
        steps = [(n, dict(k, seed=M["seed"]) if "seed" in ((R.get(n) or {}).get("keys") or {}) and "seed" not in k else k) for n, k in steps]
    only = set(only or [])
    if not redo_from:
        for w in warns: print("warning: " + w)
    blend_mode = str(C.get("build", "")).strip() == "stepwise"       # stepwise build in a single Blender (core/blender/build.py)
    modes, merrs = {}, []                                              # the step's stage from the passport: off — does not run, plan — plan only
    for n, _ in steps:
        m = rmodes.get(n) or profile_mode(R.get(n) or {}, profile)
        if m == "on": m = None                                         # the recipe told the step to run at this stage
        if m == "plan" and not _fn(R[n], "plan"): merrs.append(f"step '{n}': profiles.{profile} = plan, but the step has no plan phase")
        elif m == "plan" and not blend_mode: m = "off"                 # old command path: there is no "plan only" — the step does not run at the stage
        if m: modes[n] = m
    off = {n for n, m in modes.items() if m == "off"}
    errs = (merrs + check_order(steps, R, M, off, profile) + check_keys(steps, R, blend_mode)
            + check_placeholders([s for s in steps if s[0] not in off], R, blend_mode)
            + [f"model: {t}" for lv, t in scene_model.check(M) if lv == "error" and not (blend_mode and t.startswith("build: no .blend"))])
    names = [n for n, _ in steps]
    for opt, v in (("--from", since), ("--until", until)):
        if v and v not in names: errs.append(f"{opt} '{v}': there is no such step in the pipeline ({', '.join(names)})")
    if since and until and since in names and until in names and names.index(until) < names.index(since):
        errs.append(f"--until '{until}' is earlier than --from '{since}'")
    if errs:
        print("the contract does not hold — the run was not started:\n  " + "\n  ".join(errs)); return 100
    if until: steps = steps[:names.index(until) + 1]                   # steps after "--until" are not examined: no cache, no journal lines
    SD = os.path.join(folder, "scene")                                 # a copy of the run's scene: links to the description, its own cameras
    from core.model import scene_copy
    import tempfile
    tmp = tempfile.TemporaryDirectory(prefix="pipeline-dry-") if dry else None
    SDs = os.path.join(tmp.name, "scene") if dry else SD               # where the scene signatures are read from: a dry run — a temporary copy made of links
    if not dry: os.makedirs(folder, exist_ok=True)
    if not dry and os.path.isdir(SD): shutil.rmtree(SD)               # the run's scene is rebuilt from the original on each run: the contribution of the previous
    scene_copy.link(scene, SDs)                                        # run's steps does not leak into the keys; steps from cache put in their own snapshot
    for f in sorted({f for n, _ in steps for f in R[n].get("scene_files") or []} | {"cameras.json"}):
        src, dst = os.path.join(scene, f), os.path.join(SDs, f)        # files written by steps are their own copies, from the original on every run
        if os.path.lexists(dst): os.remove(dst)
        if os.path.exists(src): _put(src, dst, dry)
    ctx = {"pipeline": name, "scene": scene, "run_scene": SD, "profile": profile, "out": folder,
           "path": os.path.join(folder, "context.json"), "model": {k: v["files"] for k, v in M["sections"].items()},
           "gives": {}, "output_keys": [], "steps": [], "build": "stepwise" if blend_mode else "commands", "blend": None}
    b0 = (M["build"] or {}).get("blend")                              # a scene with a ready .blend: in the stepwise mode the outputs read it (the first step's input),
    if blend_mode and b0 and os.path.exists(b0) and steps and (R[steps[0][0]].get("is_output") or R[steps[0][0]].get("on_scene")):   # steps — only if
        ctx["blend"] = b0                                              # the first step does not write the scene from scratch (an output, or the passport's "on_scene: true")
    J = os.path.join(folder, "journal.jsonl"); rows, prev, fails, started = [], [], 0, since is None
    env = dict(os.environ, SK_CONTEXT=ctx["path"], SK_PROFILE=profile)
    release = (lambda: None) if dry else _lock(folder)                 # one run per folder (two processes into one folder — the second is refused)
    if release is None:
        print(f"the run folder is busy with another process ({os.path.join(folder, '.lock')} exists) — the run was not started")
        if tmp: tmp.cleanup()
        return 101
    bver = _blender_version(dry)
    jf = None if dry else open(J, "a")                                 # the journal is written line by line immediately, not at the end (an aborted run does not lose lines)
    def journal(r):
        if jf: jf.write(json.dumps(r, ensure_ascii=False) + "\n"); jf.flush()
    nofile = False                                                     # the last cached build step without a checkpoint .blend ("checkpoints: last")
    last_ok = -1                                                       # index of the last build step that has a .blend (cached or done)
    stopped = None
    for i, (sname, keys) in enumerate(steps):
        st = R[sname]; started = started or sname == since
        row = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "pipeline": name, "step": sname, "component": st["component"],
               "keys": keys, "profile": profile}
        if i == 0 and warns: row["warnings"] = warns            # old step names, a restart — into the journal (as the first line)
        mode = modes.get(sname)
        if mode == "off":                                               # stage: the step does not run, the next one gets the same .blend
            row.update(status="off", reason=f"step disabled at stage {profile} (the passport's profiles)")
            rows.append(row); journal(row); ctx["steps"].append(sname); continue
        if st.get("output_keys") and not st.get("command") and not (blend_mode and _fn(st, "build")):   # glued together in the render (not stepwise): keys go to the outputs
            args = _glue(st, keys); ctx["output_keys"] += args; prev.append(_key(sname, keys, args))
            row.update(status="keys_to_outputs", args=args); rows.append(row); journal(row); ctx["steps"].append(sname); continue
        outdir = os.path.join(folder, f"{i + 1:02d}_{sname}")
        fn_build = _fn(st, "build") if blend_mode and mode != "plan" else None   # stepwise build: the component's function over the open scene
        plan_only = bool(blend_mode and _fn(st, "plan") and (mode == "plan" or (not fn_build and not st.get("command"))))   # only the "plan" phase
        if mode == "plan": row["profile_mode"] = "plan"                # (Python, no Blender): a step with no build phase, or the passport's "plan" stage
        if fn_build:
            argv = ["build", st["build"], st.get("plan") or "", json.dumps(dict(defaults(st), **keys), ensure_ascii=False, sort_keys=True)]
        elif plan_only:
            argv = ["plan", st["plan"], json.dumps(dict(defaults(st), **keys), ensure_ascii=False, sort_keys=True)]   # with defaults: "orbit: 0" = no key
        else:
            argv = _argv(st, keys, ctx, outdir, ctx["output_keys"])
        parts = {"command": argv[1:] if argv[:1] == [PY] else argv, "profile": profile, "previous": list(prev), "code": _code_sig(st),
                 "scene": _scene_sig(SDs, (st.get("reads") or []) + (st.get("may_read") or []) + (st.get("writes") or []),
                                     exclude=[f for f in st.get("scene_files") or [] if f != "cameras.json"]),   # the cameras at the input are decisions
                 # (pinned records, cameras of earlier steps): the run's scene starts from scratch, the step will not see its own output in the key
                 "key_files": _keyfile_sig(st, keys), "blender": bver if fn_build else None,
                 "input": (_file_sig(ctx["blend"]) if ctx.get("blend") else "none") if fn_build else None}   # the real input .blend, not just the chain of keys
        key = _key(VERSION, st["component"], st["version"], sname, *parts.values())
        mark = os.path.join(outdir, ".step.json"); row["key"] = key
        mk = json.load(open(mark)) if os.path.exists(mark) else {}
        if mk and mk.get("key") != key:                                                      # cache miss — why (into the journal)
            was = mk.get("parts") or {}
            row["cache_miss"] = {k: [was.get(k), v] for k, v in parts.items() if was.get(k) != v} or "no record of the parts"
        blend_out = None if (not fn_build or st.get("is_output")) else os.path.join(outdir, "scene.blend")   # the step's checkpoint .blend
        no_blend = bool(mk.get("no_blend")) and bool(blend_out)       # this step's .blend was removed by the "checkpoints: last" mode
        cached = mk.get("key") == key and (not blend_out or os.path.exists(blend_out) or no_blend)
        if cached and blend_out and os.path.exists(blend_out) and mk.get("output_sig") != _file_sig(blend_out):
            cached = False; row["cache_miss"] = {"output": "the checkpoint .blend is newer than the label (the run was interrupted after saving)"}   # S3
        if cached and any(not os.path.exists(os.path.join(outdir, f)) for f in mk.get("outputs") or []):
            cached = False; row["cache_miss"] = {"output": "the step's output file was deleted"}                                             # a frames output with its frames missing
        if redo_from is not None and i >= redo_from: cached = False    # restart: there is no .blend for the input — rebuild from redo_from
        skip = (only and sname not in only) or not started
        will_run = not skip and (fresh or not cached)
        if will_run and blend_mode and nofile and redo_from is None and (fn_build or st.get("is_output")):   # the step needs the previous step's .blend, but it was removed ("checkpoints: last")
            j = last_ok + 1
            msg = (f"restart from step '{steps[j][0]}': the step before '{sname}' has no checkpoint .blend (checkpoints: last) — "
                   f"steps '{', '.join(n for n, _ in steps[j:i])}' are rebuilt" if j < i else f"restart: there is no checkpoint .blend before '{sname}'")
            if tmp: tmp.cleanup()
            if jf: jf.close()
            release()
            return run(conv_path, scene, folder, profile, dry, fresh, only, None, until, redo_from=j, notes=warns + [msg])   # without --from: steps from j are built, not skipped
        if skip or (cached and not fresh):
            row.update(status="skipped" if skip and not cached else "cached")
            for f in mk.get("scene_out") or st.get("scene_files") or []:   # the step's contribution to the run's scene — from the snapshot (everything it wrote)
                if os.path.exists(os.path.join(outdir, "scene", f)): _put(os.path.join(outdir, "scene", f), os.path.join(SDs, f), dry)
            if fn_build or plan_only:
                _check_layers(st, outdir, row)                          # layer-read accounting — for a step from cache as well (visible in the journal)
                if cached and _step_result(outdir).get("_skipped"):     # "skipped: …" of a step from cache — also in the line (otherwise it was lost)
                    row["skipped_part"] = str(_step_result(outdir)["_skipped"])[:300]
                if cached or (skip and os.path.exists(os.path.join(outdir, "build.json"))):   # --from: the input is whatever lies on disk, even if the key did not match
                    if not cached: row["input_from_disk"] = "key mismatch, taken from disk (--from)"
                    _after_build(st, ctx, outdir, blend_out if (blend_out and os.path.exists(blend_out)) else None, row)
                if blend_out and not os.path.exists(blend_out): nofile = True; row["no_blend"] = True   # and a skipped step without a .blend (an empty folder, --from):
                elif blend_out: nofile = False; last_ok = i                                                # the next step has no input → a restart, not "EMPTY"
        elif dry:
            row.update(status="would_run", command=argv)
        elif fn_build or plan_only:
            os.makedirs(outdir, exist_ok=True)
            if os.path.exists(mark): os.remove(mark)                   # an old label must not survive a step failure
            before = _scene_state(SD)
            t0 = time.time(); p, cmd = _run_build(st, sname, keys, ctx, outdir, blend_out, fn_build, name, env)
            tail = open(os.path.join(outdir, "step.log"), errors="replace").read().strip().splitlines()[-3:]
            ok = p.returncode == 0 and (not blend_out or os.path.exists(blend_out)) and os.path.exists(os.path.join(outdir, "build.json"))
            if p.returncode == 0 and not ok:                           # the process exited with 0 but there is no output — that is a failure, not a cache hit
                tail.append("the step left no checkpoint .blend or build.json")
            need = _check_layers(st, outdir, row)                      # a required-input layer is empty — a failure, before the cache label
            if ok and need: ok = False; tail.append(f"no layer {', '.join(need)} — the step's reads section (an earlier step writes it)")
            res_ = _step_result(outdir)                                # the step itself said it did not do part of the work: "skipped: …" in the line
            if res_.get("_skipped"): row["skipped_part"] = str(res_["_skipped"])[:300]
            if ok and st.get("kind") == "adds" and fn_build and not (json.load(open(os.path.join(outdir, "build.json"))).get("new_objects")) \
                    and _scene_state(SD) == before and not res_.get("_skipped"):   # it "adds", yet it added no object and wrote nothing
                                                                                 # and did not say why — a failure; with "skipped: …" — honest
                ok = False; tail.append("the step created nothing: neither an object in the scene nor a file in the run's scene")
            row.update(status="done" if ok else "failed", code=p.returncode, sec=round(time.time() - t0, 1),
                       tail=[t[-200:] for t in tail], command=cmd, blend=blend_out)
            if ok:                                                     # the step's cost from build.json: Blender's peak memory, the .blend size, new objects
                bi = json.load(open(os.path.join(outdir, "build.json")))
                for k in ("memory_gb", "blend_mb", "new_objects", "contract"):
                    if bi.get(k): row[k] = bi[k]                       # contract: {without_sk_id, without_sk_kind, outside_section_collection} — a warning
            _budget(st, profile, row)
            if ok:
                sout = _snapshot(st, SD, outdir, before, row)
                json.dump({"key": key, "step": sname, "t": row["t"], "sec": row["sec"], "parts": parts,
                           "output_sig": _file_sig(blend_out) if blend_out else None, "scene_out": sout,
                           "outputs": _outputs_of(outdir)}, open(mark, "w"), ensure_ascii=False)   # the label is written after the output check; outputs — what must be there
                _after_build(st, ctx, outdir, blend_out, row)
                if blend_out:
                    nofile = False; last_ok = i
                    if keep == "last":                             # previous checkpoint .blends go away after a successful step
                        for q, (qn, _) in enumerate(steps[:i]):
                            qd = os.path.join(folder, f"{q + 1:02d}_{qn}"); qb = os.path.join(qd, "scene.blend"); qm = os.path.join(qd, ".step.json")
                            if os.path.exists(qb) and os.path.exists(qm):
                                _drop_blend(qb); m = json.load(open(qm)); m["no_blend"] = True; json.dump(m, open(qm, "w"), ensure_ascii=False)
                                for r in rows:
                                    if r["step"] == qn: r["no_blend"] = True
            else:
                fails += 1
        else:
            os.makedirs(outdir, exist_ok=True)
            if os.path.exists(mark): os.remove(mark)
            cmd = argv
            if st.get("cost") == "heavy":
                cmd = [PY, "-m", "core.services.guard", "heavy", "--wait", "1800", "--what", f"{name}:{sname}", "--", *argv]
            t0 = time.time(); before = _scene_state(SD)
            left = sorted({m for a in argv for m in re.findall(r"\{([A-Za-z_]\w*)\}", str(a))})   # a template with no value: the command would have received
            with open(os.path.join(outdir, "step.log"), "w") as log:                       # "{out}" literally (a folder named "{out}" appeared in the root)
                log.write(" ".join(cmd) + "\n"); log.flush()
                if left:
                    log.write(f"no value for {{{', '.join(left)}}}: the knob is not set or the source step did not provide gives — the command was not run\n")
                    p = type("NotRun", (), {"returncode": 2})()
                else:
                    p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
            tail = open(os.path.join(outdir, "step.log"), errors="replace").read().strip().splitlines()[-3:]
            row.update(status="done" if p.returncode == 0 else "failed", code=p.returncode, sec=round(time.time() - t0, 1),
                       tail=[t[-200:] for t in tail], command=cmd)
            _budget(st, profile, row)
            if p.returncode == 0:
                sout = _snapshot(st, SD, outdir, before, row)               # a snapshot of the step's contribution to the run's scene — for the cache
                json.dump({"key": key, "step": sname, "t": row["t"], "sec": row["sec"], "parts": parts, "scene_out": sout, "outputs": _outputs_of(outdir)},
                          open(mark, "w"), ensure_ascii=False)
            else:
                fails += 1
        if row.get("status") in ("done", "cached", "would_run", "skipped") and not (fn_build or plan_only):   # old command path:
            for k, v in (st.get("command_gives") or {}).items():                         # ready values from the passport (the step has no function)
                ctx["gives"][k] = v.format(out=outdir) if isinstance(v, str) else v
        if row.get("status") == "done" and not dry:
            if st.get("is_output"):                                         # frame metrics of an output step — into the journal
                m = _metrics_of(outdir)
                if m: row["metrics"] = m
                fj = os.path.join(outdir, "frames.json")                    # frame cost and contact sheet — to the caller in one line
                F = json.load(open(fj)) if os.path.exists(fj) else {}
                if F.get("sec_mean") is not None: row["sec_per_frame"] = F["sec_mean"]
                if F.get("weight"): row["heaviest"] = [f"{c} {w['tris'] / 1e6:.1f} M tris" + (f", {w['instances']} instances" if w.get("instances") else "") + (f", {w['volumes']} volumes" if w.get("volumes") else "")
                                                       for c, w in list(F["weight"].items())[:3]]
                if F.get("contact"): row["contact"] = os.path.relpath(F["contact"], ROOT)
                if F.get("sec_per_frame"):                                   # the slowest frame — and where the time went (passes: sky, light, effects)
                    k_ = max(F["sec_per_frame"], key=lambda q: F["sec_per_frame"][q]); ps = (F.get("passes") or {}).get(str(k_)) or {}
                    row["slowest"] = f"cam{k_} {F['sec_per_frame'][k_]} s" + (" (" + ", ".join(f"{n} {v}" for n, v in ps.items()) + ")" if ps else "")
        prev.append(key); rows.append(row); journal(row); ctx["steps"].append(sname)
        if not dry: json.dump(ctx, open(ctx["path"], "w"), ensure_ascii=False, indent=1)
        if row.get("status") == "failed" and st.get("on_fail", "stop") == "stop": break
    if tmp: tmp.cleanup()
    if jf: jf.close()
    release()
    if not dry:
        _md(folder, name, scene, profile, rows, warns, until)
    cached_names = [r["step"] for r in rows if r["status"] == "cached"]
    skipped_names = [r["step"] for r in rows if r["status"] == "skipped"]
    off_names = [r["step"] for r in rows if r["status"] == "off"]
    if cached_names:
        nb = [r["step"] for r in rows if r["status"] == "cached" and r.get("no_blend")]
        print("from cache: " + ", ".join(cached_names) + (f" (no .blend: {', '.join(nb)})" if nb else ""))
    if off_names: print(f"disabled at stage {profile}: " + ", ".join(off_names))
    for r in rows:
        if r["status"] in ("cached", "skipped", "off"): continue
        extra = f" {r.get('sec')} s" if "sec" in r else ""
        if r.get("profile_mode") == "plan": extra += " (plan only)"
        pn_ = (r.get("stats") or {}).get("pinned") or (r.get("stats") or {}).get("pinned_frames")
        if pn_: extra += f" (pinned cameras: {pn_})"                   # the sketch stage's decisions (core.services.stage pin)
        if r.get("status") == "done" and (r.get("new_objects") or r.get("wrote")):   # what the step created — to the caller in one line
            w_ = r.get("wrote") or []
            extra += " —" + (f" +{r['new_objects']} obj." if r.get("new_objects") else "") + (f" writes: {', '.join(w_[:5])}{' …' if len(w_) > 5 else ''}" if w_ else "")
        if r.get("scene_undeclared"): extra += f" (outside the passport: {', '.join(r['scene_undeclared'][:4])})"
        print(f"{r['step']:<14} {r['status']}{extra}" + (f" — {' '.join(map(str, r['command']))}" if dry and r.get("command") else "")
              + (f" → {' '.join(r['args'])}" if r.get("args") else "") + (f" (no layer: {', '.join(r['no_layer'])})" if r.get("no_layer") else "")
              + (f" (skipped: {r['skipped_part']})" if r.get("skipped_part") else "")
              + (f" — {r['sec_per_frame']} s/frame; heaviest: {'; '.join(r['heaviest'])}" if r.get("sec_per_frame") is not None and r.get("heaviest") else "")
              + (f"\n    slowest: {r['slowest']}" if r.get("slowest") else "")
              + (f"\n    contact sheet: {r['contact']}" if r.get("contact") else "")
              + (f"\n    {r['tail'][-1]}" if r.get("status") == "failed" and r.get("tail") else ""))
    if skipped_names: print("skipped (--from/--only): " + ", ".join(skipped_names))
    if until: print(f"stopped after step '{until}' (--until)")
    res = {"pipeline": name, "out": os.path.relpath(folder, ROOT), "failures": fails,
                                        "done": sum(r["status"] == "done" for r in rows), "cached": len(cached_names), "until": until,
                                        "last_key": prev[-1] if prev else None, "last_blend": ctx.get("blend"), "gives": dict(ctx["gives"]),
                                        "steps_total": len(rows), "off": off_names,
                                        "all_cached": bool(cached_names) and all(r["status"] in ("cached", "off") for r in rows)}
    if not dry: json.dump(res, open(os.path.join(folder, "result.json"), "w"), ensure_ascii=False, indent=1)
    print("__RESULT__" + json.dumps(res, ensure_ascii=False))
    return fails


def _md(folder, name, scene, profile, rows, warns=(), until=None):
    cached = [r["step"] for r in rows if r["status"] == "cached"]
    skipped = [r["step"] for r in rows if r["status"] == "skipped"]
    L = [f"# Run '{name}' — {os.path.relpath(scene, ROOT)}, profile {profile}, {rows[0]['t'] if rows else ''}", "",
         "| step | component | status | sec | memory GB | .blend MB | key |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["status"] in ("cached", "skipped", "off"): continue               # only what was done; cached and disabled ones go in the lines below
        L.append(f"| {r['step']} | {r['component']} | {r['status']}{' (' + r['over_budget'] + ')' if r.get('over_budget') else ''}"
                 f"{' (no layer: ' + ', '.join(r['no_layer']) + ')' if r.get('no_layer') else ''}"
                 f"{' (skipped: ' + r['skipped_part'] + ')' if r.get('skipped_part') else ''}"
                 f"{' (no .blend)' if r.get('no_blend') else ''} | {r.get('sec', '')} | {r.get('memory_gb', '')} | {r.get('blend_mb', '')} | "
                 f"{r.get('key', ' '.join(r.get('args', [])))} |")
    done = [r for r in rows if r["status"] in ("done", "failed")]
    if done:                                                                    # run cost: total seconds, peak memory
        mem = [r["memory_gb"] for r in done if r.get("memory_gb")]
        L += ["", f"Total done: {len(done)} steps, {round(sum(r.get('sec', 0) for r in done), 1)} s" + (f", peak memory {max(mem)} GB" if mem else "") + "."]
    if cached:
        nb = [r["step"] for r in rows if r["status"] == "cached" and r.get("no_blend")]
        L += ["", "From cache: " + ", ".join(cached) + (f" (no .blend: {', '.join(nb)} — a restart from them is impossible)" if nb else "") + "."]
    if skipped: L += ["", "Skipped (--from/--only): " + ", ".join(skipped) + "."]
    off = [r["step"] for r in rows if r["status"] == "off"]
    if off: L += ["", f"Steps disabled at stage {profile} (the passport's profiles): " + ", ".join(off) + "."]
    nb2 = [r["step"] for r in rows if r.get("no_blend") and r["status"] not in ("cached",)]
    if nb2: L += ["", "Checkpoint .blends removed (checkpoints: last): " + ", ".join(nb2) + " — a restart from these steps rebuilds the chain."]
    if until: L += ["", f"Stopped after step '{until}' (--until)."]
    if warns: L += ["", "Warnings:"] + [f"- {w}" for w in warns]
    open(os.path.join(folder, "journal.md"), "w").write("\n".join(L) + "\n\nAll runs are in journal.jsonl next to it.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="pipeline manager")
    ap.add_argument("conv"); ap.add_argument("--scene"); ap.add_argument("--out"); ap.add_argument("--profile", help="sketch | preview | final")
    ap.add_argument("--dry", action="store_true", help="build nothing: show cached / would_run")
    ap.add_argument("--fresh", action="store_true"); ap.add_argument("--only", help="comma-separated steps")
    ap.add_argument("--from", dest="since", help="start from a step (earlier ones come from cache or from disk)")
    ap.add_argument("--until", help="run up to and including the step, then stop (combines with --from)")
    a = ap.parse_args()
    sys.exit(min(run(a.conv, a.scene, a.out, a.profile, a.dry, a.fresh, a.only.split(",") if a.only else None, a.since, a.until), 100))
