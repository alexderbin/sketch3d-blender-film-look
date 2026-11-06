"""Static import check (level 1, no execution — part of the code starts Blender at import): for every .py under core/,
components/, formats/ (ext, runs, out skipped):
  1) `from core.x.y import NAME` — the module exists and defines NAME (top level, incl. inside if/try/with);
  2) no bare imports of project modules by their old flat names (`import vocabulary`, `import far`) — packages only;
  3) no undefined names (pyflakes): a NameError waiting at run time — e.g. an assignment commented out by an edit (30.10:
     `; tall = []` after a new comment in components/plants/steps.py — the plants step failed in Blender);
  4) modules started by name (`-m <package.module>` in .py, .sh, passports and pipelines) and scripts in passport commands
     (`{root}/<path>.py`) exist — a renamed module breaks such a command only at run time.
Adapted from a review script of session 3d-animation-53 (30.10) after build.py imported a name core.paths no longer had.

    .venv/bin/python -m core.services.import_check        exit code = number of findings
"""
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROOTS = ("core", "components", "formats")
SKIP = ("ext", "__pycache__", "runs", "out", "venv", ".venv", "templates")   # templates — starter templates (core/templates), the names apply after copying
EXTERNAL = set(sys.stdlib_module_names) | {"bpy", "bmesh", "mathutils", "bpy_extras", "gpu", "addon_utils", "idprop", "numpy", "yaml", "PIL", "cv2",
                                            "anthropic", "scipy", "imageio", "trimesh", "fcl", "claude_agent_sdk", "matplotlib", "requests", "skimage",
                                            "shapely", "noise", "pytest", "openai", "torch", "jupedsim", "rich", "tqdm", "sketch25", "ruamel"}


def top_names(path):
    try:
        t = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None
    names = set()
    for n in t.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            for tg in n.targets:
                for x in ast.walk(tg):
                    if isinstance(x, ast.Name):
                        names.add(x.id)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names.add(n.target.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, (ast.If, ast.Try, ast.With)):
            for x in ast.walk(n):
                if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(x.name)
                elif isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
                    names.add(x.id)
                elif isinstance(x, (ast.Import, ast.ImportFrom)):
                    for a in x.names:
                        names.add((a.asname or a.name).split(".")[0])
    return names


def undefined_names(files):
    """pyflakes over the project's files: only «undefined name» (other pyflakes notes are style, not errors)."""
    try:
        from pyflakes import checker
    except ImportError:
        print("WARNING: pyflakes is not installed (requirements.txt) — undefined names were not checked")
        return []
    out = []
    for f in files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        for m in checker.Checker(tree, filename=str(f)).messages:
            if type(m).__name__ == "UndefinedName":
                out.append((f.relative_to(ROOT), m.lineno, m.message_args[0]))
            elif type(m).__name__ in ("MultiValueRepeatedKeyLiteral", "MultiValueRepeatedKeyVariable"):   # dict key twice — the field is lost
                out.append((f.relative_to(ROOT), m.lineno, f"duplicate key {m.message_args[0]!r}"))
    return out


MOD = re.compile(r"""-m["']?\s*,?\s*["']?((?:core|components|formats)(?:\.\w+)+)""")
SCRIPT = re.compile(r"\{root\}/((?:core|components|formats)/[\w/.-]+\.py)")


def started_modules():
    """(file, line, module or script) — modules after -m and {root}/… scripts that do not exist."""
    files = [f for r in ROOTS for f in (ROOT / r).rglob("*") if f.suffix in (".py", ".yaml", ".yml", ".sh") and not any(x in f.parts for x in SKIP)]
    files += list((ROOT / "bench").glob("*.sh")) + list((ROOT / "bench" / "tools").glob("*.sh"))
    out = []
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(lines, 1):
            for m in MOD.finditer(line):
                parts = m.group(1).split(".")
                base = ROOT.joinpath(*parts)
                if not (base.with_suffix(".py").exists() or (base / "__init__.py").exists() or (base / "__main__.py").exists()):
                    out.append((f.relative_to(ROOT), i, "-m " + m.group(1)))
            for m in SCRIPT.finditer(line):
                if not (ROOT / m.group(1)).exists():
                    out.append((f.relative_to(ROOT), i, m.group(1)))
    return out


DIRECT = [re.compile(r"""-P["']?\s*,?\s*["']?(?:\{root\}/)?((?:core|components|formats)/[\w/.-]+\.py)"""),   # -P script (shell, YAML, Python list)
          re.compile(r"""\$BL\s+["']?((?:core|components|formats)/[\w/.-]+\.py)"""),                       # smoke: BL=(blender … -P)
          re.compile(r"""["']-P["'],\s*os\.path\.join\([^)]*?["']((?:core|components|formats)/[\w/.-]+\.py)["']\)""")]
BARE = re.compile(r"""-P["']?\s*,?\s*["']?([\w.-]+\.py)\b""")                                       # -P bl_x.py — next to the file
ENTRY = {"core/blender/run.py", "core/blender/build.py"}            # Blender entry points that set the project paths themselves


def direct_blender():
    """(file, line, script) — a project Blender script launched through -P bypassing core/blender/run.py while itself importing
    core/components/formats (without run.py the project packages are not in sys.path — ModuleNotFoundError, as happened with
    bl_layers in the smoke test). Searches the whole text of the file (a line break after "-P", counts too), a bare script
    name is looked up next to the file; files — packages, bench/**/*.py and *.sh."""
    files = [f for r in ROOTS for f in (ROOT / r).rglob("*") if f.suffix in (".py", ".yaml", ".yml", ".sh") and not any(x in f.parts for x in SKIP)]
    files += [f for f in (ROOT / "bench").rglob("*") if f.suffix in (".py", ".sh") and "scenes" not in f.relative_to(ROOT / "bench").parts[:1]
              and not any(x in f.parts for x in SKIP)]
    out, imports = [], {}
    def uses_project(rel):
        if rel not in imports:
            try:
                t = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
                imports[rel] = any((isinstance(n, ast.Import) and any(a.name.split(".")[0] in ROOTS for a in n.names)) or
                                   (isinstance(n, ast.ImportFrom) and n.level == 0 and n.module and n.module.split(".")[0] in ROOTS) for n in ast.walk(t))
            except (SyntaxError, UnicodeDecodeError, OSError):
                imports[rel] = False
        return imports[rel]
    for f in sorted(set(files)):
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits = [(m.start(), m.group(1)) for rx in DIRECT for m in rx.finditer(text)]
        for m in BARE.finditer(text):                                 # bare name: a script in the file's folder
            near = (f.parent / m.group(1))
            if near.exists(): hits.append((m.start(), str(near.relative_to(ROOT))))
        for pos, rel in hits:
            if rel not in ENTRY and (ROOT / rel).exists() and uses_project(rel):
                out.append((f.relative_to(ROOT), text.count("\n", 0, pos) + 1, rel))
    return sorted(set(out))

# optional modules: game models and rental — they are called lazily and are absent from the public release
# (core/services/publish)
OPTIONAL = ("core.assets.models", "core.services.remote")


def _catches_import(h):
    """The except handler catches ImportError / ModuleNotFoundError — the import inside the try is optional."""
    names = h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]
    return any(isinstance(x, ast.Name) and x.id in ("ImportError", "ModuleNotFoundError") for x in names)


def main():
    files = [f for r in ROOTS for f in (ROOT / r).rglob("*.py") if not any(x in f.parts for x in SKIP)]
    stems = {}
    for f in files:
        stems.setdefault(f.stem, []).append(f)
    missing, bare, cache = [], [], {}
    for f in files:
        try:
            t = ast.parse(f.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        guarded = {id(x) for tn in ast.walk(t) if isinstance(tn, ast.Try) and any(_catches_import(h) for h in tn.handlers)
                   for b in tn.body for x in ast.walk(b)}
        for n in ast.walk(t):
            if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module and n.module.split(".")[0] in ROOTS:
                if id(n) in guarded or n.module.startswith(OPTIONAL):
                    continue
                p = ROOT.joinpath(*n.module.split("."))
                for a in n.names:
                    if a.name == "*":
                        continue
                    if p.is_dir() and ((p / (a.name + ".py")).exists() or (p / a.name).is_dir()):
                        continue
                    src = p.with_suffix(".py") if p.with_suffix(".py").exists() else p / "__init__.py"
                    if not src.exists():
                        missing.append((f.relative_to(ROOT), n.lineno, n.module, a.name, "no module")); continue
                    if src not in cache:
                        cache[src] = top_names(src)
                    if cache[src] is not None and a.name not in cache[src]:
                        missing.append((f.relative_to(ROOT), n.lineno, n.module, a.name, "no name"))
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else ([n.module] if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module else [])
            for m in mods:
                top = m.split(".")[0]
                if top in ROOTS or top in EXTERNAL:
                    continue
                if top in stems:
                    bare.append((f.relative_to(ROOT), n.lineno, m, len(stems[top])))
    for x in missing:
        print(f"IMPORT  {x[0]}:{x[1]}: from {x[2]} import {x[3]} — {x[4]}")
    for x in bare:
        print(f"BARE    {x[0]}:{x[1]}: import {x[2]} — a project module under its old flat name ({x[3]} file(s) with that name)")
    undef = undefined_names(files)
    for x in undef:
        print(f"NAME    {x[0]}:{x[1]}: {x[2]} (pyflakes)" if str(x[2]).startswith("duplicate key") else f"NAME    {x[0]}:{x[1]}: '{x[2]}' is undefined (pyflakes)")
    started = started_modules()
    for x in started:
        print(f"LAUNCH  {x[0]}:{x[1]}: {x[2]} — no such module or file")
    direct = direct_blender()
    for x in direct:
        print(f"BLENDER {x[0]}:{x[1]}: {x[2]} — launched bypassing core/blender/run.py, while the script imports project packages")
    n = len(missing) + len(bare) + len(undef) + len(started) + len(direct)
    print(f"{'OK' if not n else 'FAILURES ' + str(n)}: files {len(files)}, missing names {len(missing)}, bare imports {len(bare)}, undefined names {len(undef)}, "
          f"launches without a module {len(started)}, Blender bypassing run.py {len(direct)}")
    return min(n, 100)


if __name__ == "__main__":
    sys.exit(main())
