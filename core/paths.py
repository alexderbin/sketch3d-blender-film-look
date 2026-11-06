"""Project paths for code: ROOT, CORE, COMPONENTS, FORMATS, ASSETS, EXT, RUN.
Modules import each other by package name (`from core.light import far`); this module only answers "where is …":
    core.paths.ROOT                      project root (the folder that holds core/, components/, formats/)
    core.paths.script("frames.py")       file of a core script by name — for subprocess / `blender -P core/blender/run.py -- <file>`
    core.paths.assets("models")          folder of the workshop (core/assets/models|mocap) — a path, no sys.path change
    core.paths.component("light", "x")   folder of a component (components/light/x) — a path, no sys.path change
    core.paths.module_name(path)         dotted module name of a .py inside the packages, else None
    core.paths.python_target(path)       ["-m", "pkg.mod"] for a packaged .py, [path] otherwise (argv for the project python)
    core.paths.ext("pkg")                an external package under core/ext — the ONE sanctioned sys.path extension: external
    core.paths.component_ext("plants", "mtree", "site")   … code is not packaged (also components/<x>/ext/<pkg>)
    core.paths.add_path(p)               any other external code folder
sys.path is otherwise touched only by the two Blender entry points, core/blender/build.py and core/blender/run.py."""
import os
import sys

CORE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CORE)
COMPONENTS = os.path.join(ROOT, "components")
FORMATS = os.path.join(ROOT, "formats")
ASSETS = os.path.join(CORE, "assets")
EXT = os.path.join(CORE, "ext")
RUN = os.path.join(CORE, "blender", "run.py")
PACKAGES = ("core", "components", "formats")
PARTS = ("model", "contract", "manager", "services", "outputs", "blender", "surface", "assets", "recipes")


# child processes started by path (python <file>.py) must see the packages core/components/formats too: the root goes into
# PYTHONPATH of every process that imported core.paths (Blender ignores PYTHONPATH — its entry is core/blender/run.py).
if ROOT not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    os.environ["PYTHONPATH"] = os.pathsep.join(x for x in (ROOT, os.environ.get("PYTHONPATH", "")) if x)

def add_path(path):
    """Foreign (unpackaged) code folder into sys.path, after the project root. Returns the path."""
    if path not in sys.path:
        sys.path.insert(1, path)
    return path


def ext(*parts):
    """Path of an external package under core/ext/<parts>; added to sys.path (its code is imported by its own names)."""
    return add_path(os.path.join(EXT, *parts))


def component_ext(component, *parts):
    """Path of an external package under components/<component>/ext/<parts>; added to sys.path."""
    return add_path(os.path.join(COMPONENTS, component, "ext", *parts))


def assets(*sub):
    """Path of the workshop core/assets[/models|mocap|…] — a path only."""
    return os.path.join(ASSETS, *sub)


def component(name, *sub):
    """Path of components/<name>[/<sub>] — a path only."""
    return os.path.join(COMPONENTS, name, *sub)


def script(name):
    """File of a core script by name ("bl_pack.py"): in a core part, else in the workshop (models, mocap)."""
    for p in PARTS:
        f = os.path.join(CORE, p, name)
        if os.path.isfile(f):
            return f
    for p in ("models", "mocap"):
        f = os.path.join(ASSETS, p, name)
        if os.path.isfile(f):
            return f
    return os.path.join(ASSETS, name)


def module_name(path):
    """Dotted module name for a .py inside core/, components/ or formats/ (every folder an identifier), else None."""
    p = os.path.normpath(os.path.abspath(path))
    if not p.startswith(ROOT + os.sep) or not p.endswith(".py"):
        return None
    parts = os.path.relpath(p, ROOT)[:-3].split(os.sep)
    if parts[0] not in PACKAGES or "ext" in parts or not all(x.isidentifier() for x in parts):
        return None
    return ".".join(parts)


def python_target(path):
    """argv fragment for the project python: ["-m", "pkg.mod"] for a packaged script, [path] for anything else."""
    name = module_name(path)
    return ["-m", name] if name else [path]


def venv_site():
    """site-packages of the project venv (.venv, Python 3.13 like Blender's) or None."""
    import glob as _glob
    hits = sorted(_glob.glob(os.path.join(ROOT, ".venv", "lib", "python3.*", "site-packages")))
    return hits[-1] if hits else None


def blender_site():
    """Inside Blender: append the venv site-packages so pure-python packages (yaml and the like) import; appended, not
    inserted — Blender's own numpy/PIL stay first. No-op outside Blender or without a venv."""
    site = venv_site()
    if site and site not in sys.path:
        sys.path.append(site)
    return site


SCENES_IN = os.path.join(ROOT, "bench", "scenes")        # inputs of the benchmarks and pipelines — in git (scene descriptions)
SCENES_OUT = os.path.join(ROOT, "out", "dev", "scenes")  # scenes created by the tools are working files, outside git


def scene_dir(name, create=False):
    """A scene folder by name: the benchmark input (bench/scenes) if there is one, otherwise the working one (out/dev/scenes); create — make the working one."""
    for base in (SCENES_IN, SCENES_OUT):
        d = os.path.join(base, name)
        if os.path.isdir(d):
            return d
    d = os.path.join(SCENES_OUT, name)
    if create:
        os.makedirs(d, exist_ok=True)
    return d
