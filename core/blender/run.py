"""Blender entry point for packaged modules (the second of the two sanctioned sys.path bootstraps, with build.py):
    blender -b [--factory-startup] [file.blend] -P core/blender/run.py -- <target> [-- args…]
<target>: a dotted module (core.light.bl_far) or a .py file (absolute, relative to the cwd or to the project root).
The target runs as __main__ (its `if __name__ == "__main__":` block fires) with sys.argv in the shape the scripts already
expect — [blender…, "-P", <target file>, "--", args…]: the launcher's own "-- <target>" is removed. A .py outside the
packages (bench/, research/…) runs by path; it is not packaged, so it imports project code by package names only."""
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)                       # bootstrap: packages core/, components/, formats/

from core.paths import module_name, blender_site  # noqa: E402
blender_site()                                 # venv site-packages: yaml and other pure-Python packages (a single Python 3.13)


def resolve(target):
    """→ (dotted module or None, file path or None)."""
    if not target.endswith(".py") and "/" not in target and os.sep not in target:
        return target, None
    for cand in (target, os.path.join(os.getcwd(), target), os.path.join(ROOT, target)):
        if os.path.isfile(cand):
            cand = os.path.abspath(cand)
            return module_name(cand), cand
    raise SystemExit(f"run.py: target not found: {target}")


def main():
    argv = sys.argv
    # Fork-bomb guard (29.10: 663 Blender processes from a script that relaunched itself inside Blender). A Blender started
    # by this launcher marks its environment; a nested launch that reaches this launcher again stops here.
    if os.environ.get("SK_IN_BLENDER"):
        sys.stderr.write("run.py: nested Blender launch refused — already inside Blender (SK_IN_BLENDER); "
                         "the script must not start Blender when it runs under Blender\n")
        sys.exit(3)
    os.environ["SK_IN_BLENDER"] = "1"
    if "--" not in argv:
        raise SystemExit(__doc__)
    i = argv.index("--")
    if i + 1 >= len(argv):
        raise SystemExit("run.py: a module name or a .py file must follow `--`")
    target, rest = argv[i + 1], argv[i + 2:]
    name, path = resolve(target)
    if rest and rest[0] != "--":
        rest = ["--"] + rest
    shown = path or (name.replace(".", "/") + ".py")
    head = argv[:i]
    if head and head[-1].endswith("run.py"):
        head = head[:-1] + [shown]
    sys.argv = head + rest
    if name:
        runpy.run_module(name, run_name="__main__", alter_sys=True)
    else:
        runpy.run_path(path, run_name="__main__")


if __name__ == "__main__":
    main()
